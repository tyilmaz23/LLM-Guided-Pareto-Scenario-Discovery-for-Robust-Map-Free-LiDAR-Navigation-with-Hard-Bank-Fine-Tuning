import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class CustomEnv(gym.Env):

    GOAL_REACHED_DIST = 0.5
    COLLISION_MARGIN = 0.02

    LIDAR_ANGLES = np.linspace(0, 2 * np.pi, 120, endpoint=False)

    # --------------------------------------------------
    # OBS vs TRUE lidar çözünürlükleri
    # --------------------------------------------------
    LIDAR_SAMPLES_OBS  = 100   # policy'nin gördüğü
    LIDAR_SAMPLES_TRUE = 200  # collision / metric için

    def __init__(self):
        super().__init__()

        self.rng = np.random.default_rng()

        # ---------------- action space ----------------
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32)
        )

        # ---------------- observation space ----------------
        self.observation_space = spaces.Box(
            low=np.array([-1]*6 + [-1]*120, dtype=np.float32),
            high=np.array([ 1]*6 + [ 1]*120, dtype=np.float32)
        )

        # ---------------- robot params ----------------
        self.wall_thickness = 0.30
        self.max_steps = 1000

        self.D = 66e-3
        self.R = self.D / 2
        self.L = 160e-3

        self.agent_pos = np.zeros(3, dtype=np.float32)
        self.target_pos = np.zeros(2, dtype=np.float32)

        self.distance_buffer = deque(maxlen=5)

        self.walls = []
        self.dynamic_walls = []

        self.prev_action = np.zeros(2, dtype=np.float32)

        # ---------------- reward params ----------------
        self.DANGER_THRESH = 0.23

        self.scenario = None
        self.sensor_cfg = {
            "lidar_noise_sigma": 0.0,
            "dropout_prob": 0.0,
            "range_bias": 0.0
        }
        self._scenario_rng = np.random.default_rng(123)

    # ==================================================
    # KINEMATICS
    # ==================================================
    def ilerikin(self, wR, wL):
        v = 0.5 * self.R * (wR + wL)
        w = (self.R / self.L) * (wR - wL)
        return v, w

    def terskin(self, v, w):
        wL = (v - w * self.L / 2) / self.R
        wR = (v + w * self.L / 2) / self.R
        return wL, wR

    # ==================================================
    # LIDAR
    # ==================================================
    def _lidar_cast(self, n_samples):
        distances = np.linspace(0.01, 1.00, n_samples, dtype=np.float32)
        out = np.full(len(self.LIDAR_ANGLES), 1.00, dtype=np.float32)

        ax, ay, ath = self.agent_pos

        for i, ang in enumerate(self.LIDAR_ANGLES):
            dx = distances * np.cos(ath + ang)
            dy = distances * np.sin(ath + ang)
            px = ax + dx
            py = ay + dy

            for w in self.walls + self.dynamic_walls:
                hit = (w[0] <= px) & (px <= w[2]) & (w[1] <= py) & (py <= w[3])
                if np.any(hit):
                    out[i] = distances[np.argmax(hit)]
                    break
        return out

    def _apply_sensor_corruption(self, x):
        cfg = self.sensor_cfg

        if cfg["lidar_noise_sigma"] > 0:
            x += self._scenario_rng.normal(0, cfg["lidar_noise_sigma"], x.shape)

        if cfg["range_bias"] != 0:
            x += cfg["range_bias"]

        if cfg["dropout_prob"] > 0:
            mask = self._scenario_rng.random(x.shape) < cfg["dropout_prob"]
            x[mask] = 1.00

        return np.clip(x, 0.16, 1.00)

    def lidar_true(self):
        return self._lidar_cast(self.LIDAR_SAMPLES_TRUE)

    def lidar_obs(self):
        x = self._lidar_cast(self.LIDAR_SAMPLES_OBS)
        x = np.clip(x, 0.16, 1.00)
        return self._apply_sensor_corruption(x).astype(np.float32)

    # ==================================================
    # SCENARIO INJECTION (CRITICAL)
    # ==================================================
    def set_scenario(self, scenario):
        self.scenario = scenario

        if scenario is None:
            self.walls = []
            self.dynamic_walls = []
            self.sensor_cfg = {
                "lidar_noise_sigma": 0.0,
                "dropout_prob": 0.0,
                "range_bias": 0.0
            }
            return

        self.walls = [list(map(float, w)) for w in scenario.get("walls", [])]
        self.dynamic_walls = [list(map(float, w)) for w in scenario.get("dynamic_walls", [])]


        # --- SEED PROPAGATION (REPRODUCIBILITY) ---
        if "seed" in scenario:
            self._scenario_rng = np.random.default_rng(int(scenario["seed"]))
        else:
            self._scenario_rng = self.rng

        # --- SENSOR CONFIG ---
        self.sensor_cfg = scenario.get("sensor_cfg", {
            "lidar_noise_sigma": 0.0,
            "dropout_prob": 0.0,
            "range_bias": 0.0,
        })

    # ==================================================
    # RESET
    # ==================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.scenario is None:
            # güvenli default (stale state engeli)
            self.walls = []
            self.dynamic_walls = []
            self.sensor_cfg = {"lidar_noise_sigma": 0.0, "dropout_prob": 0.0, "range_bias": 0.0}
            self._scenario_rng = self.rng



        if self.scenario is not None:
            a = self.scenario["agent"]
            t = self.scenario["target"]

            self.agent_pos = np.array(
                [a["x"], a["y"], a.get("theta", 0.0)],
                dtype=np.float32
            )
            self.target_pos = np.array(
                [t["x"], t["y"]],
                dtype=np.float32
            )
        else:
            self.agent_pos = np.array([
                self._scenario_rng.uniform(-3, 3),
                self._scenario_rng.uniform(-3, 3),
                self._scenario_rng.uniform(0, 2*np.pi)
            ], dtype=np.float32)

            self.target_pos = np.array([
                self._scenario_rng.uniform(-4, 4),
                self._scenario_rng.uniform(-4, 4)
            ], dtype=np.float32)


        dist = np.linalg.norm(self.agent_pos[:2] - self.target_pos)
        self.distance_buffer = deque([dist]*5, maxlen=5)

        self.step_count = 0
        self.prev_action[:] = 0.0

        lidar = self.lidar_obs()
        lidar_norm = ((lidar - 0.16) / (1.00 - 0.16)) * 2 - 1

        theta = self.calculate_theta_to_target(self.agent_pos, self.target_pos)

        state = np.concatenate([
            [np.sin(theta), np.cos(theta), 0, 0, dist/100, 0],
            lidar_norm
        ]).astype(np.float32)

        return state, {}

    # ==================================================
    # STEP
    # ==================================================
    def step(self, action):
        info = {}

        pre_min = np.min(self.lidar_true())

        lin_v = (action[0] + 1) * 0.11
        ang_v = action[1] * 2.84

        wL, wR = self.terskin(lin_v, ang_v)
        v, w = self.ilerikin(wR, wL)

        dt = 0.2
        self.agent_pos[0] += v * np.cos(self.agent_pos[2]) * dt
        self.agent_pos[1] += v * np.sin(self.agent_pos[2]) * dt
        self.agent_pos[2] = (self.agent_pos[2] + w * dt) % (2*np.pi)

        dist = np.linalg.norm(self.agent_pos[:2] - self.target_pos)
        self.distance_buffer.append(dist)
        progress = self.distance_buffer[0] - self.distance_buffer[-1]

        post_min = np.min(self.lidar_true())
        min_lidar = min(pre_min, post_min)

        target = dist < self.GOAL_REACHED_DIST
        collision = min_lidar < (self.R + self.COLLISION_MARGIN)
        timeout = self.step_count >= self.max_steps

        reward = self._reward(dist, target, collision, lin_v, ang_v, min_lidar, progress)

        lidar = self.lidar_obs()
        lidar_norm = ((lidar - 0.16) / (1.00 - 0.16)) * 2 - 1

        theta = self.calculate_theta_to_target(self.agent_pos, self.target_pos)

        state = np.concatenate([
            [np.sin(theta), np.cos(theta),
             (lin_v/0.22)*2-1, ang_v/2.84,
             dist/100, progress/100],
            lidar_norm
        ]).astype(np.float32)

        self.step_count += 1

        terminated = target or collision
        truncated = timeout

        if target:
            info["reason"] = "target_reached"
        elif collision:
            info["reason"] = "collision"
        elif timeout:
            info["reason"] = "timeout"

        info["distance_to_target"] = float(dist)
        info["min_lidar"] = float(min_lidar)
        info["min_lidar_obs"] = float(np.min(lidar))

        # --- scenario metadata (paper/log için) ---
        if self.scenario is not None:
            info["scenario_id"] = self.scenario.get("scenario_id", None)
            info["scenario_seed"] = self.scenario.get("seed", None)

        return state, float(reward), terminated, truncated, info




    # ==================================================
    def calculate_theta_to_target(self, a, t):
        ang = np.arctan2(t[1]-a[1], t[0]-a[0]) - a[2]
        return (ang + np.pi) % (2*np.pi) - np.pi

    def _reward(self, d, reached, collision, v, w, min_lidar, prog):
        if reached:
            return 1.0
        if collision:
            return -1.0

        r = (0.0006 / max(1e-6, d)) \
            - 0.00035 * (self.step_count / self.max_steps) \
            - abs(w) * 0.00018 \
            + 0.0004 * prog \
            + v * 0.00022
        return r

