import os
import json
import numpy as np
from collections import Counter

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scenario_dsl import load_scenario, validate_scenario
from ilkkisim import CustomEnv

# ================= GLOBALS =================
NEAR_MISS_THRESH = 0.25
MIN_NORM_STEPS = 50
EARLY_CRASH_STEPS = 50
BOOTSTRAP_N = 1000

# ================= ENV =====================
def make_env():
    return CustomEnv()

def load_model(model_path, vecnorm_path):
    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(model_path, env=venv)
    return model, venv

# ================= BASELINES ===============
def random_policy(obs):
    return np.random.uniform(-1, 1, size=(2,))

def greedy_policy(obs):
    # sin(theta), cos(theta) → heading
    turn = obs[0][0]
    return np.array([0.6, np.clip(turn, -1, 1)])

# ================= EPISODE =================
def run_one_episode(model, venv, policy="ppo"):
    obs = venv.reset()
    min_lidar = 1e9
    near_miss = 0
    steps = 0
    ep_return = 0.0
    start_dist = None
    last_dist = None
    actions = []

    reason = "unknown"

    while True:
        if policy == "ppo":
            action, _ = model.predict(obs, deterministic=True)
        elif policy == "random":
            action = random_policy(obs)
        else:
            action = greedy_policy(obs)

        obs, reward, dones, infos = venv.step(action)
        r = float(reward[0])
        ep_return += r
        info = infos[0]
        actions.append(action)

        d = float(info.get("distance_to_target", np.nan))
        ml = float(info.get("min_lidar", np.nan))

        if start_dist is None and np.isfinite(d):
            start_dist = d
        if np.isfinite(d):
            last_dist = d

        if np.isfinite(ml):
            min_lidar = min(min_lidar, ml)
            if ml < NEAR_MISS_THRESH:
                near_miss += 1

        steps += 1
        if dones[0]:
            reason = info.get("reason", "done")
            break
        if steps > 2000:
            reason = "force_break"
            break

    progress = 0.0
    if start_dist is not None and last_dist is not None:
        progress = start_dist - last_dist

    actions = np.array(actions)
    action_var = float(np.var(actions)) if len(actions) > 5 else 0.0

    return {
        "reason": reason,
        "steps": steps,
        "return": ep_return,
        "min_lidar": min_lidar,
        "near_miss": near_miss,
        "progress": progress,
        "start_dist": start_dist,
        "final_dist": last_dist,
        "action_variance": action_var,
    }

# ================= FAILURE TAXONOMY =========
def classify_failure(e):
    if "collision" in e["reason"] and e["steps"] <= EARLY_CRASH_STEPS:
        return "early_collision"
    if "collision" in e["reason"]:
        return "late_collision"
    if e["progress"] < 0.1:
        return "stuck"
    return "other"

# ================= BOOTSTRAP =================
def bootstrap_ci(x, n=BOOTSTRAP_N):
    x = np.array(x)
    means = []
    for _ in range(n):
        sample = np.random.choice(x, size=len(x), replace=True)
        means.append(np.mean(sample))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# ================= AGGREGATE =================
def aggregate(episodes, max_steps=1000):
    n = len(episodes)
    if n == 0:
        return {}

    returns = np.array([e["return"] for e in episodes])
    steps = np.array([max(1, e["steps"]) for e in episodes])
    action_vars = np.array([e["action_variance"] for e in episodes])

    success = sum("target_reached" in e["reason"] for e in episodes)
    collision = sum("collision" in e["reason"] for e in episodes)
    timeout = sum("timeout" in e["reason"] or "max_steps" in e["reason"] for e in episodes)

    success_rate = success / n
    collision_rate = collision / n
    timeout_rate = timeout / n

    near = np.array([e["near_miss"] for e in episodes])
    near_rate = near / np.maximum(steps, MIN_NORM_STEPS)

    avg_minlid = float(np.mean([e["min_lidar"] for e in episodes]))

    unfinished = float(np.mean([
        (e["final_dist"] or 0.0) / ((e["start_dist"] or 1.0) + 1e-6)
        for e in episodes
        if e["start_dist"] is not None and e["final_dist"] is not None
    ]))

    early_crash_rate = float(np.mean([
        ("collision" in e["reason"]) and (e["steps"] <= EARLY_CRASH_STEPS)
        for e in episodes
    ]))

    difficulty = (
        2.0 * (1.0 - success_rate)
        + 1.5 * collision_rate
        + 0.8 * timeout_rate
        + unfinished
        + max(0.0, 0.35 - avg_minlid)
        + np.mean(near_rate)
    )

    survival = float(np.clip(np.mean(steps) / max_steps, 0.0, 1.0))
    difficulty_weighted = difficulty * (0.25 + 0.75 * survival) * (1.0 - 0.35 * early_crash_rate)

    return_var = float(np.var(returns))
    steps_var = float(np.var(steps))

    collapse_score = float(
        (1.0 - np.tanh(return_var * 10.0)) *
        (1.0 - success_rate) *
        np.tanh(1.0 / (steps_var + 1e-6))
    )

    failures = Counter(classify_failure(e) for e in episodes)

    ci_low, ci_high = bootstrap_ci(returns)

    return {
        "n": n,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "avg_min_lidar": avg_minlid,
        "unfinished_ratio": unfinished,
        "early_crash_rate": early_crash_rate,
        "avg_return": float(np.mean(returns)),
        "return_ci95": [ci_low, ci_high],
        "return_variance": return_var,
        "action_variance": float(np.mean(action_vars)),
        "steps_variance": steps_var,
        "difficulty": difficulty,
        "difficulty_weighted": difficulty_weighted,
        "collapse_score": collapse_score,
        "failure_modes": dict(failures),
    }

# ================= MAIN EVAL =================
def evaluate_scenario(model, venv, scenario, repeats=10, seed_base=1000, policy="ppo"):
    validate_scenario(scenario)
    max_steps = int(venv.get_attr("max_steps")[0])
    episodes = []

    try:
        if hasattr(venv.envs[0], "set_scenario"):
            venv.env_method("set_scenario", scenario)

        for i in range(repeats):
            if hasattr(venv.envs[0], "set_scenario"):
                sc = dict(scenario)
                sc["seed"] = seed_base + i
                venv.env_method("set_scenario", sc)

            episodes.append(run_one_episode(model, venv, policy=policy))
    finally:
        if hasattr(venv.envs[0], "set_scenario"):
            venv.env_method("set_scenario", None)

    return episodes, aggregate(episodes, max_steps=max_steps)

