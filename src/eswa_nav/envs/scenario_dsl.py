import json
import math
import numpy as np

WORLD_MIN, WORLD_MAX = -5.0, 5.0
WALL_THICKNESS = 0.30

# scenario_dsl.py (üstlere, sabitlerin altına ekle)
import math

def aabb_point_dist(px, py, rect):
    """
    Point-to-AABB (axis-aligned rectangle) minimum distance.
    rect = [x1,y1,x2,y2] with x1<=x2, y1<=y2
    """
    x1, y1, x2, y2 = rect
    dx = 0.0
    if px < x1: dx = x1 - px
    elif px > x2: dx = px - x2
    dy = 0.0
    if py < y1: dy = y1 - py
    elif py > y2: dy = py - y2
    return math.hypot(dx, dy)

def normalize_rect(w):
    x1, y1, x2, y2 = map(float, w)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def validate_scenario(sc: dict) -> None:
    assert "scenario_id" in sc
    assert "seed" in sc
    assert "agent" in sc and "target" in sc
    assert "dynamic_walls" in sc

    ax, ay = float(sc["agent"]["x"]), float(sc["agent"]["y"])
    tx, ty = float(sc["target"]["x"]), float(sc["target"]["y"])

    assert WORLD_MIN < ax < WORLD_MAX and WORLD_MIN < ay < WORLD_MAX
    assert WORLD_MIN < tx < WORLD_MAX and WORLD_MIN < ty < WORLD_MAX

    # duvarları normalize edip kontrol et
    norm_walls = []
    for w in sc["dynamic_walls"]:
        w = normalize_rect(w)
        x1, y1, x2, y2 = w

        assert WORLD_MIN <= x1 <= WORLD_MAX and WORLD_MIN <= x2 <= WORLD_MAX
        assert WORLD_MIN <= y1 <= WORLD_MAX and WORLD_MIN <= y2 <= WORLD_MAX
        assert (abs(x2 - x1) > 1e-6) and (abs(y2 - y1) > 1e-6)

        norm_walls.append(w)

    # === YENİ: agent/target duvardan clearance ===
    # Not: env'de wall_thickness=0.30, collision_dist=0.17.
    # 0.30 gayet iyi bir “spawn validity” eşiği.
    CLEAR = 0.20

    for w in norm_walls:
        da = aabb_point_dist(ax, ay, w)
        dt = aabb_point_dist(tx, ty, w)
        assert da >= CLEAR, f"agent too close to wall: dist={da:.3f}, wall={w}"
        assert dt >= CLEAR, f"target too close to wall: dist={dt:.3f}, wall={w}"

    # İstersen normalize edilmiş duvarları senaryoya geri yaz:
    sc["dynamic_walls"] = norm_walls


def scenario_u_trap(scenario_id: str, seed: int, center=(0.0, 0.0),
                    arm_len=2.0, gap=0.55, thickness=0.15, rotation_deg=0.0,
                    agent=(-3.5, -3.5, 1.57), target=(3.5, 3.5),
                    sensor=None):
    """
    Axis-aligned walls ile yaklaşık U-trap (rotation=0 iken net).
    rotation_deg != 0 verirsen "yaklaşık" olur (basitlik için).
    """
    cx, cy = center
    rot = math.radians(rotation_deg)
    # Basit: rotation yokken 3 duvar
    # U: iki dikey + bir yatay
    # gap U'nun ağzı
    xL = cx - gap/2 - thickness
    xR = cx + gap/2
    y0 = cy - arm_len/2
    y1 = cy + arm_len/2
    # sol kol
    w1 = [xL, y0, xL+thickness, y1]
    # sağ kol
    w2 = [xR, y0, xR+thickness, y1]
    # dip
    w3 = [xL, y0, xR+thickness, y0+thickness]

    sc = {
        "scenario_id": scenario_id,
        "seed": int(seed),
        "agent": {"x": float(agent[0]), "y": float(agent[1]), "theta": float(agent[2])},
        "target": {"x": float(target[0]), "y": float(target[1])},
        "dynamic_walls": [w1, w2, w3],
        "sensor_cfg": sensor or {"lidar_noise_sigma": 0.0, "dropout_prob": 0.0, "range_bias": 0.0}
    }
    validate_scenario(sc)
    return sc

def scenario_corridor(
    scenario_id: str,
    seed: int,
    y=0.0,
    gap=0.55,
    thickness=0.15,
    agent=(-4.0, 0.0, 0.0),
    target=(4.0, 0.0),
    sensor=None,
    x_margin=0.50,
):
    ax, ay, ath = float(agent[0]), float(agent[1]), float(agent[2])
    tx, ty = float(target[0]), float(target[1])

    # koridor merkezi: y parametresi (istersen ay/ty ile aynı verirsin)
    y = float(y)

    # DUVAR UZUNLUĞU: agent->target hattını kapsasın
    x0 = min(ax, tx) - float(x_margin)
    x1 = max(ax, tx) + float(x_margin)

    y1 = y - gap/2 - thickness
    y2 = y + gap/2

    w1 = [x0, y1, x1, y1 + thickness]
    w2 = [x0, y2, x1, y2 + thickness]

    sc = {
        "scenario_id": scenario_id,
        "seed": int(seed),
        "agent": {"x": ax, "y": ay, "theta": ath},
        "target": {"x": tx, "y": ty},
        "dynamic_walls": [w1, w2],
        "sensor_cfg": sensor or {"lidar_noise_sigma": 0.0, "dropout_prob": 0.0, "range_bias": 0.0},
    }
    validate_scenario(sc)
    return sc


def scenario_clutter(scenario_id: str, seed: int, n=12,
                     agent=(-3.5, -3.5, 1.57), target=(3.5, 3.5),
                     sensor=None):
    rng = np.random.default_rng(int(seed))
    walls = []

    ax, ay = float(agent[0]), float(agent[1])
    tx, ty = float(target[0]), float(target[1])

    CLEAR = 0.30               # agent/target clearance
    MAX_TRIES = n * 400        # bulamazsa patlatmayalım diye yüksek

    def center(w):
        return ((w[0]+w[2])*0.5, (w[1]+w[3])*0.5)

    def center_dist(w1, w2):
        x1, y1 = center(w1)
        x2, y2 = center(w2)
        return float(np.hypot(x1-x2, y1-y2))

    tries = 0
    while len(walls) < n and tries < MAX_TRIES:
        tries += 1

        if len(walls) % 2 == 0:
            L = float(rng.uniform(0.5, 3.0))
            x_start = float(rng.uniform(-5 + WALL_THICKNESS, 5 - L))
            y_pos   = float(rng.uniform(-5 + WALL_THICKNESS, 5 - WALL_THICKNESS))
            thick   = float(rng.uniform(0.10, 0.30))
            cand = [x_start, y_pos, x_start + L, y_pos + thick]
        else:
            L = float(rng.uniform(0.5, 3.0))
            y_start = float(rng.uniform(-5 + WALL_THICKNESS, 5 - L))
            x_pos   = float(rng.uniform(-5 + WALL_THICKNESS, 5 - WALL_THICKNESS))
            thick   = float(rng.uniform(0.10, 0.30))
            cand = [x_pos, y_start, x_pos + thick, y_start + L]

        cand = normalize_rect(cand)

        # 1) agent/target clearance
        if aabb_point_dist(ax, ay, cand) < CLEAR:
            continue
        if aabb_point_dist(tx, ty, cand) < CLEAR:
            continue

        # 2) duvarlar birbirine çok yığılmasın (env'deki 1.0 center mesafesine benzer)
        ok = True
        for w in walls:
            if center_dist(cand, w) < 1.0:
                ok = False
                break
        if not ok:
            continue

        walls.append(cand)

    if len(walls) < n:
        raise RuntimeError(f"scenario_clutter could not place {n} walls after {MAX_TRIES} tries")

    sc = {
        "scenario_id": scenario_id,
        "seed": int(seed),
        "agent": {"x": ax, "y": ay, "theta": float(agent[2])},
        "target": {"x": tx, "y": ty},
        "dynamic_walls": walls,
        "sensor_cfg": sensor or {"lidar_noise_sigma": 0.0, "dropout_prob": 0.0, "range_bias": 0.0}
    }
    validate_scenario(sc)
    return sc

def save_scenario(sc: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sc, f, ensure_ascii=False, indent=2)

def load_scenario(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

