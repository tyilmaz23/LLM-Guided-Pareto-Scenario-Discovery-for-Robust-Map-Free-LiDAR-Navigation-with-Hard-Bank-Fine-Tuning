# evaluate_iid.py
import os
import json
import numpy as np

from scenario_dsl import (
    scenario_u_trap,
    scenario_corridor,
    scenario_clutter,
    save_scenario
)
from evaluate_scenario import load_model, evaluate_scenario

# =========================================================
# GLOBALS
# =========================================================
np.random.seed(0)

OUT_DIR = "scenarios/iid_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_SCENARIOS = 300
REPEATS = 5

# =========================================================
# IID SCENARIO SAMPLER
# =========================================================
def propose_iid(seed: int):
    rng = np.random.default_rng(seed)

    # IID sensor = training-like (NO amplification)
    sensor = {
        "lidar_noise_sigma": float(rng.uniform(0.0, 0.01)),
        "dropout_prob": float(rng.uniform(0.0, 0.02)),
        "range_bias": float(rng.uniform(-0.005, 0.005)),
    }

    pick = int(rng.integers(0, 3))

    if pick == 0:
        return scenario_u_trap(
            f"IID_U_{seed}", seed,
            center=(0.0, 0.0),
            arm_len=rng.uniform(1.5, 2.2),
            gap=rng.uniform(0.65, 0.85),
            thickness=rng.uniform(0.12, 0.20),
            agent=(-3.5, -3.5, 1.57),
            target=(3.5, 3.5),
            sensor=sensor
        )

    elif pick == 1:
        y = rng.uniform(-0.5, 0.5)
        return scenario_corridor(
            f"IID_C_{seed}", seed,
            y=y,
            gap=rng.uniform(0.65, 0.9),
            thickness=rng.uniform(0.12, 0.20),
            agent=(-4, y, 0.0),
            target=(4, y),
            sensor=sensor
        )

    else:
        return scenario_clutter(
            f"IID_CL_{seed}", seed,
            n=int(rng.integers(8, 14)),
            agent=(-3.5, -3.5, 1.57),
            target=(3.5, 3.5),
            sensor=sensor
        )

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    model, venv = load_model(
        "checkpoints_rl_only/ppo_static_shaping_model.zip",
        "checkpoints_rl_only/ppo_static_shaping_vecnorm.pkl"
    )

    results = []

    for i in range(N_SCENARIOS):
        seed = 1000 + i
        sc = propose_iid(seed)

        episodes, agg = evaluate_scenario(
            model,
            venv,
            sc,
            repeats=REPEATS,
            seed_base=seed * 10
        )

        save_scenario(sc, f"{OUT_DIR}/iid_{i:03d}.json")
        results.append({
            "scenario_id": sc["scenario_id"],
            **agg
        })

        print(f"[IID {i+1}/{N_SCENARIOS}] success={agg['success_rate']:.2f}")

    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("DONE — IID baseline evaluation completed.")

