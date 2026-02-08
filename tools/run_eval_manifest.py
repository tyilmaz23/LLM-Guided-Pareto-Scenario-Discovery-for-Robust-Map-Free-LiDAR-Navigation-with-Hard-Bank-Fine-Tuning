# run_eval_manifest.py
import os
import json
import math
import argparse
import hashlib
from copy import deepcopy
from typing import Dict, List, Any, Tuple

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ilkkisim import CustomEnv


# -------------------------
# hashing / utils
# -------------------------
def stable_u32(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def mean_ci95(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Normal approx CI95: mean ± 1.96 * std/sqrt(n)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    mu = float(np.mean(x))
    if x.size == 1:
        return mu, mu, mu
    sd = float(np.std(x, ddof=1))
    se = sd / math.sqrt(x.size)
    lo = mu - 1.96 * se
    hi = mu + 1.96 * se
    return mu, lo, hi


# -------------------------
# env / model loading
# -------------------------
def make_eval_venv(vecnorm_path: str):
    """
    Single-env evaluation wrapper, VecNormalize loaded from training stats.
    We set training=False and norm_reward=False to report raw returns.
    """
    env = DummyVecEnv([lambda: Monitor(CustomEnv())])
    venv = VecNormalize.load(vecnorm_path, env)

    # Evaluation mode: do not update running stats
    venv.training = False

    # obs norm zaten False idi; ama vecnorm dosyası ne içeriyorsa onu korur.
    # Reward'u raporlarken raw reward istiyoruz.
    venv.norm_reward = False
    return venv

def load_policy(model_path: str, vecnorm_path: str):
    venv = make_eval_venv(vecnorm_path)
    model = PPO.load(model_path, env=venv, device="auto", print_system_info=False)
    return model, venv


# -------------------------
# scenario evaluation
# -------------------------
def set_scenario_for_repeat(venv, scenario: dict, scenario_file: str, rep: int) -> dict:
    """
    Per repeat deterministic sensor-noise seed:
    scenario["seed"] is overwritten (only affects sensor RNG in your CustomEnv).
    Walls/agent/target stay same.
    """
    sc = deepcopy(scenario)
    rep_seed = stable_u32(f"{os.path.normpath(scenario_file)}|rep={rep}")
    sc["seed"] = int(rep_seed)
    # VecEnv -> underlying CustomEnv.set_scenario
    venv.env_method("set_scenario", sc)
    return sc

def run_one_episode(model, venv, deterministic: bool = True, max_steps_guard: int = 5000) -> Dict[str, Any]:
    obs = venv.reset()
    done = False
    ep_ret = 0.0
    steps = 0

    # track min lidar (true) if provided in info
    min_lidar_series = []

    reason = None

    while not done and steps < max_steps_guard:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, dones, infos = venv.step(action)

        r = float(reward[0]) if isinstance(reward, (list, np.ndarray)) else float(reward)
        ep_ret += r
        done = bool(dones[0]) if isinstance(dones, (list, np.ndarray)) else bool(dones)

        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        if isinstance(info0, dict):
            if "min_lidar" in info0:
                try:
                    min_lidar_series.append(float(info0["min_lidar"]))
                except Exception:
                    pass
            if done:
                reason = info0.get("reason", None)

        steps += 1

    if reason is None:
        # fallback
        reason = "unknown"

    ep = {
        "return": float(ep_ret),
        "steps": int(steps),
        "reason": reason,
        "min_lidar_min": float(np.min(min_lidar_series)) if len(min_lidar_series) else float("nan"),
        "min_lidar_mean": float(np.mean(min_lidar_series)) if len(min_lidar_series) else float("nan"),
    }
    return ep

def agg_from_episodes(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(eps)
    reasons = [e["reason"] for e in eps]

    success = sum(1 for r in reasons if r == "target_reached")
    collision = sum(1 for r in reasons if r == "collision")
    timeout = sum(1 for r in reasons if r in ("timeout", "max_steps_reached"))

    rets = np.array([e["return"] for e in eps], dtype=np.float64)
    minlid = np.array([e["min_lidar_min"] for e in eps], dtype=np.float64)
    steps = np.array([e["steps"] for e in eps], dtype=np.float64)

    mu, lo, hi = mean_ci95(rets)

    out = {
        "n_episodes": n,
        "success_rate": success / n if n else float("nan"),
        "collision_rate": collision / n if n else float("nan"),
        "timeout_rate": timeout / n if n else float("nan"),
        "avg_return": float(np.mean(rets)) if n else float("nan"),
        "return_ci95": [float(lo), float(hi)],
        "avg_min_lidar": float(np.nanmean(minlid)) if n else float("nan"),
        "avg_steps": float(np.mean(steps)) if n else float("nan"),
        "failure_modes": {
            "target_reached": success,
            "collision": collision,
            "timeout": timeout,
            "other": n - success - collision - timeout
        }
    }
    return out

def evaluate_file_list(
    model, venv,
    scenario_files: List[str],
    repeats: int,
    deterministic: bool = True,
    tag: str = ""
) -> Dict[str, Any]:
    per_scenario = []
    all_eps = []

    for i, fpath in enumerate(scenario_files, start=1):
        sc0 = load_json(fpath)
        sc_id = sc0.get("scenario_id", os.path.basename(fpath))

        eps = []
        for rep in range(repeats):
            sc_used = set_scenario_for_repeat(venv, sc0, fpath, rep)

            ep = run_one_episode(model, venv, deterministic=deterministic)
            ep["scenario_file"] = os.path.normpath(fpath)
            ep["scenario_id"] = sc_id
            ep["scenario_seed_used"] = sc_used.get("seed", None)
            eps.append(ep)
            all_eps.append(ep)

        agg = agg_from_episodes(eps)
        row = {
            "scenario_id": sc_id,
            "scenario_file": os.path.normpath(fpath),
            "repeats": repeats,
            **agg
        }
        per_scenario.append(row)

        if (i % 25) == 0 or i == len(scenario_files):
            print(f"[{tag}] {i}/{len(scenario_files)} scenarios done.")

    overall = agg_from_episodes(all_eps)

    return {
        "per_scenario": per_scenario,
        "episodes": all_eps,
        "overall": overall
    }


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest_nollm", required=True, help="checkpoints_robust_nollm/robust_manifest.json")
    ap.add_argument("--manifest_llm", required=True, help="checkpoints_robust_llm/robust_manifest.json")
    ap.add_argument("--out_dir", default="eval_results", help="Output directory")

    ap.add_argument("--base_model", default="checkpoints_rl_only/ppo_static_shaping_model.zip")
    ap.add_argument("--base_vecnorm", default="checkpoints_rl_only/ppo_static_shaping_vecnorm.pkl")

    ap.add_argument("--robust_nollm_model", default="checkpoints_robust_nollm/ppo_robust_final_model.zip")
    ap.add_argument("--robust_nollm_vecnorm", default="checkpoints_robust_nollm/ppo_robust_final_vecnorm.pkl")

    ap.add_argument("--robust_llm_model", default="checkpoints_robust_llm/ppo_robust_final_model.zip")
    ap.add_argument("--robust_llm_vecnorm", default="checkpoints_robust_llm/ppo_robust_final_vecnorm.pkl")

    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic actions (recommended for fair eval)")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    man_nollm = load_json(args.manifest_nollm)
    man_llm = load_json(args.manifest_llm)

    # Benchmarks (adil kıyas)
    iid_test_files = man_nollm["iid_test_files"]  # aynı listeler zaten
    nollm_hard_test = man_nollm["hard_test_files"]
    llm_hard_test = man_llm["hard_test_files"]

    benchmarks = {
        "iid_test": iid_test_files,
        "hard_nollm_test": nollm_hard_test,
        "hard_llm_test": llm_hard_test,
    }

    models = {
        "base": (args.base_model, args.base_vecnorm),
        "robust_nollm": (args.robust_nollm_model, args.robust_nollm_vecnorm),
        "robust_llm": (args.robust_llm_model, args.robust_llm_vecnorm),
    }

    # Print quick sanity
    print("[*] Benchmarks:")
    for k, v in benchmarks.items():
        print(f"  - {k}: {len(v)} scenario files")

    print("[*] Models:")
    for k, (mp, vp) in models.items():
        print(f"  - {k}: {mp} | {vp}")

    summary = {
        "repeats": args.repeats,
        "deterministic": bool(args.deterministic),
        "benchmarks": {k: len(v) for k, v in benchmarks.items()},
        "results": {}
    }

    # Evaluate
    for model_tag, (model_path, vec_path) in models.items():
        print("\n==============================")
        print(f"LOADING MODEL: {model_tag}")
        print("==============================")
        model, venv = load_policy(model_path, vec_path)

        for bench_tag, files in benchmarks.items():
            print("\n------------------------------")
            print(f"EVAL: model={model_tag} | bench={bench_tag} | scenarios={len(files)} | repeats={args.repeats}")
            print("------------------------------")

            res = evaluate_file_list(
                model, venv,
                scenario_files=files,
                repeats=args.repeats,
                deterministic=bool(args.deterministic),
                tag=f"{model_tag}/{bench_tag}"
            )

            # write outputs
            out_sub = os.path.join(args.out_dir, bench_tag)
            ensure_dir(out_sub)

            with open(os.path.join(out_sub, f"{model_tag}_per_scenario.json"), "w", encoding="utf-8") as f:
                json.dump(res["per_scenario"], f, indent=2, ensure_ascii=False)

            with open(os.path.join(out_sub, f"{model_tag}_episodes.json"), "w", encoding="utf-8") as f:
                json.dump(res["episodes"], f, indent=2, ensure_ascii=False)

            summary["results"].setdefault(bench_tag, {})
            summary["results"][bench_tag][model_tag] = res["overall"]

            print(f"[DONE] {model_tag} on {bench_tag}: "
                  f"success={res['overall']['success_rate']:.3f} "
                  f"collision={res['overall']['collision_rate']:.3f} "
                  f"timeout={res['overall']['timeout_rate']:.3f} "
                  f"avg_return={res['overall']['avg_return']:.3f} "
                  f"CI95={res['overall']['return_ci95']} "
                  f"avg_min_lidar={res['overall']['avg_min_lidar']:.3f}")

        # cleanup vec env
        try:
            venv.close()
        except Exception:
            pass

    # Save master summary
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[OK] All evaluations complete.")
    print(f"[OK] Summary saved to: {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()

