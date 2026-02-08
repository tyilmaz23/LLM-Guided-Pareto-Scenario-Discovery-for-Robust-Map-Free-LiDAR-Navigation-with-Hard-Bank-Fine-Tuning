# run_robust_finetuning.py
import os
import json
import glob
import hashlib
import argparse
from typing import List, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from ilkkisim import CustomEnv  # <-- senin env dosyan

# -------------------------
# utils
# -------------------------
def _stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

def list_files_with_pattern(dirs: List[str], pattern: str) -> List[str]:
    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, pattern))))
    files = [f for f in files if os.path.isfile(f)]
    return files


def split_train_test(files: List[str], test_ratio: float, salt: str) -> Tuple[List[str], List[str]]:
    train, test = [], []
    for f in files:
        # basename yerine path kullan (seed klasörleri ayrışsın)
        key = f"{salt}:{os.path.normpath(f)}"
        r = (_stable_hash(key) % 10_000) / 10_000.0
        (test if r < test_ratio else train).append(f)
    return train, test


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------
# Robust mixed env
# -------------------------
class RobustMixedEnv(CustomEnv):
    """
    reset() sırasında karışık dağılım:
      - p = mix_ratio -> hard_train senaryolarından birini yükle
      - aksi -> iid_train senaryolarından birini yükle
    """
    def __init__(self, hard_train_files: List[str], iid_train_files: List[str], mix_ratio: float, seed: int):
        super().__init__()
        self.hard_train_files = hard_train_files
        self.iid_train_files = iid_train_files
        self.mix_ratio = float(mix_ratio)
        self.local_rng = np.random.default_rng(int(seed))

        if len(self.iid_train_files) == 0:
            raise ValueError("iid_train_files boş. iid_dir yanlış olabilir.")
        if len(self.hard_train_files) == 0:
            raise ValueError("hard_train_files boş. hard_dirs yanlış olabilir.")

    def reset(self, seed=None, options=None):
        # hangi mod?
        use_hard = (self.local_rng.random() < self.mix_ratio)

        if use_hard:
            fpath = self.local_rng.choice(self.hard_train_files)
            sc = load_json(str(fpath))
            self.current_mode = "hard"
        else:
            fpath = self.local_rng.choice(self.iid_train_files)
            sc = load_json(str(fpath))
            self.current_mode = "iid"

        # env'e inject et
        self.set_scenario(sc)

        # normal reset
        obs, info = super().reset(seed=seed, options=options)

        # infoya mode ekleyelim (opsiyonel)
        info = dict(info)
        info["reset_mode"] = self.current_mode
        return obs, info


def make_env_fn(hard_train_files, iid_train_files, mix_ratio, seed_offset):
    def _thunk():
        env = RobustMixedEnv(hard_train_files, iid_train_files, mix_ratio=mix_ratio, seed=seed_offset)
        return Monitor(env)
    return _thunk


# -------------------------
# Stopper: episode sayısıyla durdur
# -------------------------
class StopOnTotalEpisodes(BaseCallback):
    def __init__(self, max_episodes: int, verbose=1):
        super().__init__(verbose)
        self.max_episodes = int(max_episodes)
        self.total_episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        if dones is not None:
            self.total_episodes += int(np.sum(dones))
        if self.total_episodes >= self.max_episodes:
            if self.verbose:
                print(f"\n[STOP] Reached {self.total_episodes}/{self.max_episodes} episodes.")
            return False
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--base_vecnorm", required=True)

    ap.add_argument("--iid_dir", required=True, help="IID json klasörü (iid_*.json)")
    ap.add_argument("--hard_dirs", nargs="+", required=True, help="Hard bank klasörleri (seed klasörleri)")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--mix_ratio", type=float, default=0.5, help="Hard seçilme olasılığı")
    ap.add_argument("--episodes", type=int, default=15000)
    ap.add_argument("--n_envs", type=int, default=16)

    ap.add_argument("--new_lr", type=float, default=5e-5)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--split_salt", type=str, default="v1")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) dosyaları topla
    iid_files_all = list_files_with_pattern([args.iid_dir], "iid_*.json")
    hard_files_all = list_files_with_pattern(args.hard_dirs, "top_*.json")


    print(f"[*] IID files : {len(iid_files_all)} from {args.iid_dir}")
    print(f"[*] HARD files: {len(hard_files_all)} from {len(args.hard_dirs)} dirs")

    # 2) train/test split (deterministic)
    iid_train, iid_test = split_train_test(iid_files_all, args.test_ratio, salt=f"iid:{args.split_salt}")
    hard_train, hard_test = split_train_test(hard_files_all, args.test_ratio, salt=f"hard:{args.split_salt}")

    print(f"[*] IID  train/test: {len(iid_train)} / {len(iid_test)}")
    print(f"[*] HARD train/test: {len(hard_train)} / {len(hard_test)}")

    manifest = {
        "iid_dir": args.iid_dir,
        "hard_dirs": args.hard_dirs,
        "mix_ratio": args.mix_ratio,
        "episodes": args.episodes,
        "n_envs": args.n_envs,
        "new_lr": args.new_lr,
        "test_ratio": args.test_ratio,
        "split_salt": args.split_salt,
        "seed": args.seed,
        "iid_train_files": iid_train,
        "iid_test_files": iid_test,
        "hard_train_files": hard_train,
        "hard_test_files": hard_test,
    }
    with open(os.path.join(args.out_dir, "robust_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 3) vec env kur
    env_fns = []
    for i in range(args.n_envs):
        env_fns.append(make_env_fn(hard_train, iid_train, args.mix_ratio, seed_offset=args.seed + 1000 + i))
    venv = SubprocVecEnv(env_fns)

    # 4) vecnorm yükle (istatistikleri koru)
    venv = VecNormalize.load(args.base_vecnorm, venv)
    venv.training = True
    venv.norm_reward = True
    venv.norm_obs = False  # senin ayarın

    # 5) model yükle (LR override güvenli yöntem)
    lr_fn = (lambda _: float(args.new_lr))
    model = PPO.load(
        args.base_model,
        env=venv,
        custom_objects={"learning_rate": lr_fn},
        device="auto",
        print_system_info=True
    )

    # 6) callbacks
    stop_cb = StopOnTotalEpisodes(max_episodes=args.episodes, verbose=1)
    ckpt_cb = CheckpointCallback(
        save_freq=max(1000 // args.n_envs, 1),
        save_path=args.out_dir,
        name_prefix="robust_ppo"
    )

    print(f"[*] Robust fine-tune starting | episodes={args.episodes} | mix_ratio={args.mix_ratio} | lr={args.new_lr}")
    model.learn(
        total_timesteps=10_000_000,
        callback=CallbackList([stop_cb, ckpt_cb]),
        progress_bar=True
    )

    # 7) kaydet
    model_path = os.path.join(args.out_dir, "ppo_robust_final_model.zip")
    vec_path = os.path.join(args.out_dir, "ppo_robust_final_vecnorm.pkl")
    model.save(model_path)
    venv.save(vec_path)
    print("[OK] Saved:")
    print("  ", model_path)
    print("  ", vec_path)


if __name__ == "__main__":
    main()

