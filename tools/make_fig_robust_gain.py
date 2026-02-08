#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robust fine-tuning kazanımı figürü (Hard-LLM üzerinde).

Base-100k vs Robust-NoLLM vs Robust-LLM için Success ve Timeout oranlarını yan yana verir.

Örnek:
python tools/make_fig_robust_gain.py --summary artifacts/eval_summaries/main_results_summary.json --out figs/robust_gain_hardllm.pdf --dpi 1000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_summary(path: Path):
    s = json.loads(path.read_text(encoding="utf-8"))
    recs = s.get("records", s)
    if not isinstance(recs, list):
        raise ValueError("summary formatı beklenenden farklı: 'records' listesi bulunamadı.")
    return recs


def get_rates(recs, benchmark: str, model: str):
    for r in recs:
        if r.get("benchmark") == benchmark and r.get("model") == model:
            return float(r["success_rate"]), float(r["timeout_rate"])
    raise KeyError(f"Kayıt bulunamadı: benchmark={benchmark}, model={model}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=1000)
    args = ap.parse_args()

    recs = load_summary(args.summary)
    models = ["Base", "Robust-NoLLM", "Robust-LLM"]
    success, timeout = [], []
    for m in models:
        s, t = get_rates(recs, "Hard-LLM", m)
        success.append(100*s)
        timeout.append(100*t)

    x = np.arange(len(models))
    w = 0.35

    plt.figure(figsize=(6.0, 3.2))
    plt.bar(x - w/2, success, width=w, label="Success (%)")
    plt.bar(x + w/2, timeout, width=w, label="Timeout (%)")
    plt.xticks(x, models, rotation=0)
    plt.ylim(0, max(max(success), max(timeout)) * 1.15)
    plt.ylabel("Rate (%)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
