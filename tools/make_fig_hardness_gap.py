#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hardness-gap figürü üretir (Base-100k: Hard-NoLLM vs Hard-LLM).

Çıktı:
- PDF (ve opsiyonel PNG) 1000 dpi uyumlu.

Örnek:
python tools/make_fig_hardness_gap.py --summary artifacts/eval_summaries/main_results_summary.json --out figs/hardness_gap_base100k.pdf --dpi 1000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_summary(path: Path):
    s = json.loads(path.read_text(encoding="utf-8"))
    recs = s.get("records", s)  # esnek: {records:[...]} veya direkt liste
    if not isinstance(recs, list):
        raise ValueError("summary formatı beklenenden farklı: 'records' listesi bulunamadı.")
    return recs


def pick_rate(recs, benchmark: str, model: str):
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
    s1, t1 = pick_rate(recs, "Hard-NoLLM", "Base")
    s2, t2 = pick_rate(recs, "Hard-LLM", "Base")

    labels = ["Hard-NoLLM", "Hard-LLM"]
    success = [100*s1, 100*s2]
    timeout = [100*t1, 100*t2]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(5.2, 3.2))
    plt.bar(x - w/2, success, width=w, label="Success (%)")
    plt.bar(x + w/2, timeout, width=w, label="Timeout (%)")
    plt.xticks(x, labels)
    plt.ylim(0, max(max(success), max(timeout)) * 1.15)
    plt.ylabel("Rate (%)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
