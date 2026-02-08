#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eğitim bütçesine bağlı başarı eğrisi figürü (Hard-LLM üzerinde).

Varsayılan değerler, makaledeki ablation tablosundan alınmıştır.
İsterseniz JSON ile override edebilirsiniz.

Örnek:
python tools/make_fig_success_vs_budget.py --out figs/success_vs_budget.pdf --dpi 1000
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT = {
  "budgets": [1000, 3000, 15000],
  "Robust-NoLLM": [97.87, 98.60, 98.80],
  "Robust-LLM":   [98.23, 98.47, 98.83]
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=1000)
    ap.add_argument("--json", type=Path, default=None, help="Opsiyonel: {budgets:[...], Robust-NoLLM:[...], Robust-LLM:[...]} formatında override.")
    args = ap.parse_args()

    cfg = DEFAULT
    if args.json is not None:
        cfg = json.loads(args.json.read_text(encoding="utf-8"))

    budgets = cfg["budgets"]
    y1 = cfg["Robust-NoLLM"]
    y2 = cfg["Robust-LLM"]

    plt.figure(figsize=(5.2, 3.2))
    plt.plot(budgets, y1, marker="o", label="Robust-NoLLM")
    plt.plot(budgets, y2, marker="o", label="Robust-LLM")
    plt.xscale("log")
    plt.xticks(budgets, [f"{b//1000}k" if b>=1000 else str(b) for b in budgets])
    plt.ylim(min(min(y1), min(y2)) - 0.6, max(max(y1), max(y2)) + 0.6)
    plt.ylabel("Success (%)")
    plt.xlabel("Training budget (episodes, log scale)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
