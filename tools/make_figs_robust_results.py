#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# IO + parsing
# -------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm_model(m: str) -> str:
    m = str(m).strip().lower()
    if m in ["base", "ppo_base", "ppo_static", "ppo_static_shaping"]:
        return "Base"
    if m in ["robust_nollm", "robust-nollm", "nollm", "robust_nollm_v1"]:
        return "Robust-NoLLM"
    if m in ["robust_llm", "robust-llm", "llm", "robust_llm_v1"]:
        return "Robust-LLM"
    # fallback (capitalize)
    return m.replace("_", "-").title()


def norm_bench(b: str) -> str:
    b = str(b).strip().lower()
    if b in ["iid", "iid_test", "iid-test"]:
        return "IID"
    if b in ["hard_nollm", "hard_nollm_test", "hard-nollm", "hard-nollm-test"]:
        return "Hard-NoLLM"
    if b in ["hard_llm", "hard_llm_test", "hard-llm", "hard-llm-test"]:
        return "Hard-LLM"
    return b.replace("_", "-").title()


def parse_summary(summary_path: str) -> List[Dict[str, Any]]:
    """
    Accepts:
      - Your nested format:
        { "results": { bench: { model: metrics_dict } } }
      - Or list-of-records format (older)
    Returns flat records with:
      record["benchmark"], record["model"], plus metrics fields.
    """
    data = load_json(summary_path)

    # nested "results" format
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], dict):
        out = []
        for bench, models in data["results"].items():
            for model, metrics in models.items():
                rec = {"benchmark": norm_bench(bench), "model": norm_model(model)}
                if isinstance(metrics, dict):
                    rec.update(metrics)
                out.append(rec)
        return out

    # list-of-records format
    if isinstance(data, list):
        out = []
        for r in data:
            if not isinstance(r, dict):
                continue
            bench = r.get("benchmark", r.get("bench", ""))
            model = r.get("model", "")
            rec = {"benchmark": norm_bench(bench), "model": norm_model(model)}
            rec.update(r)
            out.append(rec)
        return out

    raise ValueError("Unsupported summary.json format. Expected dict with 'results' or list of records.")


# -------------------------
# statistics
# -------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    k = int(k)
    n = int(n)
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) / n) + ((z ** 2) / (4 * n ** 2)))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def rate_ci_from_failure_modes(rec: Dict[str, Any], key: str) -> Tuple[float, float]:
    """
    key: "success_rate" / "collision_rate" / "timeout_rate"
    Uses rec["failure_modes"] and rec["n_episodes"] if available.
    """
    fm = rec.get("failure_modes", {}) or {}
    n = int(rec.get("n_episodes", 0) or 0)

    # counts
    succ = int(fm.get("target_reached", 0) or 0)
    col  = int(fm.get("collision", 0) or 0)
    tout = int(fm.get("timeout", 0) or 0)

    if n <= 0:
        # fallback: no CI
        v = float(rec.get(key, 0.0) or 0.0)
        return v, v

    if key == "success_rate":
        return wilson_ci(succ, n)
    if key == "collision_rate":
        return wilson_ci(col, n)
    if key == "timeout_rate":
        return wilson_ci(tout, n)

    v = float(rec.get(key, 0.0) or 0.0)
    return v, v


def yerr_from_ci(val: float, ci: Tuple[float, float]) -> Tuple[float, float]:
    lo, hi = ci
    return (max(0.0, val - lo), max(0.0, hi - val))


# -------------------------
# plotting helpers
# -------------------------
def ensure_out(d: str):
    os.makedirs(d, exist_ok=True)


def save_fig(fig, out_dir: str, name: str, dpi: int):
    png = os.path.join(out_dir, f"{name}.png")
    pdf = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def set_rcparams():
    # clean "paper-ish" defaults
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.linewidth": 0.8,
    })


def build_table(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    table[benchmark][model] -> record
    """
    table: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in records:
        b = norm_bench(r.get("benchmark", ""))
        m = norm_model(r.get("model", ""))
        table.setdefault(b, {})[m] = r
    return table


# -------------------------
# FIG-R1: 2x2 metrics panels
# -------------------------
def fig_metrics_panels(table, out_dir: str, dpi: int):
    benches = ["IID", "Hard-NoLLM", "Hard-LLM"]
    models  = ["Base", "Robust-NoLLM", "Robust-LLM"]

    x = np.arange(len(benches))
    w = 0.22

    def bars(ax, metric: str, title: str, ylim: Tuple[float, float] = None, use_ci: bool = True):
        for i, m in enumerate(models):
            vals, ylo, yhi = [], [], []
            for b in benches:
                rec = table[b][m]
                v = float(rec.get(metric, 0.0) or 0.0)
                vals.append(v)

                if metric in ["success_rate", "collision_rate", "timeout_rate"]:
                    ci = rate_ci_from_failure_modes(rec, metric)
                elif metric == "avg_return":
                    ci95 = rec.get("return_ci95", None)
                    if isinstance(ci95, (list, tuple)) and len(ci95) == 2:
                        ci = (float(ci95[0]), float(ci95[1]))
                    else:
                        ci = (v, v)
                else:
                    ci = (v, v)

                e = yerr_from_ci(v, ci)
                ylo.append(e[0])
                yhi.append(e[1])

            xpos = x + (i - 1) * w
            ax.bar(
                xpos, vals, width=w,
                yerr=np.vstack([ylo, yhi]) if use_ci else None,
                capsize=3, linewidth=0.8, edgecolor="black",
                label=m
            )

        ax.set_xticks(x)
        ax.set_xticklabels(benches)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig, axs = plt.subplots(2, 2, figsize=(12, 7.2))
    bars(axs[0, 0], "success_rate",   "Success rate (95% CI)",   ylim=(0.85, 1.01))
    bars(axs[0, 1], "collision_rate", "Collision rate (95% CI)", ylim=(0.0, 0.03))
    bars(axs[1, 0], "timeout_rate",   "Timeout rate (95% CI)",   ylim=(0.0, 0.12))
    bars(axs[1, 1], "avg_return",     "Average return (95% CI)", ylim=(0.85, 1.07))

    for ax in axs.flatten():
        ax.grid(True, axis="y", alpha=0.25)

    # single legend on top
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Robust evaluation summary", y=1.06, fontsize=13)
    fig.tight_layout()
    save_fig(fig, out_dir, "FIG-R1_metrics_panels", dpi)


# -------------------------
# FIG-R2: failure modes (stacked)
# -------------------------
def fig_failure_modes(table, out_dir: str, dpi: int):
    benches = ["IID", "Hard-NoLLM", "Hard-LLM"]
    models  = ["Base", "Robust-NoLLM", "Robust-LLM"]

    # order in stack
    modes = [
        ("target_reached", "Success"),
        ("timeout", "Timeout"),
        ("collision", "Collision"),
        ("other", "Other"),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    for j, b in enumerate(benches):
        ax = axs[j]
        bottoms = np.zeros(len(models), dtype=float)

        for key, label in modes:
            vals = []
            for m in models:
                rec = table[b][m]
                fm = rec.get("failure_modes", {}) or {}
                n = int(rec.get("n_episodes", 0) or 0)
                cnt = int(fm.get(key, 0) or 0)
                vals.append(cnt / n if n > 0 else 0.0)

            ax.bar(np.arange(len(models)), vals, bottom=bottoms, edgecolor="black", linewidth=0.6, label=label)
            bottoms += np.array(vals)

        ax.set_title(b)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=10)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.25)

    axs[0].set_ylabel("Proportion of episodes")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Failure mode breakdown (normalized)", y=1.08, fontsize=13)

    fig.tight_layout()
    save_fig(fig, out_dir, "FIG-R2_failure_modes_stacked", dpi)


# -------------------------
# FIG-R3: safety/efficiency tradeoffs
# -------------------------
def fig_tradeoffs(table, out_dir: str, dpi: int):
    benches = ["IID", "Hard-NoLLM", "Hard-LLM"]
    models  = ["Base", "Robust-NoLLM", "Robust-LLM"]

    # markers per benchmark (shape), color per model (matplotlib default cycle)
    bench_marker = {"IID": "o", "Hard-NoLLM": "s", "Hard-LLM": "^"}

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))

    # (a) Success vs avg_min_lidar
    ax = axs[0]
    for b in benches:
        for m in models:
            rec = table[b][m]
            x = float(rec.get("avg_min_lidar", np.nan))
            y = float(rec.get("success_rate", 0.0))
            ax.scatter(x, y, s=90, marker=bench_marker[b], label=f"{m} | {b}")
            ax.annotate(f"{m}\n{b}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_title("Safety–performance trade-off")
    ax.set_xlabel("Average minimum LiDAR distance (higher = safer)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.85, 1.01)
    ax.grid(True, alpha=0.25)

    # (b) Success vs avg_steps
    ax = axs[1]
    for b in benches:
        for m in models:
            rec = table[b][m]
            x = float(rec.get("avg_steps", np.nan))
            y = float(rec.get("success_rate", 0.0))
            ax.scatter(x, y, s=90, marker=bench_marker[b])
            ax.annotate(f"{m}\n{b}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_title("Efficiency–performance trade-off")
    ax.set_xlabel("Average steps (lower = more efficient)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.85, 1.01)
    ax.grid(True, alpha=0.25)

    fig.suptitle("Robust evaluation trade-offs", y=1.05, fontsize=13)
    fig.tight_layout()
    save_fig(fig, out_dir, "FIG-R3_tradeoffs", dpi)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="eval_results.../summary.json")
    ap.add_argument("--out_dir", default="paper_figs_robust")
    ap.add_argument("--dpi", type=int, default=1000)

    # (optional) keep compatibility with your CLI
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--iid_n", type=int, default=None)
    ap.add_argument("--hard_nollm_n", type=int, default=None)
    ap.add_argument("--hard_llm_n", type=int, default=None)

    args = ap.parse_args()

    set_rcparams()
    ensure_out(args.out_dir)

    records = parse_summary(args.summary)
    table = build_table(records)

    # sanity check
    for b in ["IID", "Hard-NoLLM", "Hard-LLM"]:
        if b not in table:
            raise KeyError(f"Missing benchmark in summary: {b}")
        for m in ["Base", "Robust-NoLLM", "Robust-LLM"]:
            if m not in table[b]:
                raise KeyError(f"Missing model '{m}' under benchmark '{b}'")

    fig_metrics_panels(table, args.out_dir, args.dpi)
    fig_failure_modes(table, args.out_dir, args.dpi)
    fig_tradeoffs(table, args.out_dir, args.dpi)

    print("[OK] Robust figures saved under:", args.out_dir)


if __name__ == "__main__":
    main()

