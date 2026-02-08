# make_figs_scenario_discovery.py
import os
import json
import glob
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Helpers
# =========================================================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_np(x):
    """Always give matplotlib a numpy array (fix pandas multidim indexing issue)."""
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


def ensure_out(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)


def save_fig(fig, out_dir: str, name: str, dpi: int):
    png = os.path.join(out_dir, f"{name}.png")
    pdf = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def find_seed_dirs(base: str, prefix: str) -> List[str]:
    patt = os.path.join(base, f"{prefix}*")
    dirs = [d for d in sorted(glob.glob(patt)) if os.path.isdir(d)]
    return dirs


def load_summaries(seed_dirs: List[str]) -> pd.DataFrame:
    rows = []
    for d in seed_dirs:
        sp = os.path.join(d, "summary.json")
        if os.path.isfile(sp):
            data = load_json(sp)
            if isinstance(data, list):
                rows.extend(data)
        else:
            print(f"[WARN] summary.json not found in {d}, skipping.")
    if len(rows) == 0:
        raise FileNotFoundError("No summary.json found in given seed dirs.")
    return pd.DataFrame(rows)


def load_curriculum(seed_dirs: List[str]) -> Optional[pd.DataFrame]:
    logs = []
    for d in seed_dirs:
        cp = os.path.join(d, "curriculum_log.json")
        if os.path.isfile(cp):
            data = load_json(cp)
            if isinstance(data, list) and len(data) > 0:
                tmp = pd.DataFrame(data)
                tmp["seed_dir"] = os.path.basename(d)
                logs.append(tmp)

    if len(logs) == 0:
        return None

    df = pd.concat(logs, ignore_index=True)

    for c in ["iter", "temperature", "difficulty"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby("iter", as_index=False)[["temperature", "difficulty"]].mean()
    return g.sort_values("iter")


def scenario_type_from_id(sid: str) -> str:
    """
    IMPORTANT:
    - Use startswith (not 'in') to avoid accidental matches.
    - Check CL_ BEFORE C_ (otherwise CL_ can be mis-read as C_).
    """
    s = (sid or "").strip()
    if s.startswith(("IID_U_", "U_")):
        return "U-trap"
    if s.startswith(("IID_CL_", "CL_")):
        return "Clutter"
    if s.startswith(("IID_C_", "C_")):
        return "Corridor"
    return "Other"


def _annotate_bar_counts(ax, rects, fontsize=10):
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] else 1.0
    for r in rects:
        h = r.get_height()
        ax.text(
            r.get_x() + r.get_width() / 2,
            h + max(1.0, 0.015 * ymax),
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


# =========================================================
# Panel functions
# =========================================================
def panel_pareto(ax, df_r: pd.DataFrame, df_l: pd.DataFrame,
                 ycol: str, ylabel: str, title: str,
                 y_scale: str = "log10", eps: float = 1e-6):
    x_r = to_np(df_r["difficulty_weighted"])
    x_l = to_np(df_l["difficulty_weighted"])
    y_r = to_np(df_r[ycol])
    y_l = to_np(df_l[ycol])

    if y_scale == "log10":
        y_r_plot = np.log10(y_r + eps)
        y_l_plot = np.log10(y_l + eps)
        ylabel_plot = f"log10({ylabel} + {eps:g})"
    elif y_scale == "symlog":
        y_r_plot = y_r
        y_l_plot = y_l
        ylabel_plot = ylabel
        ax.set_yscale("symlog", linthresh=1e-4)
    else:
        y_r_plot = y_r
        y_l_plot = y_l
        ylabel_plot = ylabel

    ax.scatter(x_r, y_r_plot, s=14, alpha=0.75, label="Random / NoLLM", rasterized=True)
    ax.scatter(x_l, y_l_plot, s=18, alpha=0.85, marker="^", label="LLM-guided", rasterized=True)

    ax.set_xlabel("Difficulty (weighted)")
    ax.set_ylabel(ylabel_plot)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def panel_outcomes(ax, df_r: pd.DataFrame, df_l: pd.DataFrame):
    metrics = ["success_rate", "collision_rate", "timeout_rate"]
    labels = ["Success", "Collision", "Timeout"]

    r_mean = [float(df_r[m].mean()) for m in metrics]
    r_std  = [float(df_r[m].std(ddof=1)) for m in metrics]
    l_mean = [float(df_l[m].mean()) for m in metrics]
    l_std  = [float(df_l[m].std(ddof=1)) for m in metrics]

    x = np.arange(len(metrics))
    w = 0.38

    ax.bar(x - w/2, r_mean, width=w, yerr=r_std, capsize=3, label="Random / NoLLM")
    ax.bar(x + w/2, l_mean, width=w, yerr=l_std, capsize=3, label="LLM-guided")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate (mean ± std)")
    ax.grid(True, axis="y", alpha=0.25)


def panel_difficulty_hist(ax, df_r: pd.DataFrame, df_l: pd.DataFrame):
    r = to_np(df_r["difficulty_weighted"])
    l = to_np(df_l["difficulty_weighted"])
    bins = 30
    ax.hist(r, bins=bins, density=True, alpha=0.6, label="Random / NoLLM")
    ax.hist(l, bins=bins, density=True, alpha=0.6, label="LLM-guided")
    ax.set_xlabel("Difficulty (weighted)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)


def panel_curriculum(ax, curr_r: Optional[pd.DataFrame], curr_l: Optional[pd.DataFrame],
                     smooth_window: int = 7,
                     twin_ylabel_inside: bool = True):
    ax.set_title("Curriculum dynamics (mean over seeds)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Curriculum temperature")

    if curr_r is None or curr_l is None:
        ax.text(0.5, 0.5, "curriculum_log.json not found\n(skipped)", ha="center", va="center")
        ax.grid(True, alpha=0.25)
        return

    r = curr_r.copy()
    l = curr_l.copy()

    r["temp_s"] = r["temperature"].rolling(smooth_window, min_periods=1).mean()
    l["temp_s"] = l["temperature"].rolling(smooth_window, min_periods=1).mean()
    r["diff_s"] = r["difficulty"].rolling(smooth_window, min_periods=1).mean()
    l["diff_s"] = l["difficulty"].rolling(smooth_window, min_periods=1).mean()

    x_r = to_np(r["iter"])
    x_l = to_np(l["iter"])

    ax.plot(x_r, to_np(r["temp_s"]), label="Temp (Random/NoLLM)")
    ax.plot(x_l, to_np(l["temp_s"]), linestyle="--", label="Temp (LLM)")

    ax2 = ax.twinx()
    ax2.set_ylabel("Scenario difficulty", labelpad=8)

    # KEY FIX: move the twin-y label slightly INSIDE the left subplot
    # so it won't collide with the right subplot's y-label.
    if twin_ylabel_inside:
        ax2.yaxis.set_label_coords(0.97, 0.5)  # < 1.0 => inside
        ax2.tick_params(axis="y", pad=2)

    ax2.plot(x_r, to_np(r["diff_s"]), linestyle=":", label="Diff (Random/NoLLM)")
    ax2.plot(x_l, to_np(l["diff_s"]), linestyle="-.", label="Diff (LLM)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, fontsize=9)

    ax.grid(True, alpha=0.25)


def panel_difficulty_components(ax, df_r: pd.DataFrame, df_l: pd.DataFrame):
    def comp(df):
        return {
            "Failure (1-SR)" : float(2.0 * (1 - df["success_rate"].mean())),
            "Collision"      : float(1.5 * df["collision_rate"].mean()),
            "Timeout"        : float(0.8 * df["timeout_rate"].mean()),
            "Unfinished"     : float(df.get("unfinished_ratio", pd.Series([0.0])).mean()),
            "Near-wall"      : float(max(0.0, 0.35 - df.get("avg_min_lidar", pd.Series([0.35])).mean())),
        }

    cr = comp(df_r)
    cl = comp(df_l)
    keys = list(cr.keys())

    x = np.arange(len(keys))
    w = 0.38

    ax.bar(x - w/2, [cr[k] for k in keys], width=w, label="Random / NoLLM")
    ax.bar(x + w/2, [cl[k] for k in keys], width=w, label="LLM-guided")

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=0)
    ax.set_ylabel("Avg contribution", labelpad=10)
    ax.set_title("Difficulty composition")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)


# =========================================================
# Main figures
# =========================================================
def make_fig1(df_r, df_l, out_dir, dpi, y_scale="log10", eps=1e-6):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))

    panel_pareto(
        axs[0, 0], df_r, df_l,
        ycol="early_crash_rate",
        ylabel="Early crash rate",
        title="(a) Pareto projection: difficulty vs early-crash",
        y_scale=y_scale, eps=eps
    )
    panel_pareto(
        axs[0, 1], df_r, df_l,
        ycol="collapse_score",
        ylabel="Collapse score",
        title="(b) Pareto projection: difficulty vs collapse",
        y_scale=y_scale, eps=eps
    )

    panel_outcomes(axs[1, 0], df_r, df_l)
    axs[1, 0].set_title("(c) Outcome summary")

    panel_difficulty_hist(axs[1, 1], df_r, df_l)
    axs[1, 1].set_title("(d) Difficulty distribution")

    handles, labels = [], []
    for ax in axs.flatten():
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    uniq = {}
    for h, l in zip(handles, labels):
        uniq[l] = h
    if len(uniq) > 0:
        fig.legend(
            list(uniq.values()),
            list(uniq.keys()),
            loc="upper center",
            ncol=2,
            frameon=True,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.subplots_adjust(top=0.88, wspace=0.25, hspace=0.28)
    save_fig(fig, out_dir, "FIG1_discovery_overview_panels", dpi)


def make_fig2(df_r, df_l, curr_r, curr_l, out_dir, dpi, smooth_window=7):
    # KEY FIX: build with GridSpec to control spacing and prevent ylabel collisions
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.38)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    panel_curriculum(ax0, curr_r, curr_l, smooth_window=smooth_window, twin_ylabel_inside=True)
    panel_difficulty_components(ax1, df_r, df_l)

    save_fig(fig, out_dir, "FIG2_curriculum_and_components", dpi)


def make_type_fig(df_r: pd.DataFrame, df_l: pd.DataFrame, out_dir: str, dpi: int):
    types = ["U-trap", "Corridor", "Clutter", "Other"]

    dr = df_r.copy()
    dl = df_l.copy()
    dr["type"] = dr["scenario_id"].apply(scenario_type_from_id)
    dl["type"] = dl["scenario_id"].apply(scenario_type_from_id)

    cr = dr["type"].value_counts().reindex(types).fillna(0).astype(int)
    cl = dl["type"].value_counts().reindex(types).fillna(0).astype(int)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    rects0 = axs[0].bar(cr.index, cr.values)
    axs[0].set_title("Scenario types (Random / NoLLM)")
    axs[0].set_ylabel("Count")
    axs[0].grid(True, axis="y", alpha=0.25)
    _annotate_bar_counts(axs[0], rects0)

    rects1 = axs[1].bar(cl.index, cl.values)
    axs[1].set_title("Scenario types (LLM-guided)")
    axs[1].set_ylabel("Count")
    axs[1].grid(True, axis="y", alpha=0.25)
    _annotate_bar_counts(axs[1], rects1)

    fig.subplots_adjust(wspace=0.25)
    save_fig(fig, out_dir, "FIG3_scenario_type_distributions", dpi)


def make_example_render(example_json: str, out_dir: str, dpi: int):
    sc = load_json(example_json)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_title(f"Example mined scenario: {sc.get('scenario_id','')}")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    for w in sc.get("dynamic_walls", []):
        x1, y1, x2, y2 = w
        ax.fill(
            [x1, x2, x2, x1], [y1, y1, y2, y2],
            alpha=0.55,
            edgecolor="k",
            linewidth=0.4
        )

    a = sc["agent"]
    t = sc["target"]

    ax.scatter([a["x"]], [a["y"]], s=80, marker="o", label="Agent")
    ax.scatter([t["x"]], [t["y"]], s=110, marker="^", label="Target")

    th = a.get("theta", None)
    if th is not None:
        ax.arrow(
            a["x"], a["y"],
            0.35 * np.cos(th), 0.35 * np.sin(th),
            head_width=0.12,
            head_length=0.12,
            length_includes_head=True
        )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    save_fig(fig, out_dir, "FIG4_example_scenario_render", dpi)


# =========================================================
# CLI
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios_base", type=str, default="scenarios")
    ap.add_argument("--nollm_prefix", type=str, default="random_top_nollm_v2_seed")
    ap.add_argument("--llm_prefix", type=str, default="random_top_llm_v2_seed")
    ap.add_argument("--out_dir", type=str, default="paper_figs_scenario_discovery")
    ap.add_argument("--dpi", type=int, default=1000)
    ap.add_argument("--example_json", type=str, default=None, help="Example top_*.json path")
    ap.add_argument("--y_scale", type=str, default="log10", choices=["log10", "symlog", "linear"])
    ap.add_argument("--eps", type=float, default=1e-6, help="epsilon for log10(y+eps)")
    ap.add_argument("--smooth_window", type=int, default=7, help="rolling mean window for curriculum curves")
    args = ap.parse_args()

    ensure_out(args.out_dir)

    nollm_dirs = find_seed_dirs(args.scenarios_base, args.nollm_prefix)
    llm_dirs   = find_seed_dirs(args.scenarios_base, args.llm_prefix)

    print("[*] NoLLM dirs:", len(nollm_dirs))
    print("[*] LLM dirs  :", len(llm_dirs))

    df_r = load_summaries(nollm_dirs)
    df_l = load_summaries(llm_dirs)

    need_cols = [
        "scenario_id", "difficulty_weighted",
        "early_crash_rate", "collapse_score",
        "success_rate", "collision_rate", "timeout_rate"
    ]
    for c in need_cols:
        if c not in df_r.columns or c not in df_l.columns:
            raise KeyError(f"Missing required column '{c}' in summaries.")

    curr_r = load_curriculum(nollm_dirs)
    curr_l = load_curriculum(llm_dirs)

    make_fig1(df_r, df_l, args.out_dir, args.dpi, y_scale=args.y_scale, eps=args.eps)
    make_fig2(df_r, df_l, curr_r, curr_l, args.out_dir, args.dpi, smooth_window=args.smooth_window)
    make_type_fig(df_r, df_l, args.out_dir, args.dpi)

    if args.example_json is not None and os.path.isfile(args.example_json):
        make_example_render(args.example_json, args.out_dir, args.dpi)
    else:
        print("[WARN] example_json not provided or not found; FIG4 skipped.")

    df_r["type"] = df_r["scenario_id"].apply(scenario_type_from_id)
    df_l["type"] = df_l["scenario_id"].apply(scenario_type_from_id)
    print("[INFO] Type counts (NoLLM):\n", df_r["type"].value_counts())
    print("[INFO] Type counts (LLM):\n", df_l["type"].value_counts())

    print("[OK] Figures saved under:", args.out_dir)


if __name__ == "__main__":
    main()

