"""
failure_analytics.py

Deterministic failure & difficulty analysis module.
NO LLM usage.

Purpose:
- Extract structured failure signals from Pareto archive
- Provide statistically grounded summaries for LLM-guided adversarial search
- ESWA-compliant: fully explainable, reproducible, auditable
"""

import numpy as np
from collections import defaultdict


# =========================================================
# BASIC HELPERS
# =========================================================

def _safe_mean(arr):
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


def _safe_std(arr):
    return float(np.std(arr)) if len(arr) > 0 else 0.0

def _norm_rect(w):
    x1, y1, x2, y2 = map(float, w)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

def _rect_w_h(r):
    return (r[2] - r[0]), (r[3] - r[1])

def _point_aabb_dist(px, py, r):
    # point to axis-aligned rect distance
    x1, y1, x2, y2 = r
    dx = 0.0
    if px < x1: dx = x1 - px
    elif px > x2: dx = px - x2
    dy = 0.0
    if py < y1: dy = y1 - py
    elif py > y2: dy = py - y2
    return float(np.hypot(dx, dy))

def extract_geom_features(sc):
    """
    Returns small, LLM-friendly geometry features derived from walls:
    - u_trap: gap, arm_len, wall_thickness
    - corridor: gap, corridor_len, wall_thickness
    - clutter/other: obstacle_count, total_area, mean_area, mean_thickness
    """
    walls = sc.get("dynamic_walls", []) or []
    rects = [_norm_rect(w) for w in walls]

    feats = {}

    # generic
    feats["wall_count"] = len(rects)
    if len(rects) == 0:
        return feats

    areas = []
    thicknesses = []
    for r in rects:
        w, h = _rect_w_h(r)
        areas.append(w * h)
        thicknesses.append(min(w, h))

    feats["total_obstacle_area"] = float(np.sum(areas))
    feats["mean_obstacle_area"] = float(np.mean(areas))
    feats["mean_wall_thickness"] = float(np.mean(thicknesses))

    sc_id = sc.get("scenario_id", "")

    # -------- corridor: 2 horizontal walls --------
    if (sc_id.startswith("C_") or sc_id.startswith("IID_C_")) and len(rects) == 2:
        r1, r2 = rects
        # lower/upper by y
        lower, upper = (r1, r2) if r1[1] <= r2[1] else (r2, r1)
        gap = upper[1] - lower[3]              # upper bottom - lower top
        length = 0.5 * ((lower[2] - lower[0]) + (upper[2] - upper[0]))
        feats["gap"] = float(gap)
        feats["corridor_len"] = float(length)
        return feats

    # -------- u_trap: 3 walls; pick 2 vertical arms --------
    if (sc_id.startswith("U_") or sc_id.startswith("IID_U_")) and len(rects) == 3:
        verticals = []
        for r in rects:
            w, h = _rect_w_h(r)
            if h > w:  # vertical-ish
                verticals.append(r)
        if len(verticals) >= 2:
            # pick two tallest verticals as arms
            verticals = sorted(verticals, key=lambda r: _rect_w_h(r)[1], reverse=True)[:2]
            left, right = sorted(verticals, key=lambda r: r[0])
            gap = right[0] - left[2]  # right inner face - left inner face
            arm_len = min(left[3], right[3]) - max(left[1], right[1])
            feats["gap"] = float(gap)
            feats["arm_len"] = float(arm_len)
        return feats

    # -------- clutter / other --------
    feats["obstacle_count"] = len(rects)
    return feats

# =========================================================
# FAILURE MODE EXTRACTION
# =========================================================

def summarize_failures(pareto_archive):
    """
    Input:
        pareto_archive = [
            {
                "scenario": {...},
                "agg": {
                    "difficulty_weighted": float,
                    "early_crash_rate": float,
                    "collapse_score": float,
                    ...
                }
            },
            ...
        ]

    Output:
        summary_stats (dict)  → passed to LLM (read-only)
    """

    if len(pareto_archive) == 0:
        return {
            "success_rate": 1.0,
            "early_crash_rate": 0.0,
            "collapse_score": 0.0,
            "difficulty": {
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0
            },
            "top_failures": [],
            "scenario_type_distribution": {}
        }

    difficulties = []
    early_crashes = []
    collapses = []
    success_rates = []  

    # scenario-type buckets
    by_type = defaultdict(list)

    # top hard scenarios
    hard_cases = []

    for entry in pareto_archive:
        sc = entry["scenario"]
        agg = entry["agg"]

        diff = agg.get("difficulty_weighted", 0.0)
        early = agg.get("early_crash_rate", 0.0)
        collapse = agg.get("collapse_score", 0.0)
        success_rates.append(float(agg.get("success_rate", 0.0))) 

        difficulties.append(diff)
        early_crashes.append(early)
        collapses.append(collapse)

        sc_id = sc.get("scenario_id", "UNKNOWN")

        if sc_id.startswith(("U_", "IID_U_")):
            sc_type = "u_trap"
        elif sc_id.startswith(("C_", "IID_C_")):
            sc_type = "corridor"
        elif sc_id.startswith(("CL_", "IID_CL_")):
            sc_type = "clutter"
        else:
            sc_type = "other"


        by_type[sc_type].append(diff)
        sensor = sc.get("sensor_cfg", sc.get("sensor", {}))
        geom = extract_geom_features(sc)


        hard_cases.append({
            "scenario": sc_id,
            "difficulty": diff,
            "early_crash_rate": early,
            "collapse_score": collapse,
            "sensor_dropout": float(sensor.get("dropout_prob", 0.0)),
            "lidar_noise": float(sensor.get("lidar_noise_sigma", 0.0)),
            **geom,
        })

    # -----------------------------------------------------
    # Aggregate statistics
    # -----------------------------------------------------



    summary = {
        "success_rate": round(_safe_mean(success_rates), 3),
        "early_crash_rate": round(_safe_mean(early_crashes), 3),
        "collapse_score": round(_safe_mean(collapses), 3),
        "difficulty": {
            "mean": round(_safe_mean(difficulties), 3),
            "std": round(_safe_std(difficulties), 3),
            "max": round(max(difficulties), 3)
        },
        "scenario_type_distribution": {
            k: round(_safe_mean(v), 3) for k, v in by_type.items()
        }
    }

    # -----------------------------------------------------
    # Top failure cases (LLM-readable but grounded)
    # -----------------------------------------------------

    hard_cases = sorted(
        hard_cases,
        key=lambda x: x["difficulty"],
        reverse=True
    )

    summary["top_failures"] = hard_cases[:5]

    return summary


# =========================================================
# OPTIONAL: DIFFICULTY BUCKETING (for analysis / appendix)
# =========================================================

def bucket_by_difficulty(pareto_archive, bins=(0.3, 0.6)):
    """
    Groups scenarios into easy / medium / hard buckets.
    Useful for ablation plots and appendix tables.
    """

    buckets = {
        "easy": [],
        "medium": [],
        "hard": []
    }

    for entry in pareto_archive:
        diff = entry["agg"].get("difficulty_weighted", 0.0)

        if diff < bins[0]:
            buckets["easy"].append(entry)
        elif diff < bins[1]:
            buckets["medium"].append(entry)
        else:
            buckets["hard"].append(entry)

    return {k: len(v) for k, v in buckets.items()}

