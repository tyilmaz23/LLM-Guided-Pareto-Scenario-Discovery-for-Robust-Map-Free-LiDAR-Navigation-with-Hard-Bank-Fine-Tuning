"""
coverage_tracker.py

Deterministic coverage and novelty tracking over discovered scenarios.

Purpose:
- Track explored regions of scenario space
- Identify under-covered areas (coverage gaps)
- Provide explainable signals for LLM-guided proposal (soft guidance)

NO LLM usage.
ESWA-compliant, reproducible, auditable.
"""

import numpy as np
from collections import defaultdict
from failure_analytics import extract_geom_features  # dosyanın üstüne ekle

class CoverageTracker:
    """
    Tracks coverage in a low-dimensional, interpretable scenario-feature space.

    Feature axes (example):
    - geometry difficulty proxy (gap / arm_len)
    - perceptual degradation (lidar_noise, dropout)
    - achieved difficulty score
    """

    def __init__(self):
        self._records = []
        self._bins = {
            "geometry": [],
            "perception": [],
            "difficulty": []
        }

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================


    def _extract_features(self, scenario, agg):
        sc_id = scenario.get("scenario_id", "")

        geom_feats = extract_geom_features(scenario)

        # --- geometry proxy ---
        if sc_id.startswith(("U_", "IID_U_", "C_", "IID_C_")):
            gap = float(geom_feats.get("gap", 0.0))
            geom = 1.0 / max(1e-6, gap)

        else:
            geom = float(geom_feats.get("obstacle_count", geom_feats.get("wall_count", 0))) / 20.0

        # --- perception proxy ---
        sensor = scenario.get("sensor_cfg", scenario.get("sensor", {}))
        perception = float(sensor.get("lidar_noise_sigma", 0.0) + sensor.get("dropout_prob", 0.0))

        difficulty = float(agg.get("difficulty_weighted", 0.0))

        return {"geometry": geom, "perception": perception, "difficulty": difficulty}

    # =====================================================
    # UPDATE
    # =====================================================
    def update(self, scenario, agg):
        """
        Registers a new accepted scenario into coverage statistics.
        """

        feats = self._extract_features(scenario, agg)
        self._records.append(feats)

        for k in self._bins:
            self._bins[k].append(feats[k])

    # =====================================================
    # SNAPSHOT
    # =====================================================
    def snapshot(self):
        """
        Returns a summarized view of explored coverage.
        This output is safe to pass to LLM (read-only).
        """

        if len(self._records) == 0:
            return {
                "counts": 0,
                "coverage": {},
                "gaps": {}
            }

        snap = {
            "counts": len(self._records),
            "coverage": {},
            "gaps": {}
        }

        for k, vals in self._bins.items():
            arr = np.array(vals)

            snap["coverage"][k] = {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr))
            }

            # heuristic gap detection:
            # large std but narrow min-max OR low density
            span = snap["coverage"][k]["max"] - snap["coverage"][k]["min"]
            density = len(arr) / max(span, 1e-6)

            snap["gaps"][k] = {
                "span": float(span),
                "density": float(density)
            }

        return snap

    # =====================================================
    # OPTIONAL: COVERAGE SCORE
    # =====================================================
    def coverage_score(self):
        """
        Single scalar coverage proxy.
        Useful for ablation plots.
        """

        if len(self._records) == 0:
            return 0.0

        score = 0.0
        for k, vals in self._bins.items():
            arr = np.array(vals)
            score += float(np.std(arr))

        return score / len(self._bins)

