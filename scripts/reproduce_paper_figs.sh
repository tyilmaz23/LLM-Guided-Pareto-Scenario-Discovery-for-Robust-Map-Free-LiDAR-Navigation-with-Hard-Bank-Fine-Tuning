#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
mkdir -p outputs/figs

python tools/make_fig_hardness_gap.py   --summary artifacts/eval_summaries/main_results_summary.json   --out outputs/figs/hardness_gap_base100k.pdf   --dpi 1000

python tools/make_fig_robust_gain.py   --summary artifacts/eval_summaries/main_results_summary.json   --out outputs/figs/robust_gain_hardllm.pdf   --dpi 1000

python tools/make_fig_success_vs_budget.py   --out outputs/figs/success_vs_budget.pdf   --dpi 1000

echo "Done. See outputs/figs/"
