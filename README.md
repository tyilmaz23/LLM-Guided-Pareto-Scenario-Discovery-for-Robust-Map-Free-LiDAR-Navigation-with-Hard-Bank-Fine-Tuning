LLM-Guided Pareto Scenario Discovery for Robust Map-Free LiDAR Navigation (Reproducibility Repo)

This repository is a scaffold prepared to package the paper’s Hard-NoLLM / Hard-LLM benchmark generation, hard-bank assembly, robust fine-tuning, and figure/table generation steps in a reproducible manner.

Note: The LLM component is used only during discovery. Even without an LLM API key,
you can reproduce the main tables/figures using the precomputed artifacts
(audit logs / manifests / summary results) included in the repo.

1) Quick start (figures/tables only)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


Reproduce the paper figures:

python tools/make_figs_robust_results.py --help


Example (1000 dpi):

python tools/make_figs_robust_results.py   --eval_dir artifacts/eval_summaries   --out_dir outputs/figs   --dpi 1000


The folder artifacts/eval_summaries/ contains the example/output JSON files that feed the paper’s reporting pipeline.

2) Reproducibility levels

Level A — “Paper artifact replay” (recommended):

Generate tables and figures from JSON/JSONL summaries under artifacts/.

No LLM key or long training runs required.

Level B — “Evaluation replay”:

Re-run the benchmarks using the shared scenario packs + (shared) models.

Runs on CPU; takes longer.

Level C — “Full pipeline”:

Base PPO training + discovery (NoLLM/LLM) + hard-bank + robust fine-tuning + evaluation.

An API key is required to run the LLM mode.

3) Repository structure (recommended)

src/eswa_nav/ : environment / helper modules (as a Python package)

tools/ : runnable scripts (evaluation, fine-tuning, figure generation)

artifacts/ : small/medium artifacts for reproducibility

supplementary/: real-world overlays and additional visuals

docs/ : schemas and working notes

paper/ : (optional) LaTeX sources

4) Minimal files to include on GitHub (checklist)
A) Code (required)

Scenario DSL + generation/validation: scenario_dsl.py, scenario JSON schemas

Simulation environment: ilkkisim.py (CustomEnv) and dependent helpers

Discovery loop (NoLLM + LLM): discovery scripts, coverage tracker, Pareto/elite selection

Hard-bank assembly + split: make_hardbank_manifest.py, generation of robust_manifest.json

Robust fine-tuning: run_robust_finetuning.py

Benchmark evaluation: evaluate_* scripts, run_eval_manifest.py

Analysis + figure generation: make_figs_*.py, table generation and CI computation

B) Configuration (required)

requirements.txt / environment.yml

run_config.json (training/evaluation parameters)

split_salt and seed list (42–51)

CLI commands (scripts/*.sh) or a Makefile

C) Artifacts (at least for “paper replay”)

robust_manifest.json (file lists + splits)

LLM audit traces: llm_audit_log.jsonl (if too large: sample + checksum)

Benchmark/eval summaries: files like *_ep3000.json

(optional) Pareto archives, coverage snapshots

D) Models

At minimum Base-100k: ppo_static_shaping_model.zip, ppo_static_shaping_vecnorm.pkl

If robust models are too large, host them via GitHub Releases / Zenodo.

5) License & Citation

LICENSE for code

CITATION.cff for citation instructions

Contact

If you encounter issues or missing components, please open an “Issue”.
