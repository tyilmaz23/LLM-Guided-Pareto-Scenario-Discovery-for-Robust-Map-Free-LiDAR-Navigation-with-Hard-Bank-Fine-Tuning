# LLM-Guided Pareto Scenario Discovery for Robust Map-Free LiDAR Navigation (Reproducibility Repository)

This repository is a lightweight scaffold prepared to reproduce the paper’s key steps:
**Hard-NoLLM / Hard-LLM benchmark** construction, **hard-bank** assembly,
**robust fine-tuning**, and **figure/table generation** in a reproducible manner.

> Note: The LLM component is used **only during the discovery stage**. Even without an LLM/API key,
> you can still reproduce the main tables/figures using the **precomputed artefacts**
> (audit logs / manifests / evaluation summaries) provided in this repo.

---

## 1) Quick start (figures/tables only)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


> Note: The complete repository (full pipeline code and supplementary artefacts) will be released publicly upon acceptance/publication of the paper.
