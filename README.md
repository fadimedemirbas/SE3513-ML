# SE3513-ML
SE3513 Machine Learning Project — Fake Job Posting Detection

## Contents

| Path | Description |
|---|---|
| `fake_job_detection.ipynb` | Main notebook: data exploration, grouped split, Experiments 1–6, final model and limitations |
| `fake_job_structured_model.joblib` | Saved final model (class-weighted Linear SVM) and its decision threshold |
| `evaluation_framework.ipynb` | Step 1: repeated grouped cross-validation protocol and baseline results |
| `fakejob/` | Shared code: data loading and split (`data.py`), evaluation protocol (`evaluation.py`), model definitions (`models.py`) |
| `results/` | Fold-level CV results and test confidence intervals, used for paired comparisons |
| `docs/pre_development_research_report.md` | Review of public notebooks on this dataset and the planned methodology |
| `docs/pre_development_research_report_tr.md` | Turkish version of the same report |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

The dataset (version 1) is downloaded automatically with `kagglehub` on first run. Notebooks must be run from the repository root so that `import fakejob` works.
