# SE3513-ML
SE3513 Machine Learning Project — Fake Job Posting Detection

## Contents

| Path | Description |
|---|---|
| `fake_job_detection.ipynb` | Main notebook: data exploration, grouped split, Experiments 1–6, reference model and limitations |
| `fake_job_structured_model.joblib` | Saved reference model (class-weighted Linear SVM) and its decision threshold |
| `evaluation_framework.ipynb` | Step 1: repeated grouped cross-validation protocol and baseline results |
| `feature_engineering.ipynb` | Step 2: feature engineering ablations and the selected feature configuration |
| `robustness_checks.ipynb` | Checks before Step 3: Step 2 selection on fresh CV folds and label-noise sensitivity analysis |
| `model_extensions.ipynb` | Step 3: calibration, sentence-embedding features and LightGBM stacking against the Step 2 model |
| `text_cleaning.ipynb` | Data-quality check: words joined when the HTML was removed, and whether repairing them improves the selected model |
| `f1_improvements.ipynb` | F1 experiments for postings without a company profile: group thresholds, interaction features and an SVM + logistic regression ensemble |
| `fakejob/` | Shared code: data loading and split (`data.py`), evaluation protocol (`evaluation.py`), model definitions (`models.py`), Step 2 features and configurable pipelines (`features.py`), sentence embeddings (`embeddings.py`) |
| `results/` | Fold-level CV results and test confidence intervals, used for paired comparisons |
| [Experiment progress and next steps](docs/experiment_progress.md) | Working comparison of completed experiments and open verification tasks |
| `docs/pre_development_research_report.md` | Review of public notebooks on this dataset and the planned methodology |
| `docs/pre_development_research_report_tr.md` | Turkish version of the same report |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch, much smaller
pip install -r requirements.txt
```

The dataset (version 1) is downloaded automatically with `kagglehub` on first run. Notebooks must be run from the repository root so that `import fakejob` works. The sentence-embedding model (`all-MiniLM-L6-v2`, about 90 MB) is downloaded from Hugging Face on first use, and the embeddings are cached in `cache/` (not tracked).
