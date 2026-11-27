"""
Evaluate the locked Step 2 model once on the project's fixed held-out test split.

Run from the SE3513-ML project root:
    ./.venv/bin/python evaluate_final_model.py

This script does not fit or modify the model. It recreates the original fixed
grouped split, loads models/final_step2_model.joblib, evaluates the reserved
test partition once, and writes new files under results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fakejob.data import load_split
from fakejob.evaluation import expected_calibration_error, model_scores
from fakejob.features import posting_features


EXPECTED_SKLEARN = "1.6.1"
ARTIFACT_PATH = Path("models/final_step2_model.joblib")
OUTPUT_DIR = Path("results")
PREDICTIONS_PATH = OUTPUT_DIR / "final_test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "final_test_metrics.json"
CI_PATH = OUTPUT_DIR / "final_test_group_bootstrap_ci.csv"
CONFUSION_PATH = OUTPUT_DIR / "final_test_confusion_matrix.csv"

N_BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 0
CONFIDENCE_LEVEL = 0.95


def group_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    groups: np.ndarray,
    n_boot: int = N_BOOTSTRAPS,
    seed: int = BOOTSTRAP_SEED,
    level: float = CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Percentile CIs from resampling whole similarity groups."""
    group_codes, _ = pd.factorize(groups)
    members = [
        np.flatnonzero(group_codes == code)
        for code in np.unique(group_codes)
    ]
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_boot):
        selected_groups = rng.integers(0, len(members), len(members))
        positions = np.concatenate([members[i] for i in selected_groups])
        y_sample = y_true[positions]
        if np.unique(y_sample).size < 2:
            continue

        score_sample = scores[positions]
        predicted = (score_sample >= threshold).astype(int)
        samples.append(
            {
                "average_precision": average_precision_score(y_sample, score_sample),
                "roc_auc": roc_auc_score(y_sample, score_sample),
                "f1": f1_score(y_sample, predicted, zero_division=0),
                "precision": precision_score(y_sample, predicted, zero_division=0),
                "recall": recall_score(y_sample, predicted, zero_division=0),
            }
        )

    if not samples:
        raise RuntimeError("No valid bootstrap samples contained both classes.")

    sample_frame = pd.DataFrame(samples)
    alpha = (1.0 - level) / 2.0
    point_predictions = (scores >= threshold).astype(int)
    point = {
        "average_precision": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "f1": f1_score(y_true, point_predictions, zero_division=0),
        "precision": precision_score(y_true, point_predictions, zero_division=0),
        "recall": recall_score(y_true, point_predictions, zero_division=0),
    }
    return pd.DataFrame(
        {
            "estimate": pd.Series(point),
            "ci_low": sample_frame.quantile(alpha),
            "ci_high": sample_frame.quantile(1.0 - alpha),
        }
    )


def main() -> None:
    if sklearn.__version__ != EXPECTED_SKLEARN:
        raise RuntimeError(
            f"Expected scikit-learn {EXPECTED_SKLEARN}, found {sklearn.__version__}. "
            "Activate the project's .venv before running this script."
        )

    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Locked model not found: {ARTIFACT_PATH}. "
            "Run finalize_model.py successfully first."
        )

    print("=" * 72)
    print("FINAL TEST EVALUATION — LOCKED STEP 2 SVM")
    print("=" * 72)
    print(f"Python: {sys.version.split()[0]}")
    print(f"scikit-learn: {sklearn.__version__}")

    # Recreate the exact original split. This may take time because the
    # unchanged data.py recomputes description-similarity groups.
    print("\nLoading the original dataset and recreating the fixed grouped split...")
    split = load_split()

    test_index = split.test_index
    X_all = posting_features(split.model_df)
    X_test = X_all.loc[test_index].copy()
    y_test = split.y.loc[test_index].astype(int).to_numpy()
    groups_test = split.groups.loc[test_index].to_numpy()

    artifact = joblib.load(ARTIFACT_PATH)
    model = artifact["model"]
    calibrator = artifact["platt_calibrator"]
    raw_threshold = float(artifact["raw_threshold"])
    calibrated_threshold = float(artifact["calibrated_probability_threshold"])

    print(f"\nTest rows: {len(X_test):,}")
    print(f"Test fake postings: {int(y_test.sum()):,}")
    print(f"Test real postings: {int((y_test == 0).sum()):,}")
    print(f"Fixed raw SVM threshold: {raw_threshold:.8f}")
    print(f"Fixed calibrated probability threshold: {calibrated_threshold:.8f}")

    if len(X_test) != 2597 or int(y_test.sum()) != 131:
        raise RuntimeError(
            "The recreated test split does not match the project's documented "
            "2,597 rows / 131 fake postings. Stop rather than evaluating a different split."
        )

    # No fitting, threshold selection or model changes occur below.
    print("\nScoring the held-out test set with the locked model...")
    scores = np.asarray(model_scores(model, X_test), dtype=float)
    probabilities = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    predictions = (scores >= raw_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    point_metrics = {
        "average_precision": float(average_precision_score(y_test, scores)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "fake_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "fake_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "fake_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "accuracy": float((predictions == y_test).mean()),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "expected_calibration_error": float(
            expected_calibration_error(y_test, probabilities)
        ),
    }

    print("\nConfusion matrix (rows=true, columns=predicted; 0=real, 1=fake):")
    print(f"                 Predicted real   Predicted fake")
    print(f"Actual real       {tn:>8}         {fp:>8}")
    print(f"Actual fake       {fn:>8}         {tp:>8}")

    print("\nHeld-out test metrics:")
    for name, value in point_metrics.items():
        print(f"  {name}: {value:.6f}")

    print(
        f"\nRunning {N_BOOTSTRAPS:,} group-bootstrap resamples "
        f"for 95% confidence intervals..."
    )
    ci = group_bootstrap_ci(y_test, scores, raw_threshold, groups_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "original_index": test_index.to_numpy(),
            "y_true": y_test,
            "svm_decision_score": scores,
            "fake_probability_platt": probabilities,
            "y_pred": predictions,
            "correct": (predictions == y_test),
            "description_group": groups_test,
        }
    )
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)

    pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["actual_real", "actual_fake"],
        columns=["predicted_real", "predicted_fake"],
    ).to_csv(CONFUSION_PATH)

    ci.to_csv(CI_PATH, index_label="metric")

    report = {
        "model": "Selected Step 2 word + character TF-IDF LinearSVC",
        "artifact_path": str(ARTIFACT_PATH),
        "test_partition": "fixed grouped held-out test split",
        "test_rows": int(len(X_test)),
        "test_fake_rows": int(y_test.sum()),
        "test_real_rows": int((y_test == 0).sum()),
        "raw_svm_threshold": raw_threshold,
        "calibrated_probability_threshold": calibrated_threshold,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "metrics": point_metrics,
        "group_bootstrap": {
            "resamples_requested": N_BOOTSTRAPS,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "confidence_intervals_csv": str(CI_PATH),
        },
        "outputs": {
            "predictions_csv": str(PREDICTIONS_PATH),
            "confusion_matrix_csv": str(CONFUSION_PATH),
            "metrics_json": str(METRICS_PATH),
        },
        "model_refitted_on_test": False,
        "threshold_changed": False,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("FINAL TEST EVALUATION COMPLETED")
    print("=" * 72)
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Confusion matrix: {CONFUSION_PATH}")
    print(f"Group-bootstrap CIs: {CI_PATH}")
    print("Model refitted on test: NO")
    print("Threshold changed: NO")


if __name__ == "__main__":
    main()
