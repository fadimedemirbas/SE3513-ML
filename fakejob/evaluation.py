"""Evaluation protocol: repeated grouped CV, nested threshold choice, bootstrap CIs.

All fitted steps live inside the estimator passed in, so every vectoriser,
encoder or resampler is fitted on the training fold only.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

METRICS = ["average_precision", "roc_auc", "f1", "precision", "recall"]


def model_scores(model, X) -> np.ndarray:
    """Continuous fraud scores: higher means more likely fake."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def best_f1_threshold(y_true, scores) -> float:
    """Threshold that maximises fake-class F1 on the given scores (rule: score >= t)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # The last precision/recall pair has no threshold; drop it.
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator),
                   where=denominator > 0)
    return float(thresholds[np.argmax(f1)])


def threshold_metrics(y_true, scores, threshold) -> dict:
    predictions = (scores >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, predictions, zero_division=0),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
    }


def all_metrics(y_true, scores, threshold) -> dict:
    return {
        "average_precision": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        **threshold_metrics(y_true, scores, threshold),
    }


def inner_threshold(estimator, X, y, groups, n_splits, seed) -> float:
    """Pick a threshold from out-of-fold scores inside the outer training fold.

    The outer test fold is never seen here, so the F1 reported on it is not
    tuned on itself.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_scores = np.empty(len(y))
    for train_pos, valid_pos in splitter.split(X, y, groups):
        model = clone(estimator).fit(X.iloc[train_pos], y.iloc[train_pos])
        oof_scores[valid_pos] = model_scores(model, X.iloc[valid_pos])
    return best_f1_threshold(y, oof_scores)


def _run_fold(estimator, X, y, groups, train_pos, test_pos, repeat, fold, inner_splits, seed,
              fixed_threshold):
    X_train, y_train = X.iloc[train_pos], y.iloc[train_pos]
    X_test, y_test = X.iloc[test_pos], y.iloc[test_pos]

    threshold = inner_threshold(
        estimator, X_train, y_train, groups.iloc[train_pos], inner_splits, seed
    )
    model = clone(estimator).fit(X_train, y_train)
    scores = model_scores(model, X_test)

    row = {
        "repeat": repeat,
        "fold": fold,
        "n_test": len(test_pos),
        "n_fake": int(y_test.sum()),
        "threshold": threshold,
        **all_metrics(y_test, scores, threshold),
    }
    if fixed_threshold is not None:
        fixed = threshold_metrics(y_test, scores, fixed_threshold)
        row.update({f"{name}_fixed": value for name, value in fixed.items()})

    oof = pd.DataFrame({"repeat": repeat, "fold": fold, "score": scores}, index=X_test.index)
    return row, oof


@dataclass
class CVResult:
    name: str
    folds: pd.DataFrame          # one row per (repeat, fold)
    oof_scores: pd.DataFrame     # out-of-fold score for every row, per repeat

    def summary(self) -> pd.Series:
        """Mean and standard deviation of each metric across all outer folds."""
        columns = [c for c in self.folds.columns
                   if c in METRICS or c.endswith("_fixed") or c == "threshold"]
        stats = self.folds[columns].agg(["mean", "std"]).T
        return pd.Series(
            {name: f"{row['mean']:.3f} ± {row['std']:.3f}" for name, row in stats.iterrows()},
            name=self.name,
        )


def repeated_group_cv(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    name: str = "model",
    n_splits: int = 5,
    n_repeats: int = 3,
    inner_splits: int = 3,
    seed: int = 0,
    fixed_threshold: float | None = None,
    n_jobs: int = 1,
) -> CVResult:
    """Repeated StratifiedGroupKFold with a nested threshold choice in every fold.

    Each repeat reshuffles the groups with a different seed. Groups never cross
    fold boundaries, so near-duplicate postings stay on one side.
    """
    tasks = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + repeat
        )
        for fold, (train_pos, test_pos) in enumerate(splitter.split(X, y, groups)):
            tasks.append((train_pos, test_pos, repeat, fold))

    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_run_fold)(
            estimator, X, y, groups, train_pos, test_pos, repeat, fold,
            inner_splits, seed + 1000 + repeat * n_splits + fold, fixed_threshold,
        )
        for train_pos, test_pos, repeat, fold in tasks
    )
    rows, oofs = zip(*outputs)
    return CVResult(name=name, folds=pd.DataFrame(rows), oof_scores=pd.concat(oofs))


def compare(baseline: CVResult, candidate: CVResult, metrics=("average_precision", "f1")) -> pd.DataFrame:
    """Paired comparison on identical folds: candidate minus baseline, per metric.

    Both results must come from the same X, groups, n_splits, n_repeats and seed.
    """
    keys = ["repeat", "fold"]
    merged = baseline.folds.merge(candidate.folds, on=keys, suffixes=("_base", "_cand"))
    assert len(merged) == len(baseline.folds), "Results were not produced on the same folds"
    rows = []
    for metric in metrics:
        diff = merged[f"{metric}_cand"] - merged[f"{metric}_base"]
        rows.append({
            "metric": metric,
            "baseline": merged[f"{metric}_base"].mean(),
            "candidate": merged[f"{metric}_cand"].mean(),
            "mean_diff": diff.mean(),
            "std_diff": diff.std(),
            "folds_improved": f"{int((diff > 0).sum())}/{len(diff)}",
        })
    return pd.DataFrame(rows).set_index("metric")


def passes_rule(comparison: pd.DataFrame, metric: str = "average_precision", min_folds: int = 12) -> bool:
    """Decision rule from Step 1: positive mean paired gain and better in >= min_folds folds.

    With 15 folds, 12 wins corresponds to a one-sided sign test p of about 0.018.
    """
    row = comparison.loc[metric]
    wins = int(str(row["folds_improved"]).split("/")[0])
    return bool(row["mean_diff"] > 0 and wins >= min_folds)


def bootstrap_ci(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    groups: pd.Series,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> pd.DataFrame:
    """Group-level bootstrap confidence intervals for a fixed set of test predictions.

    Whole similarity groups are resampled, so near-duplicates are drawn together.
    """
    y_values = np.asarray(y_true)
    scores = np.asarray(scores)
    group_codes, _ = pd.factorize(np.asarray(groups))
    members = pd.Series(np.arange(len(y_values))).groupby(group_codes).apply(np.array).tolist()
    n_groups = len(members)

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        chosen = rng.integers(0, n_groups, n_groups)
        positions = np.concatenate([members[g] for g in chosen])
        y_sample = y_values[positions]
        if y_sample.min() == y_sample.max():
            continue  # a sample with one class has no AUC; skip it
        samples.append(all_metrics(y_sample, scores[positions], threshold))

    samples = pd.DataFrame(samples)
    alpha = (1 - level) / 2
    point = all_metrics(y_values, scores, threshold)
    return pd.DataFrame({
        "estimate": pd.Series(point),
        f"ci_low": samples.quantile(alpha),
        f"ci_high": samples.quantile(1 - alpha),
    }).loc[METRICS]
