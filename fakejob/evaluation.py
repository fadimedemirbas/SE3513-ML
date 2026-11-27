"""Evaluation protocol: repeated grouped CV, nested threshold choice, bootstrap CIs.

All fitted steps live inside the estimator passed in, so every vectoriser,
encoder or resampler is fitted on the training fold only.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
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


def best_fbeta_threshold(y_true, scores, beta: float) -> float:
    """Threshold that maximises the fake-class F-beta score (rule: score >= t).

    beta > 1 weights recall more than precision; beta = 2 counts recall twice as much.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]
    b2 = beta ** 2
    denominator = b2 * precision + recall
    fbeta = np.divide((1 + b2) * precision * recall, denominator, out=np.zeros_like(denominator),
                      where=denominator > 0)
    return float(thresholds[np.argmax(fbeta)])


def fbeta_from_counts(precision: float, recall: float, beta: float) -> float:
    b2 = beta ** 2
    denominator = b2 * precision + recall
    return float((1 + b2) * precision * recall / denominator) if denominator > 0 else 0.0


def _counts_at(y_true, scores, candidates):
    """True and false positives for each candidate threshold (rule: score >= t)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    positives, negatives = np.sort(scores[y_true == 1]), np.sort(scores[y_true == 0])
    tp = len(positives) - np.searchsorted(positives, candidates, side="left")
    fp = len(negatives) - np.searchsorted(negatives, candidates, side="left")
    return tp, fp


def best_f1_group_thresholds(y_true, scores, groups) -> dict:
    """One threshold per group (exactly two groups), chosen jointly to maximise overall F1.

    The best F1 threshold always sits at the score of a positive example, so only
    those scores (plus +inf, "flag nothing in this group") are searched.
    """
    y_true, scores, groups = np.asarray(y_true), np.asarray(scores), np.asarray(groups)
    labels = np.unique(groups)
    assert len(labels) == 2, "group thresholds are implemented for two groups"
    curves = []
    for label in labels:
        member = groups == label
        candidates = np.append(np.unique(scores[member & (y_true == 1)]), np.inf)
        tp, fp = _counts_at(y_true[member], scores[member], candidates)
        curves.append((candidates, tp, fp))
    (c0, tp0, fp0), (c1, tp1, fp1) = curves
    tp = tp0[:, None] + tp1[None, :]
    fp = fp0[:, None] + fp1[None, :]
    f1 = 2 * tp / np.maximum(tp + fp + y_true.sum(), 1)
    i, j = np.unravel_index(np.argmax(f1), f1.shape)
    return {labels[0]: float(c0[i]), labels[1]: float(c1[j])}


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


def inner_oof_scores(estimator, X, y, groups, n_splits, seed) -> np.ndarray:
    """Out-of-fold scores for every row of the outer training fold (inner grouped CV)."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_scores = np.empty(len(y))
    for train_pos, valid_pos in splitter.split(X, y, groups):
        model = clone(estimator).fit(X.iloc[train_pos], y.iloc[train_pos])
        oof_scores[valid_pos] = model_scores(model, X.iloc[valid_pos])
    return oof_scores


def inner_threshold(estimator, X, y, groups, n_splits, seed) -> float:
    """Pick a threshold from out-of-fold scores inside the outer training fold.

    The outer test fold is never seen here, so the F1 reported on it is not
    tuned on itself.
    """
    return best_f1_threshold(y, inner_oof_scores(estimator, X, y, groups, n_splits, seed))


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    """Weighted mean |observed fake rate - mean predicted probability| over equal-width bins."""
    y_true, probabilities = np.asarray(y_true), np.asarray(probabilities)
    bins = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        error += in_bin.mean() * abs(y_true[in_bin].mean() - probabilities[in_bin].mean())
    return float(error)


def fit_calibrators(scores, y) -> dict:
    """Score-to-probability maps fitted on out-of-fold scores: Platt (sigmoid) and isotonic."""
    scores = np.asarray(scores).reshape(-1, 1)
    platt = LogisticRegression(C=1e6).fit(scores, y)
    isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(scores.ravel(), y)
    return {
        "platt": lambda s: platt.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1],
        "isotonic": lambda s: isotonic.predict(np.asarray(s)),
    }


def _run_fold(estimator, X, y, groups, train_pos, test_pos, repeat, fold, inner_splits, seed,
              fixed_threshold, calibrate=False, threshold_group_column=None, fbeta=None):
    X_train, y_train = X.iloc[train_pos], y.iloc[train_pos]
    X_test, y_test = X.iloc[test_pos], y.iloc[test_pos]

    train_oof = inner_oof_scores(
        estimator, X_train, y_train, groups.iloc[train_pos], inner_splits, seed
    )
    threshold = best_f1_threshold(y_train, train_oof)
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

    if fbeta is not None:
        # Second operating point on the same scores: threshold chosen to maximise F-beta
        # on the same inner out-of-fold scores as the F1 threshold.
        name = f"f{fbeta:g}"
        row[f"{name}_at_f1_threshold"] = fbeta_from_counts(row["precision"], row["recall"], fbeta)
        beta_threshold = best_fbeta_threshold(y_train, train_oof, fbeta)
        at_beta = threshold_metrics(y_test, scores, beta_threshold)
        row[f"threshold_{name}"] = beta_threshold
        row[f"precision_{name}"] = at_beta["precision"]
        row[f"recall_{name}"] = at_beta["recall"]
        row[f"f1_{name}"] = at_beta["f1"]
        row[name] = fbeta_from_counts(at_beta["precision"], at_beta["recall"], fbeta)

    if threshold_group_column is not None:
        # Same scores, but a separate threshold per group, chosen on the inner out-of-fold scores.
        group_thresholds = best_f1_group_thresholds(y_train, train_oof, X_train[threshold_group_column])
        per_row = X_test[threshold_group_column].map(group_thresholds).to_numpy(dtype=float)
        grouped = threshold_metrics(y_test, scores, per_row)
        row.update({f"{name}_grouped": value for name, value in grouped.items()})
        row.update({f"threshold_{threshold_group_column}_{g}": t for g, t in group_thresholds.items()})

    if calibrate:
        # Calibrators are fitted on the inner out-of-fold scores of the training fold only.
        row["brier_prior"] = brier_score_loss(y_test, np.full(len(y_test), y_train.mean()))
        for method, to_probability in fit_calibrators(train_oof, y_train).items():
            probabilities = to_probability(scores)
            row[f"brier_{method}"] = brier_score_loss(y_test, probabilities)
            row[f"ece_{method}"] = expected_calibration_error(y_test, probabilities)
            oof[f"prob_{method}"] = probabilities
    return row, oof


@dataclass
class CVResult:
    name: str
    folds: pd.DataFrame          # one row per (repeat, fold)
    oof_scores: pd.DataFrame     # out-of-fold score for every row, per repeat

    def summary(self) -> pd.Series:
        """Mean and standard deviation of each metric across all outer folds."""
        columns = [c for c in self.folds.columns
                   if c in METRICS or c.endswith("_fixed") or c == "threshold"
                   or c.startswith(("brier_", "ece_", "threshold_")) or c.endswith("_grouped")]
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
    calibrate: bool = False,
    threshold_group_column: str | None = None,
    fbeta: float | None = None,
) -> CVResult:
    """Repeated StratifiedGroupKFold with a nested threshold choice in every fold.

    Each repeat reshuffles the groups with a different seed. Groups never cross
    fold boundaries, so near-duplicate postings stay on one side.

    With `calibrate=True`, Platt and isotonic calibrators are fitted on the
    inner out-of-fold scores of each training fold and their Brier score and
    expected calibration error on the test fold are added to the results.

    With `threshold_group_column` (a 0/1 column of X), a second set of threshold
    metrics (`f1_grouped`, ...) uses one threshold per group, chosen jointly on the
    inner out-of-fold scores to maximise overall F1. Scores, and therefore
    PR-AUC, are unchanged.

    With `fbeta` (for example 2), a second threshold that maximises F-beta on the same
    inner out-of-fold scores is added (`threshold_f2`, `precision_f2`, `recall_f2`,
    `f1_f2`, `f2`), together with the F-beta of the F1 threshold (`f2_at_f1_threshold`).
    The model and its scores are unchanged.
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
            inner_splits, seed + 1000 + repeat * n_splits + fold, fixed_threshold, calibrate,
            threshold_group_column, fbeta,
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
