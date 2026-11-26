"""Model definitions shared across experiments."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import LinearSVC

from .data import REFERENCE_BINARY_COLUMNS, REFERENCE_CATEGORICAL_COLUMNS

# Threshold stored with fake_job_structured_model.joblib (chosen on the old validation split).
REFERENCE_THRESHOLD = -0.05


def reference_pipeline():
    """The current final model (Experiment 3): TF-IDF + structured features + weighted Linear SVM.

    Identical to the pipeline trained in fake_job_detection.ipynb.
    """
    features = ColumnTransformer([
        ("text", TfidfVectorizer(), "combined_text"),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), REFERENCE_CATEGORICAL_COLUMNS),
        ("binary", SimpleImputer(strategy="most_frequent"), REFERENCE_BINARY_COLUMNS),
    ])
    return make_pipeline(features, LinearSVC(class_weight="balanced", random_state=42))


class TrainingRowFilter(BaseEstimator, ClassifierMixin):
    """Fit the wrapped estimator without the rows flagged in `exclude_column`.

    Used for sensitivity analyses: flagged rows are dropped from every training
    fold (including the inner threshold folds) but are still scored and
    evaluated when they fall in a test fold.
    """

    def __init__(self, estimator, exclude_column: str):
        self.estimator = estimator
        self.exclude_column = exclude_column

    def fit(self, X, y):
        keep = ~X[self.exclude_column].astype(bool).to_numpy()
        self.estimator_ = clone(self.estimator).fit(X[keep], y[keep])
        self.classes_ = self.estimator_.classes_
        return self

    def decision_function(self, X):
        return self.estimator_.decision_function(X)


class GroupedStacking(BaseEstimator, ClassifierMixin):
    """Gradient boosting on an out-of-fold text score plus structured features.

    `fit` first produces a score for every training row from a text model that
    did not see that row (grouped inner CV, so near-duplicates stay together),
    then fits the booster on [text score, structured features]. At prediction
    time the text model refitted on all training rows supplies the score.
    The group labels are read from `group_column`, which is not used as a feature.
    """

    def __init__(self, text_model, booster, numeric_columns, categorical_columns,
                 group_column="group", n_splits=3, random_state=0):
        self.text_model = text_model
        self.booster = booster
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.group_column = group_column
        self.n_splits = n_splits
        self.random_state = random_state

    def _stack(self, X, text_score):
        codes = self.encoder_.transform(X[list(self.categorical_columns)]).astype(int)
        Z = pd.DataFrame(codes, columns=list(self.categorical_columns), index=X.index)
        for col in self.numeric_columns:
            Z[col] = X[col].to_numpy(dtype=float)
        Z["text_score"] = text_score
        return Z

    def fit(self, X, y):
        splitter = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True,
                                        random_state=self.random_state)
        oof = np.empty(len(y))
        for train_pos, valid_pos in splitter.split(X, y, X[self.group_column]):
            model = clone(self.text_model).fit(X.iloc[train_pos], y.iloc[train_pos])
            oof[valid_pos] = model.decision_function(X.iloc[valid_pos])
        self.text_model_ = clone(self.text_model).fit(X, y)

        # Unknown or missing categories become -1, which LightGBM treats as missing.
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                       encoded_missing_value=-1)
        self.encoder_.fit(X[list(self.categorical_columns)])
        self.booster_ = clone(self.booster).fit(
            self._stack(X, oof), y, categorical_feature=list(self.categorical_columns),
        )
        self.classes_ = self.booster_.classes_
        return self

    def predict_proba(self, X):
        return self.booster_.predict_proba(self._stack(X, self.text_model_.decision_function(X)))

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


class ScoreAverage(BaseEstimator, ClassifierMixin):
    """Average of standardised decision scores from several models (an unweighted ensemble).

    Each model's scores are put on a common scale with the mean and standard
    deviation of its scores on the training rows, so no model dominates because
    of its score range.
    """

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(m).fit(X, y) for m in self.models]
        self.scales_ = []
        for model in self.models_:
            scores = model.decision_function(X)
            self.scales_.append((scores.mean(), scores.std() or 1.0))
        self.classes_ = self.models_[0].classes_
        return self

    def decision_function(self, X):
        return np.mean([(m.decision_function(X) - mean) / std
                        for m, (mean, std) in zip(self.models_, self.scales_)], axis=0)
