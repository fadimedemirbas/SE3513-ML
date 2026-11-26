"""Model definitions shared across experiments."""
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
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
