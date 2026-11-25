"""Model definitions shared across experiments."""
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
