"""Load, clean and split the dataset exactly as in fake_job_detection.ipynb.

The grouping and split logic is copied from the main notebook so that every
experiment uses the same held-out test set (2,597 postings, 131 fake).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

DATASET_HANDLE = "shivamb/real-or-fake-fake-jobposting-prediction/versions/1"
TARGET = "fraudulent"


def load_raw() -> pd.DataFrame:
    """Download (or reuse the cached copy of) version 1 of the dataset."""
    import kagglehub

    dataset_dir = kagglehub.dataset_download(DATASET_HANDLE)
    return pd.read_csv(Path(dataset_dir) / "fake_job_postings.csv")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates (ignoring job_id) and rows without a description."""
    comparison_columns = df.columns.drop("job_id")
    clean_df = df.drop_duplicates(subset=comparison_columns)
    return clean_df.dropna(subset=["description"]).copy()


def description_groups(model_df: pd.DataFrame, radius: float = 0.10) -> pd.Series:
    """Group postings whose descriptions are near-duplicates.

    Two postings are linked when the cosine distance between their TF-IDF
    description vectors is at most `radius`; groups are the connected
    components of that graph. Postings in one group never cross a split.
    """
    normalized = (
        model_df["description"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    matrix = TfidfVectorizer().fit_transform(normalized)
    graph = NearestNeighbors(radius=radius, metric="cosine", algorithm="brute").fit(
        matrix
    ).radius_neighbors_graph(matrix, mode="connectivity")
    _, labels = connected_components(graph, directed=False)
    return pd.Series(labels, index=model_df.index, name="group")


@dataclass
class Split:
    model_df: pd.DataFrame
    groups: pd.Series
    train_index: pd.Index
    val_index: pd.Index
    test_index: pd.Index

    @property
    def dev_index(self) -> pd.Index:
        """Training + validation rows: everything except the held-out test set."""
        return self.train_index.append(self.val_index)

    @property
    def y(self) -> pd.Series:
        return self.model_df[TARGET]


def make_split(model_df: pd.DataFrame, groups: pd.Series) -> Split:
    """Reproduce the notebook's 20-fold grouped assignment (14 train / 3 val / 3 test)."""
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)
    fold_ids = pd.Series(-1, index=model_df.index)
    for fold_number, (_, positions) in enumerate(
        splitter.split(model_df, model_df[TARGET], groups=groups)
    ):
        fold_ids.iloc[positions] = fold_number

    split = Split(
        model_df=model_df,
        groups=groups,
        train_index=fold_ids.index[fold_ids < 14],
        val_index=fold_ids.index[(fold_ids >= 14) & (fold_ids < 17)],
        test_index=fold_ids.index[fold_ids >= 17],
    )
    _check_no_group_overlap(split)
    return split


def _check_no_group_overlap(split: Split) -> None:
    parts = {
        "train": set(split.groups.loc[split.train_index]),
        "val": set(split.groups.loc[split.val_index]),
        "test": set(split.groups.loc[split.test_index]),
    }
    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = parts[first] & parts[second]
        assert not shared, f"{len(shared)} groups shared between {first} and {second}"


def load_split() -> Split:
    """One call that returns the cleaned data, the groups and the fixed split."""
    model_df = clean(load_raw())
    return make_split(model_df, description_groups(model_df))


# --- Feature frame used by the current final model (Experiment 3) ---------------

REFERENCE_TEXT_COLUMNS = ["title", "company_profile", "description"]
REFERENCE_CATEGORICAL_COLUMNS = ["employment_type", "required_experience"]
REFERENCE_BINARY_COLUMNS = ["telecommuting", "has_company_logo"]


def reference_features(model_df: pd.DataFrame) -> pd.DataFrame:
    """Build the input frame expected by the saved Experiment 3 pipeline."""
    features = model_df[REFERENCE_CATEGORICAL_COLUMNS + REFERENCE_BINARY_COLUMNS].copy()
    features["combined_text"] = (
        model_df[REFERENCE_TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)
    )
    features[REFERENCE_CATEGORICAL_COLUMNS] = features[REFERENCE_CATEGORICAL_COLUMNS].fillna(
        "Unknown"
    )
    return features
