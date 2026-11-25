"""Feature frame and configurable pipelines for the Step 2 ablations.

`posting_features()` only applies fixed, row-wise transformations (string
concatenation, lengths, counts, splitting `location`). Nothing is learned from
other rows, so it can be computed once on the whole dataset without leakage.
Everything that is fitted (TF-IDF vocabularies, one-hot categories, scaling)
lives inside the pipeline returned by `build_pipeline()` and is therefore
fitted on the training fold only.
"""
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import LinearSVC

CORE_TEXT_FIELDS = ["title", "company_profile", "description"]
EXTRA_TEXT_FIELDS = ["requirements", "benefits"]
ALL_TEXT_FIELDS = CORE_TEXT_FIELDS + EXTRA_TEXT_FIELDS

BASE_CATEGORICAL = ["employment_type", "required_experience"]
EXTRA_CATEGORICAL = ["required_education", "function", "industry"]
BASE_BINARY = ["telecommuting", "has_company_logo"]
EXTRA_BINARY = ["has_questions", "has_salary_range"]

MASKED_TOKENS = {"url": "#URL_", "email": "#EMAIL_", "phone": "#PHONE_"}


def posting_features(model_df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns plus row-wise derived features used by all Step 2 pipelines."""
    out = pd.DataFrame(index=model_df.index)

    for col in ALL_TEXT_FIELDS:
        out[col] = model_df[col].fillna("")
    out["combined_text"] = out[CORE_TEXT_FIELDS].agg(" ".join, axis=1)
    out["all_text"] = out[ALL_TEXT_FIELDS].agg(" ".join, axis=1)

    for col in BASE_CATEGORICAL + EXTRA_CATEGORICAL:
        out[col] = model_df[col].fillna("Unknown")
    for col in ["telecommuting", "has_company_logo", "has_questions"]:
        out[col] = model_df[col]
    out["has_salary_range"] = model_df["salary_range"].notna().astype(int)

    # location looks like "US, NY, New York"; parts can be empty.
    parts = model_df["location"].fillna("").str.split(",", expand=True).reindex(columns=[0, 1])
    country = parts[0].fillna("").str.strip()
    state = parts[1].fillna("").str.strip()
    out["country"] = country.replace("", "Unknown")
    out["country_state"] = (country + "-" + state).replace("-", "Unknown")

    # Numeric "meta" features describing length, completeness and contact details.
    for col in ALL_TEXT_FIELDS:
        stripped = out[col].str.strip()
        out[f"len_{col}"] = np.log1p(stripped.str.len())
        out[f"empty_{col}"] = (stripped == "").astype(int)
    for name, token in MASKED_TOKENS.items():
        out[f"n_{name}"] = np.log1p(out["all_text"].str.count(token))
    letters = out["all_text"].str.count(r"[A-Za-z]")
    out["upper_ratio"] = out["all_text"].str.count(r"[A-Z]") / letters.clip(lower=1)
    out["exclaim_per_1k"] = out["all_text"].str.count("!") / out["all_text"].str.len().clip(lower=1) * 1000
    return out


META_COLUMNS = (
    [f"len_{c}" for c in ALL_TEXT_FIELDS]
    + [f"empty_{c}" for c in ALL_TEXT_FIELDS]
    + [f"n_{name}" for name in MASKED_TOKENS]
    + ["upper_ratio", "exclaim_per_1k"]
)


@dataclass(frozen=True)
class Config:
    """One pipeline variant. The defaults reproduce the Step 1 reference SVM."""
    text_fields: tuple = tuple(CORE_TEXT_FIELDS)
    per_field: bool = False            # one TF-IDF per text field instead of one on the joined text
    word_ngrams: tuple = (1, 1)
    sublinear_tf: bool = False
    min_df: int = 1
    char_ngrams: tuple | None = None   # e.g. (3, 5) adds a char_wb TF-IDF on the joined text
    categorical: tuple = tuple(BASE_CATEGORICAL)
    binary: tuple = tuple(BASE_BINARY)
    meta: bool = False
    location: str | None = None        # None, "country" or "country_state"
    C: float = 1.0
    max_iter: int = 5000               # converged models stop earlier, so results match the 1000 default

    def but(self, **changes) -> "Config":
        return replace(self, **changes)


def _text_column(fields: tuple) -> str:
    if tuple(fields) == tuple(CORE_TEXT_FIELDS):
        return "combined_text"
    if tuple(fields) == tuple(ALL_TEXT_FIELDS):
        return "all_text"
    raise ValueError(f"Unsupported text field set: {fields}")


def build_pipeline(config: Config = Config()):
    word_kwargs = dict(
        ngram_range=config.word_ngrams, sublinear_tf=config.sublinear_tf, min_df=config.min_df,
    )
    transformers = []
    if config.per_field:
        for col in config.text_fields:
            transformers.append((f"word_{col}", TfidfVectorizer(**word_kwargs), col))
    else:
        transformers.append(("word", TfidfVectorizer(**word_kwargs), _text_column(config.text_fields)))

    if config.char_ngrams:
        transformers.append((
            "char",
            TfidfVectorizer(analyzer="char_wb", ngram_range=config.char_ngrams, sublinear_tf=True,
                            min_df=3),
            _text_column(config.text_fields),
        ))

    categorical = list(config.categorical)
    if config.location:
        categorical.append(config.location)
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))
    if config.binary:
        transformers.append(("binary", SimpleImputer(strategy="most_frequent"), list(config.binary)))
    if config.meta:
        # Scaled to [0, 1] like the TF-IDF values; StandardScaler made liblinear fail to converge.
        transformers.append(("meta", MinMaxScaler(clip=True), META_COLUMNS))

    return make_pipeline(
        ColumnTransformer(transformers),
        LinearSVC(C=config.C, class_weight="balanced", random_state=42, max_iter=config.max_iter),
    )
