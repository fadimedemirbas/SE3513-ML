"""Feature frame and configurable pipelines for the Step 2 ablations.

`posting_features()` only applies fixed, row-wise transformations (string
concatenation, lengths, counts, splitting `location`). Nothing is learned from
other rows, so it can be computed once on the whole dataset without leakage.
Everything that is fitted (TF-IDF vocabularies, one-hot categories, scaling)
lives inside the pipeline returned by `build_pipeline()` and is therefore
fitted on the training fold only.
"""
from dataclasses import dataclass, field, replace
import html
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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


_GLUED_WORDS = re.compile(r"(?<=[a-z])(?=[A-Z][a-z])")
_GLUED_SENTENCES = re.compile(r"(?<=[a-z][.!?:;,])(?=[A-Za-z])")


def clean_posting_text(text: pd.Series) -> pd.Series:
    """Repair text joined when the HTML of a posting was removed.

    List items and paragraphs were concatenated without a space ("platformDevelop",
    "them.Carefully"), so the word TF-IDF saw invented tokens such as "platformdevelop".
    A space is inserted between a lower-case letter and a following capitalised word,
    and after sentence punctuation directly followed by a letter. HTML entities
    ("&amp;") are decoded and runs of whitespace collapsed. Each posting is changed on
    its own, so nothing is learned across rows.
    """
    text = text.map(html.unescape)
    text = text.str.replace(_GLUED_WORDS, " ", regex=True)
    text = text.str.replace(_GLUED_SENTENCES, " ", regex=True)
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def posting_features(model_df: pd.DataFrame, clean_text: bool = False) -> pd.DataFrame:
    """Raw columns plus row-wise derived features used by all Step 2 pipelines.

    With `clean_text=True` the text fields are repaired with `clean_posting_text()`
    before any feature is derived from them. The default keeps the original text, so
    earlier results are unchanged.
    """
    out = pd.DataFrame(index=model_df.index)

    for col in ALL_TEXT_FIELDS:
        out[col] = model_df[col].fillna("")
        if clean_text:
            out[col] = clean_posting_text(out[col])
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

# Columns of fakejob.embeddings.posting_embeddings() (384-dimensional all-MiniLM-L6-v2 vectors).
EMBEDDING_COLUMNS = [f"emb_{i}" for i in range(384)]


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
    embeddings: bool = False           # append the sentence-embedding columns (Step 3)
    profile_interaction: bool = False  # word TF-IDF repeated for postings with an empty company profile
    classifier: str = "svm"            # "svm" (LinearSVC) or "logreg" (LogisticRegression)
    C: float = 1.0
    max_iter: int = 5000               # converged models stop earlier, so results match the 1000 default

    def but(self, **changes) -> "Config":
        return replace(self, **changes)


# Configuration selected in Step 2 (feature_engineering.ipynb).
STEP2_CONFIG = Config(word_ngrams=(1, 2), sublinear_tf=True, min_df=2, char_ngrams=(3, 5))


class EmptyProfileInteraction(BaseEstimator, TransformerMixin):
    """Word TF-IDF block that is non-zero only for postings with an empty company profile.

    Added next to the ordinary word block, it lets a linear model learn separate
    word weights for postings without a profile, where most errors occur.
    Expects two columns: the text column and the 0/1 `empty_company_profile` flag.
    """

    def __init__(self, ngram_range=(1, 1), sublinear_tf=False, min_df=1):
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df

    def fit(self, X, y=None):
        text, empty = X.iloc[:, 0], X.iloc[:, 1].to_numpy(dtype=bool)
        # Vocabulary from the empty-profile postings only: these are the rows the block describes.
        self.vectorizer_ = TfidfVectorizer(
            ngram_range=self.ngram_range, sublinear_tf=self.sublinear_tf, min_df=self.min_df,
        ).fit(text[empty])
        return self

    def transform(self, X):
        text, empty = X.iloc[:, 0], X.iloc[:, 1].to_numpy(dtype=float)
        return self.vectorizer_.transform(text).multiply(empty[:, None]).tocsr()

    def get_feature_names_out(self, input_features=None):
        return np.array([f"empty_profile__{t}" for t in self.vectorizer_.get_feature_names_out()])


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
    if config.embeddings:
        # Already L2-normalised, like each TF-IDF block, so they are passed through unscaled.
        transformers.append(("embeddings", "passthrough", EMBEDDING_COLUMNS))
    if config.profile_interaction:
        transformers.append((
            "empty_profile_word",
            EmptyProfileInteraction(**word_kwargs),
            [_text_column(config.text_fields), "empty_company_profile"],
        ))
        transformers.append(("empty_profile_flag", "passthrough", ["empty_company_profile"]))

    if config.classifier == "svm":
        classifier = LinearSVC(C=config.C, class_weight="balanced", random_state=42,
                               max_iter=config.max_iter)
    elif config.classifier == "logreg":
        classifier = LogisticRegression(C=config.C, class_weight="balanced", solver="liblinear",
                                        max_iter=config.max_iter, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {config.classifier}")
    return make_pipeline(ColumnTransformer(transformers), classifier)
