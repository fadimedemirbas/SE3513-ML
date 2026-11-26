"""Sentence embeddings of the postings (Step 3).

The embedding model is pre-trained and used without fine-tuning, so each
posting's vector depends only on its own text and never on labels or other
rows. Computing the embeddings once for the whole dataset therefore does not
leak information; any model fitted on top of them is still fitted inside the
training folds.
"""
from pathlib import Path

import numpy as np
import pandas as pd

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256          # longest input this model was trained with (word pieces)
CACHE_DIR = Path("cache")


def embedding_text(model_df: pd.DataFrame) -> pd.Series:
    """Title and description: what the job is about. Company information is covered by other features."""
    title = model_df["title"].fillna("").str.strip()
    description = model_df["description"].fillna("").str.strip()
    return title + ". " + description


def posting_embeddings(model_df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    """One L2-normalised 384-dimensional vector per posting, cached on disk by row index."""
    cache_file = CACHE_DIR / f"embeddings_{EMBEDDING_MODEL.split('/')[-1]}_{MAX_SEQ_LENGTH}.npz"
    if cache_file.exists():
        cached = np.load(cache_file)
        if np.array_equal(cached["index"], model_df.index.to_numpy()):
            vectors = cached["vectors"]
            columns = [f"emb_{i}" for i in range(vectors.shape[1])]
            return pd.DataFrame(vectors, index=model_df.index, columns=columns)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    model.max_seq_length = MAX_SEQ_LENGTH
    vectors = model.encode(
        embedding_text(model_df).tolist(), batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True,
    ).astype(np.float32)

    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(cache_file, index=model_df.index.to_numpy(), vectors=vectors)
    columns = [f"emb_{i}" for i in range(vectors.shape[1])]
    return pd.DataFrame(vectors, index=model_df.index, columns=columns)
