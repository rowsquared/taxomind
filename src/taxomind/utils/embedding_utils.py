"""Embedding utilities that keep multilingual parity across components."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import numpy as np
from txtai.embeddings import Embeddings


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> Embeddings:
    """Load and cache multilingual embedding models per name."""

    return Embeddings({"path": model_name})


def get_embedding_model(model_name: str) -> Embeddings:
    """Return (and cache) the requested multilingual embedding model."""

    return _load_model(model_name)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(texts: Sequence[str], model_name: str) -> List[List[float]]:
    """Encode mixed-language text chunks using multilingual models."""

    model = get_embedding_model(model_name)
    if hasattr(model, "embed"):
        vectors = model.embed(list(texts))
    else:
        vectors = model.transform(list(texts))
    vectors = _normalize(np.asarray(vectors, dtype=np.float32))
    return vectors.tolist()


def embed_text(text: str, model_name: str) -> List[float]:
    """Encode a single multilingual text sample."""

    return embed_texts([text], model_name)[0]
