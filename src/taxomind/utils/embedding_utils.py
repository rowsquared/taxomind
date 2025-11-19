"""Embedding utilities that keep multilingual parity across components."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load and cache multilingual embedding models per name."""

    return SentenceTransformer(model_name)


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Return (and cache) the requested multilingual embedding model."""

    return _load_model(model_name)


def embed_texts(texts: Sequence[str], model_name: str) -> List[List[float]]:
    """Encode mixed-language text chunks using multilingual models."""

    embeddings = get_embedding_model(model_name).encode(
        list(texts), normalize_embeddings=True
    )
    return embeddings.tolist()


def embed_text(text: str, model_name: str) -> List[float]:
    """Encode a single multilingual text sample."""

    return embed_texts([text], model_name)[0]
