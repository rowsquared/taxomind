"""Embedding utilities that keep multilingual parity across components."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

import logging

import numpy as np
from sentence_transformers import SentenceTransformer



def load_embedding_model(
    model_name: str,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> SentenceTransformer:
    """Load a SentenceTransformer model with safe remote code handling."""

    logger = logging.getLogger(__name__)
    logger.info("Loading embedding model: %s", model_name)

    try:
        kwargs = {}
        if cache_dir:
            kwargs["cache_folder"] = cache_dir
        if local_files_only:
            kwargs["local_files_only"] = True
        model = SentenceTransformer(model_name, trust_remote_code=True, **kwargs)
        logger.info("Model loaded with trust_remote_code=True: %s", model_name)
    except Exception as exc:
        logger.warning("Failed with trust_remote_code=True: %s", exc)
        model = SentenceTransformer(model_name, **kwargs)
        logger.info("Model loaded without trust_remote_code: %s", model_name)

    return model


def apply_input_prefix(
    texts: Iterable[str], input_prefix: Optional[str]
) -> List[str]:
    """Prefix non-empty inputs for models that require task labels."""
    if not input_prefix:
        return list(texts)
    prefixed: List[str] = []
    for text in texts:
        text_stripped = str(text).strip()
        if text_stripped:
            prefixed.append(f"{input_prefix}{text_stripped}")
        else:
            prefixed.append(str(text))
    return prefixed


def encode_texts(
    embedding_model: Any,
    texts: Sequence[str],
    embed_all: bool,
    input_prefix: Optional[str] = None,
    batch_size: int = 32,
    show_progress_bar: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    Encode texts with a SentenceTransformer model.

    Returns:
        (embeddings, indices) where indices map embeddings to input rows.
    """
    text_list = list(texts)

    if embed_all:
        embed_texts = apply_input_prefix(text_list, input_prefix)
        indices = list(range(len(text_list)))
    else:
        embed_texts = []
        indices = []
        for idx, text in enumerate(text_list):
            text_stripped = str(text).strip()
            if text_stripped:
                embed_texts.append(text_stripped)
                indices.append(idx)
        embed_texts = apply_input_prefix(embed_texts, input_prefix)

    if not embed_texts:
        return np.empty((0, 0), dtype=np.float32), indices

    embeddings = embedding_model.encode(
        embed_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    return embeddings, indices
