"""Embedding helpers for long-term conversation retrieval.

Model choice is intentionally explicit for reproducibility:
sentence-transformers/all-MiniLM-L6-v2.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and cache the sentence-transformers model once per process."""
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "sentence-transformers is required for embedding retrieval. "
            "Install it with: pip install sentence-transformers --break-system-packages"
        ) from error
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_text(text: str) -> list[float]:
    """Return a normalized embedding for one text."""
    model = get_embedding_model()
    vector = model.encode(
        text or "",
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vector.astype(float).tolist()


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    """Return normalized embeddings for a batch of texts."""
    model = get_embedding_model()
    vectors = model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype(float).tolist()


def top_n_by_cosine(
    query_embedding: list[float],
    candidates: Iterable[tuple[int, list[float], dict]],
    top_k: int,
) -> list[tuple[float, int, dict]]:
    """Rank normalized candidate vectors by dot product cosine similarity."""
    scored: list[tuple[float, int, dict]] = []
    for index, embedding, item in candidates:
        score = sum(q * c for q, c in zip(query_embedding, embedding))
        scored.append((score, index, item))
    return sorted(scored, key=lambda x: (-x[0], -x[1]))[:top_k]
