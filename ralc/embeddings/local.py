"""Local embedding backends for semantic retrieval."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors.

    Implementations should return an ``(n, dim)`` float array for ``n`` input
    texts. Vectors are expected to be L2-normalized, but callers must not rely
    on it: :class:`~ralc.retrieval.semantic.SemanticRetriever` normalizes
    internally so that an unnormalized embedder still ranks correctly.
    """

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    """Embedder backed by the sentence-transformers library.

    The library is imported lazily on first use, so the core package works
    without the optional ``embed`` extra installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "SentenceTransformerEmbedder requires sentence-transformers. "
                    'Install the optional extra: pip install "ralc[embed]"'
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        vectors = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)
