"""Embedding generation.

Production backend: SentenceTransformers (all-MiniLM-L6-v2 by default;
swap to BGE etc. via EMBEDDING_MODEL). A deterministic hashed n-gram
fallback keeps the whole pipeline runnable — and testable — on machines
without torch or network access to model hubs. Both produce L2-normalized
vectors so FAISS inner product == cosine similarity.
"""
from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache

import numpy as np

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class BaseEmbedder:
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info("Loaded embedding model %s (dim=%d)", model_name, self.dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")


class HashEmbedder(BaseEmbedder):
    """Deterministic bag-of-hashed-ngrams embedding (no ML dependencies).

    Not a semantic model — it captures lexical/character overlap — but it is
    stable, fast, and lets the retrieval/QA plumbing run end-to-end where
    torch is unavailable. Never use in production; set
    EMBEDDING_BACKEND=sentence-transformers there.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        grams = list(toks)
        grams += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for t in toks:
            if len(t) >= 5:
                grams += [t[i : i + 4] for i in range(len(t) - 3)]
        return grams

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for g in self._tokens(text):
                h = int.from_bytes(
                    hashlib.blake2b(g.encode(), digest_size=8).digest(), "little"
                )
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[i, idx] += sign
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    if settings.embedding_backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder(settings.embedding_model)
        except Exception as exc:
            logger.warning(
                "sentence-transformers unavailable (%s); falling back to "
                "hash embedder — retrieval quality will be lexical only.",
                exc,
            )
    return HashEmbedder(settings.embedding_dim)
