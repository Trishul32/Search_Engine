"""Second-stage reranking of fused candidates.

Primary: a cross-encoder (ms-marco-MiniLM-L-6-v2 by default) scoring
(query, title+abstract) pairs jointly — per SciRerankBench (ref [4]),
cross-encoders substantially beat BM25 ordering on scientific text.

Fallback: if torch/CE weights are unavailable, a lexical-overlap heuristic
blended with fused scores keeps the pipeline functional (graceful
degradation is an explicit NFR).
"""
from __future__ import annotations

import logging
import math
import re
from functools import lru_cache

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder

        m = CrossEncoder(settings.reranker_model, max_length=512)
        logger.info("Loaded cross-encoder %s", settings.reranker_model)
        return m
    except Exception as exc:
        logger.warning(
            "Cross-encoder unavailable (%s); using lexical fallback reranker", exc
        )
        return None


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


_TOKEN = re.compile(r"[a-z0-9]+")


def _lexical_score(query: str, doc: str) -> float:
    q = set(_TOKEN.findall(query.lower()))
    d = set(_TOKEN.findall(doc.lower()))
    if not q or not d:
        return 0.0
    overlap = len(q & d) / len(q)
    bigrams_q = {f"{a} {b}" for a, b in zip(*(lambda t: (t, t[1:]))(
        _TOKEN.findall(query.lower())))}
    text = " ".join(_TOKEN.findall(doc.lower()))
    phrase = sum(1 for bg in bigrams_q if bg in text) / max(1, len(bigrams_q))
    return 0.7 * overlap + 0.3 * phrase


class Reranker:
    """Reorders (paper, fused_score) candidates; also used for chunk rerank."""

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        text_key: str = "text",
        top_k: int | None = None,
    ) -> list[dict]:
        """candidates: dicts containing `text_key` and optional 'fused_score'.

        Returns candidates sorted desc with 'rerank_score' in [0, 1] added.
        """
        if not candidates:
            return []
        top_k = top_k or settings.rerank_top_k
        if not settings.reranker_enabled:
            return candidates[:top_k]

        model = _load_cross_encoder()
        if model is not None:
            pairs = [(query, c.get(text_key, "")[:2000]) for c in candidates]
            raw = model.predict(pairs, show_progress_bar=False)
            for c, s in zip(candidates, raw):
                c["rerank_score"] = _sigmoid(float(s))
        else:
            fused = [c.get("fused_score", 0.0) for c in candidates]
            hi = max(fused) or 1.0
            for c, f in zip(candidates, fused):
                lex = _lexical_score(query, c.get(text_key, ""))
                # relevance dominates; fused score is only a mild tiebreaker
                c["rerank_score"] = 0.85 * lex + 0.15 * (f / hi)

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def confident(ranked: list[dict]) -> bool:
        """Confidence gate for corrective retrieval (Corrective-RAG style)."""
        if not ranked:
            return False
        return ranked[0].get("rerank_score", 0.0) >= settings.corrective_score_threshold
