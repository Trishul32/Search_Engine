"""Hybrid retrieval: BM25 and dense retrieval run in parallel, then fused."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from backend.app.core.config import settings
from backend.app.services.query_processor import ProcessedQuery
from backend.app.services.retrieval.bm25_retriever import BM25Retriever
from backend.app.services.retrieval.dense_retriever import DenseRetriever
from backend.app.services.retrieval.fusion import (
    dynamic_alpha,
    rrf_fuse,
    weighted_fuse,
)

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever):
        self.bm25 = bm25
        self.dense = dense
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retriever")

    def add_papers(self, papers: list[dict]) -> int:
        n1 = self.bm25.add_papers(papers)
        n2 = self.dense.add_papers(papers)
        return max(n1, n2)

    def retrieve(
        self, pq: ProcessedQuery, top_k: int | None = None
    ) -> list[tuple[str, float]]:
        top_k = top_k or settings.retrieval_top_k
        f_sparse = self._pool.submit(self.bm25.retrieve, pq.sparse_query, top_k)
        f_dense = self._pool.submit(self.dense.retrieve, pq.dense_query, top_k)
        sparse, dense = f_sparse.result(), f_dense.result()

        if settings.fusion_method == "weighted":
            alpha = (
                dynamic_alpha(pq, settings.fusion_alpha)
                if settings.dynamic_alpha
                else settings.fusion_alpha
            )
            fused = weighted_fuse(sparse, dense, alpha)
        else:
            # RRF, optionally biased by duplicating the modality that the
            # dynamic-alpha analysis favors for this query.
            lists = [sparse, dense]
            if settings.dynamic_alpha:
                a = dynamic_alpha(pq, settings.fusion_alpha)
                if a >= 0.65:
                    lists.append(dense)
                elif a <= 0.35:
                    lists.append(sparse)
            fused = rrf_fuse(lists, k=settings.rrf_k)

        logger.debug(
            "hybrid retrieve '%s': sparse=%d dense=%d fused=%d",
            pq.normalized[:60], len(sparse), len(dense), len(fused),
        )
        return fused[:top_k]
