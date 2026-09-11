"""Dense semantic retrieval over the paper-level FAISS index."""
from __future__ import annotations

import logging

from backend.app.services.embeddings import BaseEmbedder
from backend.app.services.vector_store import FaissStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, store: FaissStore, embedder: BaseEmbedder):
        self.store = store
        self.embedder = embedder

    def add_papers(self, papers: list[dict]) -> int:
        """Embed title+abstract for papers not yet in the index (dedup)."""
        known = {m.get("arxiv_id") for m in self.store.all_meta().values()}
        new = [p for p in papers if p["arxiv_id"] not in known]
        if not new:
            return 0
        texts = [f"{p.get('title','')}. {p.get('abstract','')}" for p in new]
        vecs = self.embedder.encode(texts)
        self.store.add(
            vecs,
            [{"arxiv_id": p["arxiv_id"], "title": p.get("title", "")} for p in new],
        )
        return len(new)

    def retrieve(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        if self.store.ntotal == 0:
            return []
        qv = self.embedder.encode_one(query)
        hits = self.store.search(qv, top_k=top_k)
        return [(m["arxiv_id"], score) for _, score, m in hits]
