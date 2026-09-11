"""Related-paper recommendations (Review-2 deck objective #7).

Score = cosine similarity of paper embeddings (title+abstract vectors from
the paper-level FAISS index) blended with arXiv category overlap.
"""
from __future__ import annotations

import logging

from sqlalchemy import select

from backend.app.db.models import Paper
from backend.app.db.session import db_session
from backend.app.services.vector_store import FaissStore

logger = logging.getLogger(__name__)

_CATEGORY_WEIGHT = 0.15


class Recommender:
    def __init__(self, paper_store: FaissStore):
        self.paper_store = paper_store

    def _vector_id(self, arxiv_id: str) -> int | None:
        for vid, meta in self.paper_store.all_meta().items():
            if meta.get("arxiv_id") == arxiv_id:
                return vid
        return None

    def similar_papers(self, arxiv_id: str, top_k: int = 5) -> list[dict]:
        vid = self._vector_id(arxiv_id)
        if vid is None:
            return []
        vec = self.paper_store.get_vector(vid)
        if vec is None:
            return []
        hits = self.paper_store.search(
            vec,
            top_k=top_k + 1,
            filter_fn=lambda m: m.get("arxiv_id") != arxiv_id,
        )

        with db_session() as s:
            src = s.execute(
                select(Paper).where(Paper.arxiv_id == arxiv_id)
            ).scalar_one_or_none()
            src_cats = set((src.categories or "").split(",")) if src else set()

            out = []
            for _, sim, meta in hits[:top_k]:
                p = s.execute(
                    select(Paper).where(Paper.arxiv_id == meta["arxiv_id"])
                ).scalar_one_or_none()
                if not p:
                    continue
                cats = set((p.categories or "").split(","))
                jacc = (
                    len(src_cats & cats) / len(src_cats | cats)
                    if src_cats and cats
                    else 0.0
                )
                out.append(
                    {
                        **p.to_dict(),
                        "score": round(
                            (1 - _CATEGORY_WEIGHT) * sim + _CATEGORY_WEIGHT * jacc, 4
                        ),
                    }
                )
            out.sort(key=lambda d: d["score"], reverse=True)
            return out
