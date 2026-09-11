"""Sparse retrieval: BM25 (Okapi) over paper title + abstract.

Titles are repeated in the token stream to weight them higher — a standard
field-boost trick that noticeably improves paper-level precision.
"""
from __future__ import annotations

import logging
import re
import threading

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9\-]+")
_TITLE_BOOST = 2  # title tokens appear (boost+1)x


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._paper_ids: list[str] = []
        self._docs: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._known: set[str] = set()

    def add_papers(self, papers: list[dict]) -> int:
        """papers: [{arxiv_id, title, abstract}]. Dedup + incremental rebuild."""
        added = 0
        with self._lock:
            for p in papers:
                pid = p["arxiv_id"]
                if pid in self._known:
                    continue
                toks = tokenize(p.get("title", "")) * (_TITLE_BOOST + 1)
                toks += tokenize(p.get("abstract", ""))
                if not toks:
                    continue
                self._paper_ids.append(pid)
                self._docs.append(toks)
                self._known.add(pid)
                added += 1
            if added:
                # rank_bm25 has no incremental add; rebuild is O(corpus) and
                # cheap at metadata scale (tokenized docs are cached).
                self._bm25 = BM25Okapi(self._docs)
        return added

    def retrieve(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        with self._lock:
            if not self._bm25:
                return []
            scores = self._bm25.get_scores(tokenize(query))
            order = scores.argsort()[::-1][:top_k]
            return [
                (self._paper_ids[i], float(scores[i])) for i in order if scores[i] > 0
            ]

    @property
    def size(self) -> int:
        return len(self._paper_ids)

    def clear(self) -> None:
        with self._lock:
            self._paper_ids, self._docs, self._known = [], [], set()
            self._bm25 = None
