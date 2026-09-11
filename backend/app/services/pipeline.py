"""End-to-end pipeline orchestrator.

Search flow (mirrors Fig 4.1/4.2/4.5 of the report):

  query -> QueryProcessor -> [real-time arXiv ingest] -> HybridRetriever
        -> Reranker -> (corrective retry if low confidence)
        -> lazy load+chunk+embed for top-N -> results

QA flow adds chunk retrieval from FAISS and LLM generation with sources.
"""
from __future__ import annotations

import json
import logging
import time

from sqlalchemy import func, select

from backend.app.core.config import settings
from backend.app.db.models import Chunk, Paper, QueryLog
from backend.app.db.session import db_session, init_db
from backend.app.services.arxiv_client import search_arxiv, upsert_papers
from backend.app.services.documents.chunker import chunk_document
from backend.app.services.documents.loader import load_document
from backend.app.services.embeddings import get_embedder
from backend.app.services.qa.engine import QAEngine, QAResult
from backend.app.services.query_processor import ProcessedQuery, process_query
from backend.app.services.recommender import Recommender
from backend.app.services.reranker import Reranker
from backend.app.services.retrieval.bm25_retriever import BM25Retriever
from backend.app.services.retrieval.dense_retriever import DenseRetriever
from backend.app.services.retrieval.hybrid import HybridRetriever
from backend.app.services.vector_store import FaissStore

logger = logging.getLogger(__name__)


class SearchPipeline:
    """Singleton-ish orchestrator wired at app startup."""

    def __init__(self) -> None:
        init_db()
        self.embedder = get_embedder()
        self.paper_store = FaissStore("papers", self.embedder.dim, settings.index_dir)
        self.chunk_store = FaissStore("chunks", self.embedder.dim, settings.index_dir)
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(self.paper_store, self.embedder)
        self.hybrid = HybridRetriever(self.bm25, self.dense)
        self.reranker = Reranker()
        self.qa = QAEngine(self.chunk_store, self.embedder, self.reranker)
        self.recommender = Recommender(self.paper_store)
        self._warm_bm25_from_db()

    # ------------------------------------------------------------- warmup
    def _warm_bm25_from_db(self) -> None:
        """BM25 is in-memory; rebuild it from persisted metadata on boot."""
        with db_session() as s:
            papers = s.execute(select(Paper)).scalars().all()
            if papers:
                self.hybrid.add_papers([p.to_dict() for p in papers])
                logger.info("Warmed retrievers with %d papers from DB", len(papers))

    # ------------------------------------------------------------- ingest
    def ingest_metadata(self, metas) -> int:
        """Upsert paper metadata (from arXiv or bulk scripts) into DB+indexes."""
        with db_session() as s:
            papers = upsert_papers(s, metas)
            dicts = [p.to_dict() for p in papers]
        added = self.hybrid.add_papers(dicts)
        if added:
            self.paper_store.save()
        return added

    # ------------------------------------------------------------- search
    def search(
        self,
        query: str,
        top_k: int | None = None,
        fetch_arxiv: bool = True,
        lazy_load: bool = True,
        session_id: str = "",
    ) -> dict:
        t0 = time.perf_counter()
        top_k = top_k or settings.rerank_top_k
        pq = process_query(query)

        # 1. Real-time ingestion: pull fresh candidates from the arXiv API
        if fetch_arxiv:
            metas = search_arxiv(pq.sparse_query)
            if metas:
                self.ingest_metadata(metas)

        # 2-4. hybrid retrieve -> rerank -> corrective retry
        ranked = self._retrieve_and_rerank(pq, top_k)
        if (
            settings.corrective_retrieval
            and not Reranker.confident(ranked)
            and pq.expansions == []
        ):
            logger.info("Low retrieval confidence; corrective re-query")
            pq2 = process_query(query, expand=True)
            pq2.sparse_query = f"{pq2.sparse_query} {' '.join(pq2.keywords[:5])}"
            if fetch_arxiv:
                metas = search_arxiv(pq2.sparse_query)
                if metas:
                    self.ingest_metadata(metas)
            retry = self._retrieve_and_rerank(pq2, top_k)
            if retry and (
                not ranked
                or retry[0]["rerank_score"] > ranked[0]["rerank_score"]
            ):
                ranked = retry

        # 5. Lazy loading for the top-N papers
        if lazy_load:
            for r in ranked[: settings.lazy_load_top_n]:
                try:
                    self.ensure_paper_indexed(r["arxiv_id"])
                except Exception as exc:  # reliability NFR: never crash
                    logger.warning("lazy load failed for %s: %s", r["arxiv_id"], exc)

        latency = (time.perf_counter() - t0) * 1000
        self._log(session_id, "search", query, "", [r["arxiv_id"] for r in ranked], latency)
        return {
            "query": pq.normalized,
            "expansions": pq.expansions,
            "alpha_hint": round(1 - pq.technicality, 3),
            "latency_ms": round(latency, 1),
            "results": [
                {
                    "arxiv_id": r["arxiv_id"],
                    "title": r["title"],
                    "abstract": r["abstract"],
                    "authors": r.get("authors", ""),
                    "categories": r.get("categories", ""),
                    "published": r.get("published", ""),
                    "pdf_status": r.get("pdf_status", "none"),
                    "score": round(r["rerank_score"], 4),
                }
                for r in ranked
            ],
        }

    def _retrieve_and_rerank(self, pq: ProcessedQuery, top_k: int) -> list[dict]:
        fused = self.hybrid.retrieve(pq)
        if not fused:
            return []
        ids = [pid for pid, _ in fused]
        scores = dict(fused)
        with db_session() as s:
            rows = s.execute(select(Paper).where(Paper.arxiv_id.in_(ids))).scalars().all()
        by_id = {p.arxiv_id: p for p in rows}
        candidates = []
        for pid in ids:
            p = by_id.get(pid)
            if not p:
                continue
            candidates.append(
                {
                    **p.to_dict(),
                    "text": f"{p.title}. {p.abstract}",
                    "fused_score": scores[pid],
                }
            )
        return self.reranker.rerank(pq.normalized, candidates, top_k=top_k)

    # ---------------------------------------------------------- lazy load
    def ensure_paper_indexed(self, arxiv_id: str) -> int:
        """Lazy pipeline for one paper: download -> parse -> chunk -> embed.

        Idempotent: returns existing chunk count if already embedded
        (no duplicate indexing NFR).
        """
        with db_session() as s:
            paper = s.execute(
                select(Paper).where(Paper.arxiv_id == arxiv_id)
            ).scalar_one_or_none()
            if paper is None:
                raise ValueError(f"unknown paper {arxiv_id}")
            if paper.pdf_status == "embedded" and paper.n_chunks > 0:
                return paper.n_chunks

            doc = load_document(paper)
            chunks = chunk_document(doc)
            if not chunks:
                paper.pdf_status = "failed"
                return 0

            texts = [c.text for c in chunks]
            vecs = self.embedder.encode(texts)
            metas = [
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                }
                for c in chunks
            ]
            faiss_ids = self.chunk_store.add(vecs, metas)
            # replace any stale chunk rows, then persist provenance
            s.query(Chunk).filter(Chunk.paper_id == paper.id).delete()
            for c, fid in zip(chunks, faiss_ids):
                s.add(
                    Chunk(
                        paper_id=paper.id,
                        chunk_index=c.chunk_index,
                        section=c.section,
                        text=c.text,
                        faiss_id=fid,
                        embedded=True,
                    )
                )
            paper.n_chunks = len(chunks)
            paper.pdf_status = "embedded"
        self.chunk_store.save()
        logger.info("Indexed %s: %d chunks (%s)", arxiv_id, len(chunks), doc.source)
        return len(chunks)

    # ---------------------------------------------------------------- QA
    def ask_single(
        self, question: str, arxiv_id: str,
        history: list[dict] | None = None, session_id: str = "",
    ) -> QAResult:
        t0 = time.perf_counter()
        self.ensure_paper_indexed(arxiv_id)
        with db_session() as s:
            p = s.execute(
                select(Paper).where(Paper.arxiv_id == arxiv_id)
            ).scalar_one_or_none()
        res = self.qa.answer_single(
            question, arxiv_id, title=p.title if p else "", history=history
        )
        self._log(
            session_id, "qa_single", question, res.answer, res.papers,
            (time.perf_counter() - t0) * 1000,
        )
        return res

    def ask_multi(
        self, question: str, paper_ids: list[str] | None = None,
        history: list[dict] | None = None, session_id: str = "",
    ) -> QAResult:
        t0 = time.perf_counter()
        if not paper_ids:
            # retrieval decides which papers matter for this question
            hit = self.search(question, fetch_arxiv=False, lazy_load=True)
            paper_ids = [r["arxiv_id"] for r in hit["results"][: settings.lazy_load_top_n]]
        titles = {}
        with db_session() as s:
            for pid in paper_ids:
                try:
                    self.ensure_paper_indexed(pid)
                except Exception as exc:
                    logger.warning("skip %s: %s", pid, exc)
                p = s.execute(
                    select(Paper).where(Paper.arxiv_id == pid)
                ).scalar_one_or_none()
                if p:
                    titles[pid] = p.title
        res = self.qa.answer_multi(question, paper_ids, titles=titles, history=history)
        self._log(
            session_id, "qa_multi", question, res.answer, res.papers,
            (time.perf_counter() - t0) * 1000,
        )
        return res

    # -------------------------------------------------------------- admin
    def stats(self) -> dict:
        with db_session() as s:
            n_papers = s.execute(select(func.count(Paper.id))).scalar() or 0
            n_chunks = s.execute(select(func.count(Chunk.id))).scalar() or 0
            n_queries = s.execute(select(func.count(QueryLog.id))).scalar() or 0
        return {
            "papers": n_papers,
            "chunks": n_chunks,
            "queries": n_queries,
            "bm25_docs": self.bm25.size,
            "paper_vectors": self.paper_store.ntotal,
            "chunk_vectors": self.chunk_store.ntotal,
            "embedding_backend": type(self.embedder).__name__,
            "embedding_dim": self.embedder.dim,
            "llm_backend": settings.llm_backend,
        }

    def rebuild_indexes(self) -> dict:
        """Admin: rebuild vector + BM25 indexes from the metadata DB."""
        self.paper_store.clear()
        self.chunk_store.clear()
        self.bm25.clear()
        with db_session() as s:
            papers = s.execute(select(Paper)).scalars().all()
            dicts = [p.to_dict() for p in papers]
            for p in papers:  # chunks will be lazily re-embedded on demand
                p.pdf_status = "none" if p.pdf_status != "failed" else "failed"
                p.n_chunks = 0
            s.query(Chunk).delete()
        self.hybrid.add_papers(dicts)
        self.paper_store.save()
        self.paper_store.maybe_train_ivf()
        return self.stats()

    def _log(self, session_id, mode, query, answer, papers, latency_ms) -> None:
        try:
            with db_session() as s:
                s.add(
                    QueryLog(
                        session_id=session_id, mode=mode, query=query,
                        answer=answer[:5000], sources=json.dumps(papers),
                        latency_ms=latency_ms,
                    )
                )
        except Exception:  # logging must never break a request
            logger.debug("query log write failed", exc_info=True)


_pipeline: SearchPipeline | None = None


def get_pipeline() -> SearchPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SearchPipeline()
    return _pipeline
