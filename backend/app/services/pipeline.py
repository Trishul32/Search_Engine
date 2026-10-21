"""End-to-end pipeline orchestrator.

Search flow (mirrors Fig 4.1/4.2/4.5 of the report) — metadata only, fast:

  query -> QueryProcessor (topic / exclusions / fields)
        -> [structured real-time arXiv query: ti/abs fields, cat:, ANDNOT]
        -> HybridRetriever (+ live arXiv matches always in the pool)
        -> exclusion & field filters -> Reranker (positive topic only)
        -> (corrective retry) -> best-matching abstract sentence -> results

PDF download + parsing + chunk embedding are deferred ("lazy loading"):
they run only inside ask_single/ask_multi, i.e. when the user selects a
paper for QA, via the idempotent ensure_paper_indexed().
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import func, select

from backend.app.core.config import settings
from backend.app.db.models import Chunk, Paper, QueryLog
from backend.app.db.session import db_session, init_db
from backend.app.services.arxiv_client import (
    PaperMeta,
    search_arxiv,
    upsert_papers,
    wait_for_slot,
)
from backend.app.services.documents.chunker import chunk_document
from backend.app.services.documents.loader import load_document
from backend.app.services.embeddings import get_embedder
from backend.app.services.qa.engine import QAEngine, QAResult
from backend.app.services.query_processor import (
    ProcessedQuery,
    build_arxiv_query,
    process_query,
)
from backend.app.services.relevance import (
    best_evidence_sentence,
    concept_coverage,
    exclusion_hit,
    field_match,
    split_sentences,
)
from backend.app.services.query_rewriter import needs_rewrite, rewrite_query
from backend.app.services.knowledge_graph import build_graph, load_papers
from backend.app.services.reading_path import ReadingAdvisor
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
        sig = self.embedder.signature
        self.paper_store = FaissStore("papers", self.embedder.dim, settings.index_dir, signature=sig)
        self.chunk_store = FaissStore("chunks", self.embedder.dim, settings.index_dir, signature=sig)
        if self.chunk_store.reset_reason:
            self._reset_chunk_state()
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(self.paper_store, self.embedder)
        self.hybrid = HybridRetriever(self.bm25, self.dense)
        self.reranker = Reranker()
        self.qa = QAEngine(self.chunk_store, self.embedder, self.reranker)
        self.recommender = Recommender(self.paper_store)
        self._bg = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline-bg")
        self.reading = ReadingAdvisor(self)
        self._warm_bm25_from_db()
        if settings.auto_seed_if_empty and self.bm25.size == 0:
            n = self.seed_from_file()
            if n:
                logger.info("Cold start: seeded %d bundled papers", n)
        self._warm_models()

    def _warm_models(self) -> None:
        """Load model weights at startup, not on the first user query.

        The embedder loads at construction, but the cross-encoder used to
        load lazily inside the first rerank call — making the first search
        pay 5–15s of model loading. A tiny forward pass through both here
        moves that cost to app startup.
        """
        try:
            self.embedder.encode_one("warmup")
            self.reranker.rerank(
                "warmup",
                [{"arxiv_id": "_", "text": "warmup", "fused_score": 1.0}],
                top_k=1,
            )
            logger.info("Models warmed (embedder + reranker)")
        except Exception:  # warming is best-effort, never fatal
            logger.debug("model warmup failed", exc_info=True)

    # ------------------------------------------------------------- warmup
    def _warm_bm25_from_db(self) -> None:
        """BM25 is in-memory; rebuild it from persisted metadata on boot.

        Also re-embeds papers missing from the dense index — e.g. after the
        embedding model changed and the old index was discarded.
        """
        with db_session() as s:
            papers = s.execute(select(Paper)).scalars().all()
            dicts = [p.to_dict() for p in papers]
        if dicts:
            before = self.paper_store.ntotal
            self.hybrid.add_papers(dicts)
            if self.paper_store.ntotal != before:
                self.paper_store.save()
                logger.info(
                    "Re-embedded %d papers into the dense index",
                    self.paper_store.ntotal - before,
                )
            logger.info("Warmed retrievers with %d papers from DB", len(dicts))

    def _reset_chunk_state(self) -> None:
        """Chunk vectors from another embedder are unusable: forget them so
        papers are lazily re-chunked/re-embedded (cached PDFs are reused)."""
        with db_session() as s:
            s.query(Chunk).delete()
            for p in s.execute(select(Paper).where(Paper.pdf_status == "embedded")).scalars():
                p.pdf_status = "none"
                p.n_chunks = 0
        self.chunk_store.save()
        logger.info("Chunk index reset (%s)", self.chunk_store.reset_reason)

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

    def seed_from_file(self, path=None) -> int:
        """Ingest the bundled offline seed corpus (JSONL of paper metadata).

        Local ingestion workflow that never touches the network — used for
        cold starts and by `scripts/ingest.py --seed`. Idempotent: existing
        arxiv_ids are skipped by the usual dedup.
        """
        from pathlib import Path

        from backend.app.core.config import PROJECT_ROOT

        path = Path(path or settings.seed_file or (PROJECT_ROOT / "seed" / "arxiv_seed.jsonl"))
        if not path.exists():
            logger.warning("seed file not found: %s", path)
            return 0
        metas: list[PaperMeta] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("arxiv_id") and rec.get("title"):
                metas.append(PaperMeta(
                    arxiv_id=rec["arxiv_id"],
                    title=rec["title"],
                    abstract=rec.get("abstract", ""),
                    authors=rec.get("authors", ""),
                    categories=rec.get("categories", ""),
                    published=rec.get("published", ""),
                    pdf_url=rec.get("pdf_url", ""),
                ))
        return self.ingest_metadata(metas)

    # ------------------------------------------------------------- search
    def search(
        self,
        query: str,
        top_k: int | None = None,
        fetch_arxiv: bool = True,
        session_id: str = "",
        exclude: list[str] | None = None,
        field: str | None = None,
        smart: bool = True,
    ) -> dict:
        """Metadata-only search: understand -> live fetch -> hybrid
        retrieve -> filter -> rerank -> evidence.

        Deliberately performs NO PDF download or full-text indexing — that
        work is deferred until a paper is actually selected for QA
        (``ask_single`` / ``ask_multi``), keeping search fast regardless of
        corpus or PDF sizes.
        """
        t0 = time.perf_counter()
        top_k = top_k or settings.rerank_top_k
        pq = self._understand(query, exclude, field)

        if not pq.keywords and not pq.fields:
            return self._empty_response(
                pq, t0,
                "Your query only says what to exclude. Add a topic or a field "
                "(e.g. 'computer science without AI', 'graph algorithms "
                "without machine learning').",
            )

        # 0. Conversational query? Ask the LLM for academic vocabulary in
        #    the background, overlapping with the first arXiv request.
        rewrite_job = (
            self._bg.submit(rewrite_query, pq.topic, pq.exclusions)
            if smart and needs_rewrite(pq) else None
        )

        # 1. Real-time ingestion: a *structured* arXiv query (title/abstract
        #    fields, category filters, ANDNOT exclusions), metadata only.
        t_fetch = 0.0
        fetch_error: str | None = None
        live_ids: list[str] = []
        live_queries: list[str] = []
        metas: list[PaperMeta] = []
        tf = time.perf_counter()
        if fetch_arxiv:
            metas, fetch_error, live_queries = self._fetch_live(
                pq, thin_ok=rewrite_job is not None
            )
        if rewrite_job is not None:
            self._apply_rewrite(pq, rewrite_job)
            if fetch_arxiv and pq.rewrite is not None and pq.rewrite.keywords:
                rw_query = build_arxiv_query(
                    pq, relaxed=True,
                    concepts=[k.lower().split() for k in pq.rewrite.keywords],
                )
                if rw_query and wait_for_slot(max_wait_s=3.5):
                    more, err2 = search_arxiv(rw_query)
                    live_queries.append(rw_query)
                    seen = {m.arxiv_id for m in metas}
                    # the user's own wording keeps priority in live order
                    metas = metas + [m for m in more if m.arxiv_id not in seen]
                    if err2 and not metas:
                        fetch_error = err2
        if metas:
            self.ingest_metadata(metas)
            live_ids = list(dict.fromkeys(m.arxiv_id for m in metas))
        if fetch_arxiv or rewrite_job is not None:
            t_fetch = (time.perf_counter() - tf) * 1000

        # 2-4. hybrid retrieve (+ live candidates) -> filter -> rerank ->
        # corrective retry (local only; the fetch above already ingested
        # fresh candidates)
        tr = time.perf_counter()
        stats: dict = {}
        ranked = self._retrieve_and_rerank(pq, top_k, extra_ids=live_ids, stats=stats)
        if (
            settings.corrective_retrieval
            and not Reranker.confident(ranked)
            and pq.expansions == []
            and not pq.browse_only
        ):
            logger.info("Low retrieval confidence; corrective re-query")
            pq2 = self._understand(query, exclude, field)
            if pq.rewrite is not None:
                pq2.rewrite, pq2.dense_query = pq.rewrite, pq.dense_query
                pq2.requires_intersection = False
            pq2.sparse_query = f"{pq2.sparse_query} {' '.join(pq2.keywords[:5])}"
            retry = self._retrieve_and_rerank(pq2, top_k, extra_ids=live_ids)
            if retry and (
                not ranked
                or retry[0]["rerank_score"] > ranked[0]["rerank_score"]
            ):
                ranked = retry
        self._attach_evidence(pq, ranked)
        t_retrieve = (time.perf_counter() - tr) * 1000

        latency = (time.perf_counter() - t0) * 1000
        self._log(session_id, "search", query, "", [r["arxiv_id"] for r in ranked], latency)
        notice = None
        if fetch_error:
            notice = f"Live arXiv fetch skipped — {fetch_error}. "
            notice += (
                "The local corpus is empty, so there is nothing to search yet: "
                "check your network/proxy access to export.arxiv.org, or bulk-"
                "ingest papers with scripts/ingest.py."
                if self.bm25.size == 0
                else "Showing results from the local corpus only."
            )
        elif not ranked and self.bm25.size == 0:
            notice = (
                "The local corpus is empty and the live fetch returned no "
                "matches — try a broader query, or ingest papers with "
                "scripts/ingest.py."
            )
        elif not ranked and stats.get("excluded"):
            notice = (
                f"All {stats['excluded']} candidate papers matched your "
                "exclusions, so none are shown. Try a more specific topic."
            )
        elif not ranked:
            notice = "No papers in the available corpus matched the query evidence."
        return {
            "notice": notice,
            "query": pq.normalized,
            "interpretation": pq.interpretation(),
            "excluded_count": stats.get("excluded", 0),
            "live_query": live_queries,
            "expansions": pq.expansions,
            "alpha_hint": round(1 - pq.technicality, 3),
            "latency_ms": round(latency, 1),
            "timings_ms": {
                "arxiv_fetch": round(t_fetch, 1),
                "retrieve_rerank": round(t_retrieve, 1),
            },
            "results": [
                {
                    "arxiv_id": r["arxiv_id"],
                    "title": r["title"],
                    "abstract": r["abstract"],
                    "authors": r.get("authors", ""),
                    "categories": r.get("categories", ""),
                    "published": r.get("published", ""),
                    "pdf_status": r.get("pdf_status", "none"),
                    "score": (
                        None if pq.browse_only
                        else round(r.get("final_relevance_score", r["rerank_score"]), 4)
                    ),
                    "evidence": r.get("evidence", ""),
                    "live_match": r.get("live_rank") is not None,
                }
                for r in ranked
            ],
        }

    @staticmethod
    def _understand(query: str, exclude: list[str] | None, field: str | None) -> ProcessedQuery:
        pq = process_query(query, exclude=exclude, field_filter=field)
        if pq.has_filters:
            pq.pool_size = settings.retrieval_top_k * settings.filtered_pool_multiplier
        return pq

    def _empty_response(self, pq: ProcessedQuery, t0: float, notice: str) -> dict:
        return {
            "notice": notice,
            "query": pq.normalized,
            "interpretation": pq.interpretation(),
            "excluded_count": 0,
            "live_query": [],
            "expansions": pq.expansions,
            "alpha_hint": round(1 - pq.technicality, 3),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "timings_ms": {"arxiv_fetch": 0.0, "retrieve_rerank": 0.0},
            "results": [],
        }

    @staticmethod
    def _apply_rewrite(pq: ProcessedQuery, job) -> None:
        """Fold an LLM rewrite into the query (HyDE-style dense query)."""
        try:
            rw = job.result(timeout=settings.query_rewrite_timeout_s)
        except Exception:
            logger.info("query rewrite unavailable or timed out")
            return
        if rw is None or not (rw.keywords or rw.academic_query):
            return
        pq.rewrite = rw
        extra = " ".join(rw.keywords)
        pq.sparse_query = f"{pq.sparse_query} {extra} {rw.academic_query}".strip()
        pq.dense_query = ". ".join(
            x for x in (pq.topic, rw.academic_query, rw.pseudo_abstract) if x
        )
        # The rewrite only ADDS candidates and enriches the dense query.
        # Ranking evidence stays anchored to the user's own words (a small
        # LLM can drift: "making up facts" -> "knowledge distillation"),
        # but the all-concepts requirement is relaxed, since relevant
        # papers use the technical phrasing instead of the user's wording.
        pq.requires_intersection = False

    def _fetch_live(
        self, pq: ProcessedQuery, thin_ok: bool = False
    ) -> tuple[list[PaperMeta], str | None, list[str]]:
        """Strict structured arXiv query, plus a relaxed one if it is thin.

        Strict = every concept must appear in the title or abstract.
        Relaxed = any concept may appear (arXiv's relevance sort still puts
        papers matching more concepts first). Both keep field filters and
        exclusions, so the live results never reintroduce excluded topics.
        """
        strict = build_arxiv_query(pq)
        if not strict:
            return [], None, []
        sort_by = "submittedDate" if pq.browse_only else "relevance"
        metas, error = (
            search_arxiv(strict) if sort_by == "relevance"
            else search_arxiv(strict, sort_by=sort_by)
        )
        queries = [strict]
        if (
            not error
            and not thin_ok      # an LLM-rewrite query will follow instead
            and len(metas) < settings.arxiv_min_strict_results
            and (len(pq.concepts) > 1 or any(len(c) > 3 for c in pq.concepts))
        ):
            relaxed = build_arxiv_query(pq, relaxed=True)
            if relaxed and relaxed != strict and wait_for_slot():
                more, error2 = search_arxiv(relaxed)
                queries.append(relaxed)
                seen = {m.arxiv_id for m in metas}
                metas = metas + [m for m in more if m.arxiv_id not in seen]
                if error2 and not metas:
                    error = error2
        return metas, error, queries

    def _browse_ids(self, pq: ProcessedQuery, limit: int = 300) -> list[str]:
        """Local papers in the requested fields (topic-less field browse)."""
        with db_session() as s:
            rows = s.execute(
                select(Paper.arxiv_id, Paper.categories, Paper.published)
                .order_by(Paper.published.desc())
            ).all()
        out = []
        for arxiv_id, cats, _published in rows:
            if field_match(pq.fields, {"categories": cats or ""}):
                out.append(arxiv_id)
                if len(out) >= limit:
                    break
        return out

    def _retrieve_and_rerank(
        self,
        pq: ProcessedQuery,
        top_k: int,
        extra_ids: list[str] | None = None,
        stats: dict | None = None,
    ) -> list[dict]:
        fused = self.hybrid.retrieve(pq)
        scores = dict(fused)
        ids = [pid for pid, _ in fused]
        # Papers returned by the structured live query matched the user's
        # concepts in their *abstract or title* on arXiv's side; they always
        # get scored by the cross-encoder even if the local hybrid ranking
        # (built over the whole corpus) did not surface them.
        live_rank = {pid: rank for rank, pid in enumerate(extra_ids or [])}
        if pq.browse_only:
            ids += [pid for pid in self._browse_ids(pq) if pid not in scores]
        ids += [pid for pid in live_rank if pid not in scores and pid not in ids]
        if not ids:
            return []
        with db_session() as s:
            rows = s.execute(select(Paper).where(Paper.arxiv_id.in_(ids))).scalars().all()
        by_id = {p.arxiv_id: p for p in rows}
        candidates = []
        excluded = 0
        n_local = 0
        for pid in dict.fromkeys(ids):
            p = by_id.get(pid)
            if not p:
                continue
            paper = p.to_dict()
            # Hard filters from query understanding run BEFORE reranking:
            # they are cheap, and excluded papers must never be shown.
            if not field_match(pq.fields, paper):
                continue
            if exclusion_hit(pq.exclusions, paper):
                excluded += 1
                continue
            is_live = pid in live_rank
            if not is_live:
                if n_local >= settings.max_rerank_candidates:
                    continue
                n_local += 1
            candidates.append(
                {
                    **paper,
                    "text": f"{p.title}. {p.abstract}",
                    "fused_score": scores.get(pid, 0.0),
                    "live_rank": live_rank.get(pid),
                }
            )
        if stats is not None:
            stats["excluded"] = excluded
        if not candidates:
            return []
        if pq.browse_only:
            return self._browse_order(candidates, top_k)
        # The cross-encoder reads the query against the full title+abstract,
        # using only the *positive* topic: negated words would otherwise
        # count as evidence for the very papers the user excluded.
        ranked = self.reranker.rerank(
            pq.normalized, candidates, top_k=len(candidates)
        )
        for candidate in ranked:
            evidence = concept_coverage(
                pq.concepts,
                candidate.get("abstract", ""),
                candidate.get("categories", ""),
                title=candidate.get("title", ""),
            )
            candidate["concept_coverage"] = evidence

        if pq.requires_intersection and len(pq.concepts) > 1:
            ranked = [
                candidate for candidate in ranked
                if candidate["concept_coverage"]["minimum"]
                >= settings.retrieval_min_concept_coverage
            ]

        rerank_rank = {
            candidate["arxiv_id"]: rank
            for rank, candidate in enumerate(ranked)
        }
        evidence_ranked = sorted(
            (candidate for candidate in ranked
             if candidate["concept_coverage"]["score"] > 0.0),
            key=lambda candidate: (
                candidate["concept_coverage"]["minimum"],
                candidate["concept_coverage"]["score"],
                candidate["concept_coverage"]["title_score"],
            ),
            reverse=True,
        )
        evidence_rank = {
            candidate["arxiv_id"]: rank
            for rank, candidate in enumerate(evidence_ranked)
        }

        def final_score(candidate: dict) -> float:
            # Rank fusion of three relevance signals: cross-encoder reading
            # of title+abstract, per-concept text evidence, and arXiv's own
            # field-aware relevance for live matches (weaker weight).
            pid = candidate["arxiv_id"]
            score = 1 / (settings.rrf_k + rerank_rank[pid] + 1)
            if pid in evidence_rank:
                score += 1 / (settings.rrf_k + evidence_rank[pid] + 1)
            if candidate.get("live_rank") is not None:
                score += 0.5 / (settings.rrf_k + candidate["live_rank"] + 1)
            return score

        for candidate in ranked:
            candidate["final_relevance_score"] = final_score(candidate)
        ranked.sort(
            key=lambda c: (c["final_relevance_score"], c["rerank_score"]),
            reverse=True,
        )

        if pq.requested_year is not None:
            year = str(pq.requested_year)
            ranked = [
                candidate for candidate in ranked
                if str(candidate.get("published", "")).startswith(year)
            ]
        elif pq.recent or pq.browse_only:
            ranked.sort(
                key=lambda candidate: str(candidate.get("published", "")),
                reverse=True,
            )
        return ranked[:top_k]

    @staticmethod
    def _browse_order(candidates: list[dict], top_k: int) -> list[dict]:
        """Field browse without a topic ("computer science without AI").

        There is no topic to judge relevance against, so the cross-encoder
        is skipped: scoring every paper against the field name took 10-20s
        and its order was then discarded for a newest-first sort anyway.
        Newest first; on equal dates, arXiv's own order for live papers.
        """
        candidates.sort(
            key=lambda c: c["live_rank"] if c.get("live_rank") is not None else 10**6
        )
        candidates.sort(key=lambda c: str(c.get("published", "")), reverse=True)
        for c in candidates:
            c["rerank_score"] = 0.0
            c["final_relevance_score"] = None   # no relevance in browse mode
        return candidates[:top_k]

    def _attach_evidence(self, pq: ProcessedQuery, ranked: list[dict]) -> None:
        """Find the abstract sentence that best matches the query.

        Shown in the UI as "why this matched": the paper is judged by its
        abstract, not only its title, and users can verify that at a glance.
        """
        if not ranked or not pq.keywords:
            return
        per_paper = [split_sentences(r.get("abstract", "")) for r in ranked]
        flat = [s for sents in per_paper for s in sents]
        if not flat:
            return
        try:
            vecs = self.embedder.encode([pq.dense_query] + flat)
            qv, svecs = vecs[0], vecs[1:]
            sims = svecs @ qv
        except Exception:
            logger.debug("evidence embedding failed", exc_info=True)
            sims = None
        i = 0
        for r, sents in zip(ranked, per_paper):
            if not sents:
                continue
            if sims is None:
                r["evidence"] = best_evidence_sentence(pq.keywords, r.get("abstract", ""))
            else:
                local = sims[i:i + len(sents)]
                r["evidence"] = sents[int(local.argmax())]
            i += len(sents)

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
    def _paper_meta(self, ids: list[str]) -> tuple[dict[str, dict], dict[str, str]]:
        """Metadata (title, abstract, authors, date) + indexing status per id."""
        metas: dict[str, dict] = {}
        status: dict[str, str] = {}
        with db_session() as s:
            rows = s.execute(select(Paper).where(Paper.arxiv_id.in_(ids))).scalars().all()
            for p in rows:
                metas[p.arxiv_id] = p.to_dict()
                status[p.arxiv_id] = p.pdf_status + (f" ({p.parse_error})" if p.parse_error else "")
        for pid in ids:
            status.setdefault(pid, "unknown paper id")
        return metas, status

    def ask_single(
        self, question: str, arxiv_id: str,
        history: list[dict] | None = None, session_id: str = "",
    ) -> QAResult:
        return QAEngine._collect(self.ask_single_stream(question, arxiv_id, history, session_id))

    def ask_single_stream(
        self, question: str, arxiv_id: str,
        history: list[dict] | None = None, session_id: str = "",
    ):
        """Chat with one paper. Yields events: status / sources / token / final."""
        t0 = time.perf_counter()
        metas, _ = self._paper_meta([arxiv_id])
        if arxiv_id not in metas:
            raise ValueError(f"unknown paper {arxiv_id}")
        yield {"type": "status", "text": "Preparing the paper (first time: download + index)…"}
        self.ensure_paper_indexed(arxiv_id)
        metas, _ = self._paper_meta([arxiv_id])
        meta = metas[arxiv_id]
        final = ""
        for ev in self.qa.answer_single_stream(
            question, arxiv_id, title=meta["title"], history=history, meta=meta,
        ):
            if ev["type"] == "final":
                final = ev["answer"]
            yield ev
        self._log(session_id, "qa_single", question, final, [arxiv_id],
                  (time.perf_counter() - t0) * 1000)

    def ask_multi(
        self, question: str, paper_ids: list[str] | None = None,
        history: list[dict] | None = None, session_id: str = "",
    ) -> QAResult:
        return QAEngine._collect(self.ask_multi_stream(question, paper_ids, history, session_id))

    def ask_multi_stream(
        self, question: str, paper_ids: list[str] | None = None,
        history: list[dict] | None = None, session_id: str = "",
    ):
        """Chat across papers. Yields events: status / sources / token / final."""
        t0 = time.perf_counter()
        if not paper_ids:
            # retrieval decides which papers matter for this question;
            # only those selected papers get downloaded/indexed (below)
            hit = self.search(question, fetch_arxiv=False, smart=False)
            paper_ids = [r["arxiv_id"] for r in hit["results"][: settings.lazy_load_top_n]]
            yield {"type": "status", "text": f"Picked {len(paper_ids)} relevant papers for you…"}
        yield {"type": "status", "text": "Preparing the papers (first time: download + index)…"}
        for pid in paper_ids:
            try:
                self.ensure_paper_indexed(pid)
            except Exception as exc:
                logger.warning("skip %s: %s", pid, exc)
        metas, status = self._paper_meta(paper_ids)
        titles = {pid: m["title"] for pid, m in metas.items()}
        final, papers = "", paper_ids
        for ev in self.qa.answer_multi_stream(
            question, paper_ids, titles=titles, history=history,
            paper_status=status, metas=metas,
        ):
            if ev["type"] == "final":
                final, papers = ev["answer"], ev.get("papers", paper_ids)
            yield ev
        self._log(session_id, "qa_multi", question, final, papers,
                  (time.perf_counter() - t0) * 1000)

    # ------------------------------------------------- research landscape
    def knowledge_graph(
        self,
        paper_ids: list[str] | None = None,
        query: str | None = None,
        top_n: int = 10,
        use_llm: bool = False,
        fetch_arxiv: bool = False,
    ) -> dict:
        """Knowledge graph over explicit papers, or over a query's results."""
        notice = None
        if not paper_ids:
            if not query:
                return {"nodes": [], "edges": [], "insights": [], "extractions": {},
                        "notice": "Give paper ids or a query."}
            hit = self.search(query, top_k=top_n, fetch_arxiv=fetch_arxiv)
            paper_ids = [r["arxiv_id"] for r in hit["results"]]
            notice = hit.get("notice")
        papers = load_papers(paper_ids)
        graph = build_graph(papers, use_llm=use_llm)
        graph["notice"] = notice
        return graph

    def citation_map(
        self, paper_ids: list[str] | None = None, query: str | None = None,
        top_n: int = 12,
    ) -> dict:
        """Timeline research map: papers placed by year and citations, with
        who-cites-whom edges and shared foundations."""
        from backend.app.services.citations import citation_map

        notice = None
        if not paper_ids:
            if not query:
                return {"nodes": [], "edges": [], "notice": "Give paper ids or a query."}
            hit = self.search(query, top_k=top_n)
            paper_ids = [r["arxiv_id"] for r in hit["results"]]
            notice = hit.get("notice")
        out = citation_map(paper_ids[:50])
        out["notice"] = notice
        return out

    def paper_concept_map(self, arxiv_id: str, explain: bool = False) -> dict:
        """Concept map of ONE paper: the topics needed to understand it,
        branched by importance. Indexes the full text first if needed."""
        from backend.app.services.paper_concepts import ConceptMapper

        metas, _ = self._paper_meta([arxiv_id])
        if arxiv_id not in metas:
            raise ValueError(f"unknown paper {arxiv_id}")
        try:
            self.ensure_paper_indexed(arxiv_id)
        except Exception as exc:          # abstract-only map is still useful
            logger.warning("concept map: could not index %s: %s", arxiv_id, exc)
        return ConceptMapper(self).build(arxiv_id, explain=explain)

    def read_next(
        self, read_ids: list[str], top_k: int | None = None,
        fetch_arxiv: bool = True, use_llm: bool = False,
    ) -> dict:
        return self.reading.recommend(read_ids, top_k=top_k, fetch_arxiv=fetch_arxiv, use_llm=use_llm)

    # -------------------------------------------------------------- admin
    def stats(self) -> dict:
        with db_session() as s:
            n_papers = s.execute(select(func.count(Paper.id))).scalar() or 0
            n_chunks = s.execute(select(func.count(Chunk.id))).scalar() or 0
            n_queries = s.execute(select(func.count(QueryLog.id))).scalar() or 0
        from backend.app.services.llm.generator import get_llm

        llm = get_llm()
        llm_model = (
            "n/a (grounded sentence synthesis)"
            if llm.name == "extractive"
            else (settings.llm_api_model or settings.llm_model)
            if llm.name in ("openai-compatible", "ollama")
            else settings.llm_model
        )
        return {
            "papers": n_papers,
            "chunks": n_chunks,
            "queries": n_queries,
            "bm25_docs": self.bm25.size,
            "paper_vectors": self.paper_store.ntotal,
            "chunk_vectors": self.chunk_store.ntotal,
            "embedding_backend": type(self.embedder).__name__,
            "embedding_dim": self.embedder.dim,
            "llm_backend": llm.name
            + (" (auto-selected)" if settings.llm_backend == "auto" else ""),
            "llm_model": llm_model,
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
