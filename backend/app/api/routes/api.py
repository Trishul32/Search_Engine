"""REST API routes (functional requirement: 'provide APIs for retrieval and
query operations')."""
from __future__ import annotations

import dataclasses
import json
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import or_, select

from backend.app.api import schemas
from backend.app.db.models import Paper, QueryLog
from backend.app.db.session import db_session
from backend.app.services.pipeline import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------ search
@router.post("/search", response_model=schemas.SearchResponse)
def search(req: schemas.SearchRequest):
    pipe = get_pipeline()
    return pipe.search(
        req.query,
        top_k=req.top_k,
        fetch_arxiv=req.fetch_arxiv,
        session_id=req.session_id,
        exclude=req.exclude,
        field=req.field,
        smart=req.smart,
    )


# ------------------------------------------------------------------ papers
@router.get("/papers")
def list_papers(
    q: str = Query(default="", max_length=200),
    limit: int = Query(default=1000, ge=1, le=5000),
):
    """Papers in the corpus (for pick-by-title selectors).

    Papers whose full text is already indexed come first (instant QA),
    then the newest. ``q`` filters by title/author words.
    """
    with db_session() as s:
        stmt = select(Paper.arxiv_id, Paper.title, Paper.authors, Paper.published, Paper.pdf_status)
        for word in q.split()[:6]:
            like = f"%{word}%"
            stmt = stmt.where(or_(Paper.title.ilike(like), Paper.authors.ilike(like)))
        rows = s.execute(stmt).all()
    rows.sort(key=lambda r: (r.pdf_status != "embedded", -int((r.published or "0000")[:4] or 0), r.title))
    return [
        {"arxiv_id": r.arxiv_id, "title": r.title, "authors": r.authors or "",
         "published": r.published or "", "pdf_status": r.pdf_status}
        for r in rows[:limit]
    ]


@router.get("/papers/{arxiv_id}", response_model=schemas.PaperOut)
def get_paper(arxiv_id: str):
    with db_session() as s:
        p = s.execute(select(Paper).where(Paper.arxiv_id == arxiv_id)).scalar_one_or_none()
        if not p:
            raise HTTPException(404, f"paper {arxiv_id} not indexed")
        return p.to_dict()


@router.post("/papers/{arxiv_id}/index")
def index_paper(arxiv_id: str):
    """On-demand full-text indexing: download -> parse -> chunk -> embed.

    Called when the user selects a paper for QA (idempotent — an already
    embedded paper returns immediately). Search never triggers this.
    """
    pipe = get_pipeline()
    try:
        n = pipe.ensure_paper_indexed(arxiv_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    with db_session() as s:
        p = s.execute(select(Paper).where(Paper.arxiv_id == arxiv_id)).scalar_one_or_none()
        status = p.pdf_status if p else "unknown"
    return {"arxiv_id": arxiv_id, "n_chunks": n, "pdf_status": status}


@router.post("/papers/{arxiv_id}/concept-map")
def paper_concept_map(arxiv_id: str, explain: bool = False):
    """Concept map of one paper: major topics and subtopics ranked by their
    importance to the paper, each with its role, evidence and locations."""
    try:
        return get_pipeline().paper_concept_map(arxiv_id, explain=explain)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@router.get("/papers/{arxiv_id}/recommendations")
def recommendations(arxiv_id: str, top_k: int = Query(default=5, ge=1, le=20)):
    pipe = get_pipeline()
    recs = pipe.recommender.similar_papers(arxiv_id, top_k=top_k)
    return {"arxiv_id": arxiv_id, "recommendations": recs}


# ------------------------------------------------------ research landscape
@router.post("/graph")
def knowledge_graph(req: schemas.GraphRequest):
    """Paper -> knowledge graph (methods, datasets, metrics, tasks and how
    papers introduce / use / extend / evaluate them)."""
    return get_pipeline().knowledge_graph(
        paper_ids=req.paper_ids, query=req.query, top_n=req.top_n, use_llm=req.use_llm,
    )


@router.post("/graph/citations")
def citation_graph(req: schemas.GraphRequest):
    """Research map: papers by year x citations, citation edges, and the
    shared foundations (papers several of them cite)."""
    return get_pipeline().citation_map(paper_ids=req.paper_ids, query=req.query, top_n=req.top_n)


@router.post("/read-next")
def read_next(req: schemas.ReadNextRequest):
    """Given papers the user has read, recommend the missing connections."""
    return get_pipeline().read_next(
        req.read_ids, top_k=req.top_k, fetch_arxiv=req.fetch_arxiv, use_llm=req.use_llm,
    )


# ---------------------------------------------------------------------- QA
def _ndjson(events):
    """Stream pipeline events as newline-delimited JSON; errors become an
    'error' event instead of a broken connection."""
    try:
        for ev in events:
            yield json.dumps(ev, ensure_ascii=False) + "\n"
    except ValueError as exc:
        yield json.dumps({"type": "error", "text": str(exc)}) + "\n"
    except Exception as exc:  # never leave the chat hanging
        logger.exception("QA stream failed")
        yield json.dumps({"type": "error", "text": f"Answer failed: {type(exc).__name__}: {exc}"}) + "\n"


@router.post("/qa/single/stream")
def qa_single_stream(req: schemas.SingleQARequest):
    """Chat with one paper; streams status / sources / token / final events."""
    pipe = get_pipeline()
    return StreamingResponse(
        _ndjson(pipe.ask_single_stream(
            req.question, req.arxiv_id,
            history=[t.model_dump() for t in req.history], session_id=req.session_id,
        )),
        media_type="application/x-ndjson",
    )


@router.post("/qa/multi/stream")
def qa_multi_stream(req: schemas.MultiQARequest):
    """Chat across papers; streams status / sources / token / final events."""
    pipe = get_pipeline()
    return StreamingResponse(
        _ndjson(pipe.ask_multi_stream(
            req.question, paper_ids=req.paper_ids,
            history=[t.model_dump() for t in req.history], session_id=req.session_id,
        )),
        media_type="application/x-ndjson",
    )


@router.post("/qa/single", response_model=schemas.QAResponse)
def qa_single(req: schemas.SingleQARequest):
    pipe = get_pipeline()
    try:
        res = pipe.ask_single(
            req.question,
            req.arxiv_id,
            history=[t.model_dump() for t in req.history],
            session_id=req.session_id,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return _qa_response(res)


@router.post("/qa/multi", response_model=schemas.QAResponse)
def qa_multi(req: schemas.MultiQARequest):
    pipe = get_pipeline()
    res = pipe.ask_multi(
        req.question,
        paper_ids=req.paper_ids,
        history=[t.model_dump() for t in req.history],
        session_id=req.session_id,
    )
    return _qa_response(res)


def _qa_response(res) -> dict:
    return {
        "answer": res.answer,
        "mode": res.mode,
        "papers": res.papers,
        "sources": [dataclasses.asdict(s) for s in res.sources],
    }


# ------------------------------------------------------------------- admin
@router.get("/admin/stats")
def stats():
    return get_pipeline().stats()


@router.post("/admin/rebuild")
def rebuild():
    return get_pipeline().rebuild_indexes()


@router.get("/admin/history")
def history(limit: int = Query(default=50, ge=1, le=500), session_id: str = ""):
    with db_session() as s:
        q = select(QueryLog).order_by(QueryLog.id.desc()).limit(limit)
        if session_id:
            q = select(QueryLog).where(QueryLog.session_id == session_id).order_by(
                QueryLog.id.desc()
            ).limit(limit)
        rows = s.execute(q).scalars().all()
        return [
            {
                "id": r.id,
                "mode": r.mode,
                "query": r.query,
                "latency_ms": r.latency_ms,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


@router.get("/health")
def health():
    return {"status": "ok"}
