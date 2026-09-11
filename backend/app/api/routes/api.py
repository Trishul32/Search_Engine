"""REST API routes (functional requirement: 'provide APIs for retrieval and
query operations')."""
from __future__ import annotations

import dataclasses
import logging

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

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
        lazy_load=req.lazy_load,
        session_id=req.session_id,
    )


# ------------------------------------------------------------------ papers
@router.get("/papers/{arxiv_id}", response_model=schemas.PaperOut)
def get_paper(arxiv_id: str):
    with db_session() as s:
        p = s.execute(select(Paper).where(Paper.arxiv_id == arxiv_id)).scalar_one_or_none()
        if not p:
            raise HTTPException(404, f"paper {arxiv_id} not indexed")
        return p.to_dict()


@router.post("/papers/{arxiv_id}/index")
def index_paper(arxiv_id: str):
    """Force the lazy load+chunk+embed pipeline for one paper."""
    pipe = get_pipeline()
    try:
        n = pipe.ensure_paper_indexed(arxiv_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return {"arxiv_id": arxiv_id, "chunks": n}


@router.get("/papers/{arxiv_id}/recommendations")
def recommendations(arxiv_id: str, top_k: int = Query(default=5, ge=1, le=20)):
    pipe = get_pipeline()
    recs = pipe.recommender.similar_papers(arxiv_id, top_k=top_k)
    return {"arxiv_id": arxiv_id, "recommendations": recs}


# ---------------------------------------------------------------------- QA
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
