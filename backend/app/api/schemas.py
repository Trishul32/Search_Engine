"""Pydantic request/response schemas for the public API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    fetch_arxiv: bool = True
    lazy_load: bool = True
    session_id: str = ""


class PaperOut(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: str = ""
    categories: str = ""
    published: str = ""
    pdf_status: str = "none"
    score: float | None = None


class SearchResponse(BaseModel):
    query: str
    expansions: list[str]
    alpha_hint: float
    latency_ms: float
    results: list[PaperOut]


class ChatTurn(BaseModel):
    role: str
    content: str


class SingleQARequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    arxiv_id: str
    history: list[ChatTurn] = []
    session_id: str = ""


class MultiQARequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    paper_ids: list[str] | None = None
    history: list[ChatTurn] = []
    session_id: str = ""


class SourceOut(BaseModel):
    n: int
    arxiv_id: str
    title: str
    section: str
    snippet: str


class QAResponse(BaseModel):
    answer: str
    mode: str
    papers: list[str]
    sources: list[SourceOut]
