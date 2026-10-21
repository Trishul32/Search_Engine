"""Pydantic request/response schemas for the public API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    fetch_arxiv: bool = True
    session_id: str = ""
    # explicit filters (in addition to what is understood from the query)
    exclude: list[str] = Field(default_factory=list, max_length=20)
    field: str | None = Field(default=None, max_length=64)
    # LLM rewrite of conversational queries into academic vocabulary
    smart: bool = True


class PaperOut(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: str = ""
    categories: str = ""
    published: str = ""
    pdf_status: str = "none"
    n_chunks: int = 0
    score: float | None = None
    evidence: str = ""          # best-matching abstract sentence
    live_match: bool = False    # matched the structured live arXiv query


class SearchResponse(BaseModel):
    query: str
    timings_ms: dict[str, float] | None = None
    notice: str | None = None
    interpretation: dict | None = None
    excluded_count: int = 0
    live_query: list[str] = []
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


class GraphRequest(BaseModel):
    paper_ids: list[str] | None = Field(default=None, max_length=50)
    query: str | None = Field(default=None, max_length=1000)
    top_n: int = Field(default=10, ge=2, le=50)
    use_llm: bool = False


class ReadNextRequest(BaseModel):
    read_ids: list[str] = Field(min_length=1, max_length=20)
    top_k: int = Field(default=5, ge=1, le=20)
    fetch_arxiv: bool = True
    use_llm: bool = False
