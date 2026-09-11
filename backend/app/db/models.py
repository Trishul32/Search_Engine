"""Relational metadata store (PostgreSQL in production, SQLite for dev).

Vector data lives in FAISS; this DB is the source of truth for paper
metadata, chunk text/provenance, and query history. The unique constraint on
``papers.arxiv_id`` enforces the "no duplicate indexing" reliability
requirement at the storage layer.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    arxiv_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str] = mapped_column(Text)
    authors: Mapped[str] = mapped_column(Text, default="")           # "; " joined
    categories: Mapped[str] = mapped_column(String(256), default="")  # "," joined
    published: Mapped[str] = mapped_column(String(32), default="")
    pdf_url: Mapped[str] = mapped_column(Text, default="")
    # lazy-loading state machine: none -> downloaded -> parsed -> embedded
    pdf_status: Mapped[str] = mapped_column(String(16), default="none")
    pdf_path: Mapped[str] = mapped_column(Text, default="")
    parse_error: Mapped[str] = mapped_column(Text, default="")
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "published": self.published,
            "pdf_url": self.pdf_url,
            "pdf_status": self.pdf_status,
            "n_chunks": self.n_chunks,
        }


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("paper_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String(128), default="body")
    text: Mapped[str] = mapped_column(Text)
    # position of this chunk's vector inside the FAISS chunk index
    faiss_id: Mapped[int] = mapped_column(Integer, index=True, default=-1)
    embedded: Mapped[bool] = mapped_column(Boolean, default=False)

    paper: Mapped[Paper] = relationship(back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True, default="")
    mode: Mapped[str] = mapped_column(String(32), default="search")
    query: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="")
    sources: Mapped[str] = mapped_column(Text, default="")     # JSON list
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
