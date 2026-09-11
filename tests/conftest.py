"""Shared fixtures. Tests run fully offline: hash embedder, no reranker
model, extractive LLM, temp DB + indexes, and synthetic PDFs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Configure BEFORE importing app modules.
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("LLM_BACKEND", "extractive")
os.environ.setdefault("RERANKER_ENABLED", "true")
os.environ.setdefault("CORRECTIVE_RETRIEVAL", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(scope="session", autouse=True)
def isolated_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("appdata")
    from backend.app.core.config import settings

    settings.database_url = f"sqlite:///{tmp / 'test.db'}"
    settings.data_dir = tmp
    settings.pdf_dir = tmp / "pdfs"
    settings.index_dir = tmp / "indexes"
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    # rebind the DB engine to the temp database
    import backend.app.db.session as dbs
    from sqlalchemy import create_engine

    dbs.engine = create_engine(
        settings.database_url, connect_args={"check_same_thread": False}
    )
    dbs.SessionLocal.configure(bind=dbs.engine)
    yield tmp


CORPUS = [
    {
        "arxiv_id": "2503.23013",
        "title": "DAT: Dynamic Alpha Tuning for Hybrid Retrieval in RAG",
        "abstract": "We propose dynamic alpha tuning, a method that adjusts the "
        "interpolation weight between BM25 sparse retrieval and dense embedding "
        "retrieval on a per-query basis, improving retrieval augmented generation "
        "across diverse query types.",
        "categories": "cs.IR,cs.CL",
    },
    {
        "arxiv_id": "2402.01767",
        "title": "HiQA: Hierarchical Contextual Augmentation RAG for Multi-Documents QA",
        "abstract": "HiQA performs coarse-to-fine multi-document question answering: "
        "document-level selection, passage retrieval, and multi-hop reasoning chains "
        "for complex questions requiring evidence from multiple sources.",
        "categories": "cs.CL",
    },
    {
        "arxiv_id": "2506.17277",
        "title": "Chunk Twice, Embed Once: Segmentation Trade-offs in Chemistry-Aware RAG",
        "abstract": "A systematic study of chunking strategies showing dual-granularity "
        "structure aware segmentation improves retrieval faithfulness over fixed-size "
        "chunking in scientific retrieval augmented generation pipelines.",
        "categories": "cs.CL,cs.IR",
    },
    {
        "arxiv_id": "2508.08742",
        "title": "SciRerankBench: Benchmarking Rerankers for Scientific RAG",
        "abstract": "A benchmark of cross-encoder and LLM based reranking models over "
        "twelve scientific domains with fifty thousand expert labeled query document "
        "pairs, showing rerankers substantially outperform BM25 ordering.",
        "categories": "cs.IR",
    },
    {
        "arxiv_id": "2411.16019",
        "title": "M3: Mamba-assisted Multi-Circuit Optimization via MBRL",
        "abstract": "Model based reinforcement learning with Mamba networks for "
        "electronic circuit design optimization and pipeline scheduling under "
        "resource constraints.",
        "categories": "cs.LG",
    },
    {
        "arxiv_id": "2508.17694",
        "title": "Semantic Search for Information Retrieval",
        "abstract": "A study of transformer embedding models mapping queries and "
        "documents into a shared vector space, outperforming BM25 for paraphrased "
        "conceptual queries in semantic search systems.",
        "categories": "cs.IR",
    },
]


@pytest.fixture(scope="session")
def pipeline(isolated_env):
    from backend.app.services.arxiv_client import PaperMeta
    from backend.app.services.pipeline import SearchPipeline

    pipe = SearchPipeline()
    metas = [PaperMeta(**{**c, "pdf_url": ""}) for c in CORPUS]
    pipe.ingest_metadata(metas)
    return pipe


@pytest.fixture()
def sample_pdf(tmp_path):
    """Generate a small structured PDF with real section headings."""
    import fitz

    doc = fitz.open()
    sections = [
        ("Abstract", "We study hybrid retrieval combining BM25 and dense embeddings. "
                     "Our reranking stage uses a cross encoder model to refine candidates."),
        ("1 Introduction", "Keyword search misses semantically related papers. " * 8),
        ("2 Methodology", "Our method fuses sparse and dense scores with dynamic alpha. "
                          "Chunks are embedded with sentence transformers and stored in FAISS. " * 6),
        ("3 Results", "Hybrid retrieval improves recall at ten by eighteen percent over BM25. " * 6),
        ("4 Conclusion", "Hybrid retrieval with reranking is effective for scientific search."),
        ("References", "[1] Some citation. [2] Another citation."),
    ]
    page = doc.new_page()
    y = 60
    for heading, body in sections:
        if y > 720:
            page = doc.new_page()
            y = 60
        page.insert_text((60, y), heading, fontsize=16)
        y += 26
        for i in range(0, len(body), 90):
            if y > 760:
                page = doc.new_page()
                y = 60
            page.insert_text((60, y), body[i : i + 90], fontsize=10)
            y += 14
        y += 16
    path = tmp_path / "sample.pdf"
    doc.save(path)
    doc.close()
    return path
