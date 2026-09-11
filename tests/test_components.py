"""Component-level tests for every pipeline stage."""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------- query processing
def test_query_normalization_and_expansion():
    from backend.app.services.query_processor import process_query

    pq = process_query("  What are   RAG systems for  LLM  QA?  ")
    assert "  " not in pq.normalized
    assert pq.is_question
    joined = " ".join(pq.expansions).lower()
    assert "retrieval augmented generation" in joined
    assert "large language model" in joined
    assert "question answering" in joined


def test_technicality_drives_dynamic_alpha():
    from backend.app.services.query_processor import process_query
    from backend.app.services.retrieval.fusion import dynamic_alpha

    technical = process_query('"BM25" IVF-PQ FAISS')
    conceptual = process_query(
        "how do researchers make language models stop inventing facts when answering questions"
    )
    a_tech = dynamic_alpha(technical)
    a_conc = dynamic_alpha(conceptual)
    assert a_tech < a_conc            # technical query leans sparse
    assert 0.1 <= a_tech <= 0.9 and 0.1 <= a_conc <= 0.9


# ------------------------------------------------------------------ fusion
def test_rrf_prefers_items_ranked_high_in_both_lists():
    from backend.app.services.retrieval.fusion import rrf_fuse

    sparse = [("A", 9.0), ("B", 5.0), ("C", 1.0)]
    dense = [("B", 0.9), ("A", 0.8), ("D", 0.2)]
    fused = rrf_fuse([sparse, dense])
    top2 = {pid for pid, _ in fused[:2]}
    assert top2 == {"A", "B"}


def test_weighted_fusion_alpha_extremes():
    from backend.app.services.retrieval.fusion import weighted_fuse

    sparse = [("S", 10.0), ("X", 1.0)]
    dense = [("D", 0.99), ("X", 0.1)]
    assert weighted_fuse(sparse, dense, alpha=0.0)[0][0] == "S"
    assert weighted_fuse(sparse, dense, alpha=1.0)[0][0] == "D"


# --------------------------------------------------------------- retrieval
def test_bm25_retrieves_lexical_match(pipeline):
    from backend.app.services.query_processor import process_query

    hits = pipeline.bm25.retrieve("dynamic alpha tuning hybrid retrieval", top_k=3)
    assert hits and hits[0][0] == "2503.23013"

    pq = process_query("benchmark for reranking models on scientific text")
    fused = pipeline.hybrid.retrieve(pq, top_k=5)
    assert "2508.08742" in [pid for pid, _ in fused][:3]


def test_dense_retrieval_returns_results(pipeline):
    hits = pipeline.dense.retrieve("multi document question answering", top_k=5)
    assert len(hits) > 0
    assert all(isinstance(s, float) for _, s in hits)


def test_dedup_no_duplicate_indexing(pipeline):
    from backend.app.services.arxiv_client import PaperMeta

    before_bm25 = pipeline.bm25.size
    before_vec = pipeline.paper_store.ntotal
    added = pipeline.ingest_metadata(
        [PaperMeta(arxiv_id="2503.23013", title="DAT duplicate", abstract="dup")]
    )
    assert added == 0
    assert pipeline.bm25.size == before_bm25
    assert pipeline.paper_store.ntotal == before_vec


# --------------------------------------------------------------- reranking
def test_reranker_orders_by_relevance(pipeline):
    cands = [
        {"arxiv_id": "x", "text": "cooking recipes for pasta and pizza", "fused_score": 0.9},
        {"arxiv_id": "y", "text": "cross encoder reranking improves scientific retrieval",
         "fused_score": 0.1},
    ]
    ranked = pipeline.reranker.rerank(
        "reranking models for scientific retrieval", cands, top_k=2
    )
    assert ranked[0]["arxiv_id"] == "y"
    assert 0.0 <= ranked[0]["rerank_score"] <= 1.0


# ------------------------------------------------------ PDF parse + chunk
def test_pdf_parse_and_structure_aware_chunking(sample_pdf):
    from backend.app.services.documents.chunker import chunk_document
    from backend.app.services.documents.loader import parse_pdf

    doc = parse_pdf(sample_pdf, "test.0001")
    assert doc is not None and doc.blocks
    headings = [h for h, _ in doc.blocks if h]
    assert any("Methodology" in h for h in headings)

    chunks = chunk_document(doc)
    sections = {c.section for c in chunks}
    assert "methodology" in sections
    assert "results" in sections
    assert "references" not in sections            # noise sections dropped
    assert all(len(c.text) > 0 for c in chunks)


def test_chunk_overlap_and_size(sample_pdf):
    from backend.app.core.config import settings
    from backend.app.services.documents.chunker import chunk_document
    from backend.app.services.documents.loader import parse_pdf

    doc = parse_pdf(sample_pdf, "test.0002")
    chunks = chunk_document(doc, chunk_size=300, overlap=80)
    body = [c for c in chunks if c.section == "methodology"]
    if len(body) >= 2:
        # consecutive chunks in a section share overlapping sentences
        assert body[0].text[-40:] in body[0].text  # sanity
        tail_words = body[0].text.split()[-5:]
        assert any(w in body[1].text for w in tail_words)
    assert all(len(c.text) <= 300 * 3 for c in chunks)
    assert settings.min_chunk_chars > 0


def test_failed_pdf_falls_back_to_abstract(pipeline):
    """Reliability NFR: bad/missing PDFs must not crash; abstract fallback."""
    from sqlalchemy import select

    from backend.app.db.models import Paper
    from backend.app.db.session import db_session
    from backend.app.services.documents.loader import load_document

    with db_session() as s:
        p = s.execute(select(Paper).where(Paper.arxiv_id == "2411.16019")).scalar_one()
        p.pdf_url = "http://127.0.0.1:9/definitely-unreachable.pdf"
        doc = load_document(p)
        assert doc.source == "abstract"
        assert doc.blocks and doc.blocks[0][0] == "Abstract"
        assert p.pdf_status == "failed"


# ------------------------------------------------------------ vector store
def test_faiss_metadata_filtering(isolated_env):
    from backend.app.services.vector_store import FaissStore

    store = FaissStore("t_filter", dim=8, directory=isolated_env / "vs")
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(10, 8)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    metas = [{"arxiv_id": f"p{i % 2}", "i": i} for i in range(10)]
    store.add(vecs, metas)

    hits = store.search(vecs[0], top_k=5, filter_fn=lambda m: m["arxiv_id"] == "p1")
    assert hits and all(m["arxiv_id"] == "p1" for _, _, m in hits)


def test_faiss_persistence_roundtrip(isolated_env):
    from backend.app.services.vector_store import FaissStore

    d = isolated_env / "vs2"
    store = FaissStore("t_persist", dim=4, directory=d)
    v = np.eye(4, dtype="float32")
    store.add(v, [{"k": i} for i in range(4)])
    store.save()

    reloaded = FaissStore("t_persist", dim=4, directory=d)
    assert reloaded.ntotal == 4
    hits = reloaded.search(v[2], top_k=1)
    assert hits[0][2]["k"] == 2
