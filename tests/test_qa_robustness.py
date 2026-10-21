"""Regression tests for the reported multi-QA failure and slow search.

Reproduces: user selects papers, asks "compare the results", and gets
"None of the selected papers contain indexed content relevant to this
question" — caused by literal word matching (results != result) in the
extractive map stage, and undiagnosable when indexing itself had failed.
"""
from __future__ import annotations

from backend.app.db.models import Paper
from backend.app.db.session import db_session
from backend.app.services.llm.generator import ExtractiveLLM, _words
from backend.app.services.qa.engine import QAEngine


def test_stemming_matches_morphological_variants():
    assert _words("compare the results") & _words("The result shows improvement")
    assert _words("compared accuracy") & _words("we compare accuracies")
    assert _words("embeddings") == _words("embedding")


def test_vague_compare_query_yields_evidence_not_sentinel():
    """The user's exact failure: 'compare the results' vs a results section."""
    llm = ExtractiveLLM()
    prompt = (
        'Context excerpts from the paper "X" (arXiv:0000.0000):\n\n'
        "[1] section: results\n"
        "The result shows hybrid retrieval outperforms BM25 by 12 points "
        "on recall. Dense retrieval alone performed worse on keyword-heavy "
        "queries in our experiments.\n\n"
        "Question: compare the results\n\n"
        "List every fact... reply exactly: NO RELEVANT INFORMATION."
    )
    out = llm.generate(prompt)
    assert "NO RELEVANT INFORMATION" not in out
    assert "hybrid" in out.lower()


def test_no_info_sentinel_detection_is_precise():
    assert QAEngine._is_no_info("NO RELEVANT INFORMATION.")
    assert QAEngine._is_no_info("  no relevant information ")
    # a partial answer mentioning the phrase must NOT be discarded
    assert not QAEngine._is_no_info(
        "There is NO RELEVANT INFORMATION about datasets here; however, the "
        "paper reports a 12-point recall gain for hybrid retrieval over BM25, "
        "and describes the reranking model used in the experiments."
    )


def test_multi_qa_diagnostics_name_the_failing_paper(pipeline, monkeypatch):
    # a paper whose indexing genuinely fails (no extractable content at all)
    with db_session() as s:
        if not s.query(Paper).filter_by(arxiv_id="9999.00001").first():
            s.add(Paper(
                arxiv_id="9999.00001",
                title="Broken Scanned Paper",
                abstract="",
                pdf_url="http://127.0.0.1:9/nope.pdf",
            ))
    import backend.app.services.pipeline as pl
    monkeypatch.setattr(pl, "chunk_document", lambda doc: [])
    res = pipeline.ask_multi(
        "compare the quantum teleportation throughput numbers",
        paper_ids=["9999.00001"],
    )
    assert "9999.00001" in res.answer
    assert "no indexed full text" in res.answer
    assert "failed" in res.answer


def test_multi_qa_diagnostics_suggest_rephrasing(pipeline):
    pipeline.ensure_paper_indexed("2503.23013")
    # indexed paper, but a question about content that cannot match
    res = pipeline.ask_multi(
        "compare xylophone manufacturing techniques",
        paper_ids=["2503.23013"],
    )
    # The engine now always reads the paper (abstract + best passages), so
    # an off-topic question gets an honest "not covered" answer plus a tip,
    # instead of an empty diagnostics message.
    assert "does not contain" in res.answer
    assert "Tip:" in res.answer


def test_search_reports_timing_breakdown(pipeline):
    out = pipeline.search("hybrid retrieval", fetch_arxiv=False)
    t = out["timings_ms"]
    assert set(t) == {"arxiv_fetch", "retrieve_rerank"}
    assert t["arxiv_fetch"] == 0.0          # no live fetch requested
    assert t["retrieve_rerank"] >= 0.0
