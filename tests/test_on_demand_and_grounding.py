"""Tests for on-demand (QA-time) indexing and the upgraded grounded LLM layer.

Contract under test:
* search returns metadata fast and NEVER downloads or indexes full text;
* PDFs are fetched/chunked/embedded only when a paper is selected for QA;
* answers are structured, cited, and cannot carry unsupported citations.
"""
from __future__ import annotations

from sqlalchemy import select

from backend.app.db.models import Paper
from backend.app.db.session import db_session
from backend.app.services.llm.generator import (
    ExtractiveLLM,
    enforce_grounding,
    safe_generate,
)


def _statuses(ids):
    with db_session() as s:
        rows = s.execute(select(Paper).where(Paper.arxiv_id.in_(ids))).scalars().all()
        return {p.arxiv_id: (p.pdf_status, p.n_chunks) for p in rows}


# --------------------------------------------------- search stays metadata-only
def test_search_never_downloads_or_indexes(pipeline):
    out = pipeline.search("semantic search with transformer embeddings",
                          fetch_arxiv=False)
    assert out["results"]
    ids = [r["arxiv_id"] for r in out["results"]]
    before = _statuses(ids)
    # run several more searches — indexing state must not move at all
    for q in ("chunking strategies for RAG", "reranking scientific papers"):
        pipeline.search(q, fetch_arxiv=False)
    after = _statuses(ids)
    assert before == after, "search must not change any paper's indexing state"


def test_search_response_reports_pdf_status(pipeline):
    out = pipeline.search("hierarchical multi document QA", fetch_arxiv=False)
    assert all("pdf_status" in r for r in out["results"])


# ------------------------------------------------------ QA-time on-demand index
def test_qa_selection_triggers_indexing(pipeline):
    pid = "2508.08742"  # untouched by other tests
    assert _statuses([pid])[pid] == ("none", 0), "precondition: not yet indexed"
    res = pipeline.ask_single("What retrieval methods does this paper study?", pid)
    status, n_chunks = _statuses([pid])[pid]
    assert status == "embedded" and n_chunks >= 1, \
        "selecting a paper for QA must download+index it"
    assert res.answer


def test_index_endpoint_reports_status(pipeline):
    from fastapi.testclient import TestClient

    import backend.app.services.pipeline as pl
    from backend.app.main import app

    pl._pipeline = pipeline
    client = TestClient(app)
    # 2411.16019 was left in pdf_status="failed" by the abstract-fallback
    # test — indexing on selection must recover it via the abstract path.
    r = client.post("/api/v1/papers/2411.16019/index")
    assert r.status_code == 200
    body = r.json()
    assert body["n_chunks"] >= 1 and body["pdf_status"] == "embedded"


# ------------------------------------------------------------ grounded answers
def test_extractive_answer_is_structured_and_cited(pipeline):
    pipeline.ensure_paper_indexed("2503.23013")
    res = pipeline.ask_single(
        "How does dynamic alpha tuning combine sparse and dense retrieval?",
        "2503.23013",
    )
    assert "[" in res.answer and "]" in res.answer, "inline citations required"
    cited = {int(n) for n in __import__("re").findall(r"\[(\d+)\]", res.answer)}
    valid = {s.n for s in res.sources}
    assert cited and cited <= valid, "every citation must point at a real source"


def test_extractive_map_mode_no_markers_and_sentinel():
    llm = ExtractiveLLM()
    map_prompt = (
        'Context excerpts from the paper "X" (arXiv:0000.0000):\n\n'
        "[1] section: results\n"
        "Hybrid retrieval improves recall by combining BM25 with dense vectors.\n\n"
        "Question: how is retrieval improved?\n\n"
        "List every fact... reply exactly: NO RELEVANT INFORMATION."
    )
    out = llm.generate(map_prompt)
    assert "[1]" not in out, "map summaries must not carry chunk markers"

    irrelevant = map_prompt.replace(
        "Hybrid retrieval improves recall by combining BM25 with dense vectors.",
        "The weather in Paris is often rainy during autumn months there.",
    ).replace("Question: how is retrieval improved?",
              "Question: what embedding dimensionality is used?")
    assert "NO RELEVANT INFORMATION" in llm.generate(irrelevant)


def test_enforce_grounding_strips_invalid_citations():
    ans, grounded = enforce_grounding(
        "Based on the context, hybrid search helps [2]. It also scales [7].", 3
    )
    assert grounded
    assert "[2]" in ans and "[7]" not in ans
    assert not ans.lower().startswith("based on the context")

    ans2, grounded2 = enforce_grounding("Hybrid search definitely helps a lot.", 3)
    assert not grounded2, "claims with zero valid citations are ungrounded"

    ans3, grounded3 = enforce_grounding(
        "The provided context does not contain this information.", 3
    )
    assert grounded3, "honest no-info answers pass the guard"


def test_safe_generate_grounding_guard_regenerates(monkeypatch):
    """A generative answer with no valid citations must be replaced by a
    grounded extractive answer, never shown to the user."""
    import backend.app.services.llm.generator as gen

    class FakeLLM(gen.BaseLLM):
        name = "fake"
        generative = True

        def generate(self, prompt, system=""):
            return "Everything works great and scales infinitely."  # no citations

    monkeypatch.setattr(gen, "get_llm", lambda: FakeLLM())
    prompt = (
        "[1] section: results\n"
        "Hybrid retrieval improves recall by combining BM25 with dense vectors.\n\n"
        "Question: how is retrieval improved?\n"
    )
    out = gen.safe_generate(prompt, n_sources=1, require_citations=True)
    assert "scales infinitely" not in out
    assert "[1]" in out, "fallback answer is extractive and cited"


def test_safe_generate_rewrites_verbatim_context_copy(monkeypatch):
    import backend.app.services.llm.generator as gen

    class CopyingLLM(gen.BaseLLM):
        name = "copying"
        generative = True
        calls = 0

        def generate(self, prompt, system=""):
            self.calls += 1
            if self.calls == 1:
                return (
                    "Hybrid retrieval combines BM25 sparse retrieval with dense "
                    "embedding retrieval for better search results. [1]"
                )
            return "The method combines two complementary retrieval signals. [1]"

    llm = CopyingLLM()
    monkeypatch.setattr(gen, "get_llm", lambda: llm)
    prompt = (
        "[1] section: results\n"
        "Hybrid retrieval combines BM25 sparse retrieval with dense embedding "
        "retrieval for better search results.\n\n"
        "Question: how does the method work?\n"
    )
    out = safe_generate(prompt, n_sources=1, require_citations=True)
    assert out == "The method combines two complementary retrieval signals. [1]"
    assert llm.calls == 2


def test_safe_generate_rewrites_stitched_repeated_fragments(monkeypatch):
    import backend.app.services.llm.generator as gen

    class StitchingLLM(gen.BaseLLM):
        name = "stitching"
        generative = True
        calls = 0

        def generate(self, prompt, system=""):
            self.calls += 1
            if self.calls == 1:
                return (
                    "The method was tested on IBM's quantum computer and the "
                    "results were evaluated. The method was tested on IBM's "
                    "quantum computer and the results were evaluated. [1]"
                )
            return "The experiments evaluated the proposed method on quantum hardware. [1]"

    llm = StitchingLLM()
    monkeypatch.setattr(gen, "get_llm", lambda: llm)
    prompt = (
        "[1] section: conclusion\n"
        "The method was tested on IBM's quantum computer and the results were evaluated.\n\n"
        "Question: what were the final results?\n"
    )
    out = safe_generate(prompt, n_sources=1, require_citations=True)
    assert out == "The experiments evaluated the proposed method on quantum hardware. [1]"
    assert llm.calls == 2
