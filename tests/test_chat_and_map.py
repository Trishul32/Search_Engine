"""Paper chat (intent-aware retrieval, streaming, citation repair) and the
citation timeline map."""
from __future__ import annotations

import json

import pytest

from backend.app.services.qa.intent import contextualize, detect_intent


# ------------------------------------------------------------------ intent
@pytest.mark.parametrize("question, name, generic", [
    ("Whats the key difference in these paper", "compare", True),
    ("summarize this paper", "overview", True),
    ("what about its limitations?", "limitation", True),
    ("what are the main contributions", "contribution", True),
    ("what accuracy did they get on NQ?", "results", False),
    ("what is this paper about", "overview", True),
])
def test_intent_detection(question, name, generic):
    intent = detect_intent(question)
    assert intent.name == name and intent.generic is generic


def test_follow_up_is_contextualised_for_retrieval():
    history = [{"role": "user", "content": "how does the trust framework work"},
               {"role": "assistant", "content": "It distinguishes reliance from trust."}]
    assert contextualize("what about its limitations?", history).startswith("how does the trust framework")
    long_q = "what datasets were used to evaluate the dense retriever in the experiments"
    assert contextualize(long_q, history) == long_q


# ------------------------------------------------------------ QA engine
def test_generic_compare_question_gets_an_answer_not_diagnostics(pipeline):
    """The reported failure: 'key difference' matched no passage -> no answer."""
    ids = ["2503.23013", "2402.01767"]
    for pid in ids:
        pipeline.ensure_paper_indexed(pid)
    res = pipeline.ask_multi("Whats the key difference in these paper", paper_ids=ids)
    assert "couldn't extract evidence" not in res.answer
    assert res.answer.strip()
    assert {s.arxiv_id for s in res.sources} == set(ids)
    assert any(s.section == "abstract" for s in res.sources)


def test_per_paper_retrieval_is_exact_not_post_filtered(pipeline):
    """With many chunks indexed, a paper still gets several passages."""
    from backend.app.services.qa.intent import detect_intent as di

    pid = "2503.23013"
    pipeline.ensure_paper_indexed(pid)
    n_chunks = len(pipeline.qa._paper_chunks(pid))
    chunks, _ = pipeline.qa._select_chunks(pid, "summarize this paper", di("summarize this paper"), 10_000)
    assert chunks and len(chunks) == min(n_chunks, len(chunks))
    assert all(c["arxiv_id"] == pid for c in chunks)
    if n_chunks > 1:
        assert len(chunks) > 1


def test_single_chat_stream_events_and_follow_up(pipeline):
    pid = "2503.23013"
    events = list(pipeline.ask_single_stream("summarize this paper", pid))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "status" and "sources" in kinds and kinds[-1] == "final"
    final = events[-1]["answer"]
    assert final and "[1]" in final or "[" in final
    sources = next(e for e in events if e["type"] == "sources")["sources"]
    assert sources[0]["section"] == "abstract"
    res = pipeline.ask_single("what about its limitations?", pid,
                              history=[{"role": "user", "content": "summarize this paper"},
                                       {"role": "assistant", "content": final}])
    assert res.answer


def test_unknown_paper_stream_raises_value_error(pipeline):
    with pytest.raises(ValueError):
        list(pipeline.ask_single_stream("hi", "0000.00000"))


# --------------------------------------------------- generation + guard
def test_attach_citations_repairs_missing_markers_and_skips_tables():
    from backend.app.services.llm.generator import attach_citations

    contexts = [(1, "Dense passage retrieval uses a dual encoder trained on question passage pairs."),
                (2, "The system improves top twenty retrieval accuracy on Natural Questions.")]
    answer = ("The method trains a dual encoder on question passage pairs.\n"
              "| Paper | Goal |\n| --- | --- |\n| DPR | retrieval |\n"
              "It improves top twenty retrieval accuracy on Natural Questions.")
    fixed, support = attach_citations(answer, contexts)
    assert "pairs. [1]" in fixed or "pairs [1]" in fixed
    assert "Natural Questions. [2]" in fixed or "Natural Questions [2]" in fixed
    assert "| DPR | retrieval |" in fixed            # table row untouched
    assert support == 1.0


class _FakeLLM:
    name = "fake"
    generative = True

    def __init__(self, text):
        self.text = text

    def generate_stream(self, prompt, system="", max_tokens=None):
        for word in self.text.split(" "):
            yield word + " "


def _prompt():
    return ('Paper: "X"\n\n[1] Abstract\nDense passage retrieval uses a dual encoder trained '
            'on question passage pairs.\n\nQuestion: how does it work?\nAnswer:')


def test_stream_generate_keeps_honest_not_covered_answers(monkeypatch):
    import backend.app.services.llm.generator as g

    monkeypatch.setattr(g, "get_llm", lambda: _FakeLLM("The paper does not discuss blockchain at all."))
    events = list(g.stream_generate(_prompt(), n_sources=1, require_citations=True))
    assert events[0][0] == "token"
    assert events[-1] == ("final", "The paper does not discuss blockchain at all.")


def test_stream_generate_replaces_unsupported_answers(monkeypatch):
    import backend.app.services.llm.generator as g

    monkeypatch.setattr(g, "get_llm", lambda: _FakeLLM(
        "Quantum teleportation scales infinitely across galaxies today. Bananas are purple fruit."))
    final = list(g.stream_generate(_prompt(), n_sources=1, require_citations=True))[-1][1]
    assert "galaxies" not in final and "[1]" in final     # grounded extractive fallback


def test_comparison_answers_with_repeated_titles_are_not_flagged():
    from backend.app.services.llm.generator import _is_degenerate_chat

    title = "AI-PACE: A Framework for Integrating AI into Medical Education"
    answer = (f"| Paper | Goal |\n| --- | --- |\n| {title} | education |\n\n"
              f"- {title} focuses on curricula while the other paper studies trust.\n"
              f"- {title} reviews 23 articles whereas the other paper is philosophical.")
    assert not _is_degenerate_chat(answer)
    looping = "\n".join(["The model repeats this exact same long sentence again and again."] * 4)
    assert _is_degenerate_chat(looping)


def test_ollama_requests_a_large_context_window():
    from backend.app.core.config import settings
    from backend.app.services.llm.generator import OllamaLLM

    payload = OllamaLLM()._payload("hello", "sys", True, 50)
    assert payload["options"]["num_ctx"] == settings.llm_num_ctx >= 4096
    assert payload["options"]["num_predict"] == 50 and payload["stream"] is True
    assert payload["messages"][0] == {"role": "system", "content": "sys"}


# ------------------------------------------------------------------- API
def test_streaming_chat_endpoint_and_paper_list(pipeline, monkeypatch):
    import backend.app.api.routes.api as api_mod
    from fastapi.testclient import TestClient

    from backend.app.main import app

    monkeypatch.setattr(api_mod, "get_pipeline", lambda: pipeline)
    client = TestClient(app)
    papers = client.get("/api/v1/papers", params={"q": "alpha tuning"}).json()
    assert papers and papers[0]["title"] and "pdf_status" in papers[0]
    with client.stream("POST", "/api/v1/qa/single/stream",
                       json={"question": "summarize this paper", "arxiv_id": "2503.23013"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert events[-1]["type"] == "final" and events[-1]["answer"]
    with client.stream("POST", "/api/v1/qa/single/stream",
                       json={"question": "hi there", "arxiv_id": "0000.00000"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert events == [{"type": "error", "text": "unknown paper 0000.00000"}]


# --------------------------------------------------------- citation map
def test_citation_map_edges_and_shared_foundations(pipeline, monkeypatch):
    import backend.app.services.citations as cit

    a, b, c = "2503.23013", "2402.01767", "2506.17277"
    fake = {
        a: {"citations": 50, "year": 2025, "first_author": "Hsu", "tldr": "Tunes alpha.",
            "references": [{"arxiv_id": b, "key": b},
                           {"arxiv_id": None, "key": "doi:bert", "title": "BERT", "year": 2018, "citations": 100000}],
            "source": "Semantic Scholar"},
        b: {"citations": 10, "year": 2024, "first_author": "Chen", "tldr": "",
            "references": [{"arxiv_id": None, "key": "doi:bert", "title": "BERT", "year": 2018, "citations": 100000}],
            "source": "Semantic Scholar"},
        c: {"citations": 3, "year": 2025, "first_author": "Lee", "tldr": "", "references": [],
            "source": "Semantic Scholar"},
    }
    monkeypatch.setattr(cit, "_cached", lambda ids: {})
    monkeypatch.setattr(cit, "_store", lambda data: None)
    monkeypatch.setattr(cit, "_from_semantic_scholar", lambda ids: {i: fake[i] for i in ids})
    m = cit.citation_map([a, b, c])
    nodes = {n["id"]: n for n in m["nodes"]}
    assert nodes[a]["label"] == "Hsu 2025" and nodes[a]["citations"] == 50
    assert {"source": a, "target": b, "kind": "cites"} in m["edges"]
    found = [n for n in m["nodes"] if n["kind"] == "foundation"]
    assert [f["title"] for f in found] == ["BERT"]          # cited by 2 papers
    assert {(e["source"], e["target"]) for e in m["edges"] if e["target"] == found[0]["id"]} == \
        {(a, found[0]["id"]), (b, found[0]["id"])}
    assert m["source"] == "Semantic Scholar"


def test_citation_map_works_offline_from_metadata(pipeline, monkeypatch):
    import backend.app.services.citations as cit

    monkeypatch.setattr(cit, "_cached", lambda ids: {})
    m = cit.citation_map(["2503.23013"], online=False)
    assert m["nodes"][0]["citations"] is None and m["source"] == "local metadata only"


def test_timeline_html_places_papers_and_escapes_text():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
    from graph_view import timeline_html

    cmap = {"nodes": [
        {"id": "1", "kind": "paper", "title": "Old </script><b>x</b>", "label": "A 2018", "year": 2018,
         "published": "2018-01-01", "citations": 1000},
        {"id": "2", "kind": "paper", "title": "New", "label": "B 2024", "year": 2024,
         "published": "2024-06-01", "citations": 5},
    ], "edges": [{"source": "2", "target": "1", "kind": "cites"}]}
    page = timeline_html(cmap)
    assert "</script><b>" not in page                    # no HTML injection via titles
    data = json.loads(page.split("const DATA = ", 1)[1].split(";\n", 1)[0].replace("<\\/", "</"))
    by_id = {n["id"]: n for n in data["nodes"]}
    assert by_id["1"]["x"] < by_id["2"]["x"]              # older -> left
    assert by_id["1"]["y"] < by_id["2"]["y"]              # more citations -> higher
