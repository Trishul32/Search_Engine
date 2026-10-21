"""Single-paper concept map: key topics of one paper, branched by importance."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from backend.app.services.paper_concepts import _phrases, _verb_form

PID = "8101.00001"
TITLE = "RAG-Lite: Retrieval-Augmented Generation for Clinical Question Answering"
ABSTRACT = (
    "We introduce RAG-Lite, a retrieval-augmented generation (RAG) system for clinical question "
    "answering over electronic health records. Large language models hallucinate when they answer "
    "clinical questions without evidence. RAG-Lite pairs a dense retriever with a small generator. "
    "The dense retriever encodes electronic health records into a vector index. "
    "Retrieval-augmented generation grounds every answer in retrieved passages. "
    "RAG models reduce hallucination compared with closed-book language models. "
    "Clinical question answering is evaluated on MIMIC-III notes with exact match and F1. "
    "The dense retriever is trained with contrastive learning on clinical question answering pairs. "
    "Electronic health records contain long notes, so the vector index stores short passages. "
    "Hallucination drops sharply when retrieved passages are shown to the generator. "
    "Our results show that RAG-Lite outperforms closed-book language models on MIMIC-III."
)


def test_phrases_stop_at_verbs_and_keep_compound_nouns():
    out = _phrases("Neural networks learn representations of electronic health records.")
    assert "Neural networks" in out
    assert not any("learn" in p for p in out)
    assert "electronic health records" in _phrases("We study electronic health records.")
    assert "machine learning" in _phrases("This is about machine learning methods.")
    assert _verb_form("outperforms") == "s" and _verb_form("learning") == "ing" and _verb_form("design") is None


@pytest.fixture(scope="module")
def concept_map(pipeline):
    from backend.app.services.arxiv_client import PaperMeta

    pipeline.ingest_metadata([PaperMeta(arxiv_id=PID, title=TITLE, abstract=ABSTRACT,
                                        categories="cs.CL", published="2025-02-01")])
    return pipeline.paper_concept_map(PID)


def test_concept_map_shape_and_ranking(concept_map):
    topics = concept_map["topics"]
    assert 2 <= len(topics) <= 6
    assert all(len(t["subtopics"]) <= 5 for t in topics)
    imps = [t["importance"] for t in topics]
    assert imps == sorted(imps, reverse=True)                 # most important first
    assert [t["tier"] for t in topics[:2]] == ["Core", "Core"]
    assert all(s["tier"] == "Supporting" for t in topics for s in t["subtopics"])
    roles = {"background", "concept", "method", "contribution", "evaluation"}
    assert all(t["role"] in roles for t in topics)
    assert {p["id"] for p in concept_map["learning_path"]} == {t["id"] for t in topics}
    assert concept_map["paper"]["arxiv_id"] == PID


def test_acronym_merges_with_hyphenated_long_form(concept_map):
    names = [t["name"] for t in concept_map["topics"]]
    rag = [n for n in names if "retrieval-augmented generation" in n.lower()]
    assert rag and "(RAG)" in rag[0]
    # "RAG models" is a branch of RAG, not a separate major topic
    assert not any(n.lower() == "rag models" for n in names)


def test_key_topics_found_with_evidence(concept_map):
    everything = [t for t in concept_map["topics"]] + [s for t in concept_map["topics"] for s in t["subtopics"]]
    names = " | ".join(t["name"].lower() for t in everything)
    for expected in ("dense retriever", "electronic health records", "hallucination"):
        assert expected in names
    for t in everything:
        if t["evidence"]:
            head = t["name"].split(" (")[0].lower()
            assert head.split()[0][:5] in t["evidence"].lower()


def test_introduced_method_is_marked_as_contribution(concept_map):
    by_name = {t["name"].lower(): t for t in concept_map["topics"]}
    by_name.update({s["name"].lower(): s for t in concept_map["topics"] for s in t["subtopics"]})
    lite = next(v for k, v in by_name.items() if k.startswith("rag-lite"))
    assert lite["role"] == "contribution"


def test_concept_map_api(pipeline, monkeypatch, concept_map):
    import backend.app.api.routes.api as api_mod
    from fastapi.testclient import TestClient

    from backend.app.main import app

    monkeypatch.setattr(api_mod, "get_pipeline", lambda: pipeline)
    client = TestClient(app)
    r = client.post(f"/api/v1/papers/{PID}/concept-map")
    assert r.status_code == 200 and r.json()["topics"]
    assert client.post("/api/v1/papers/0000.00000/concept-map").status_code == 404


def test_rendered_map_has_no_overlapping_labels_and_escapes(concept_map):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
    import json

    from graph_view import _box, concept_map_html

    evil = json.loads(json.dumps(concept_map))
    evil["paper"]["title"] = "</script><img src=x onerror=alert(1)>"
    page = concept_map_html(evil)
    assert "</script><img" not in page
    data = json.loads(page.split("const DATA = ", 1)[1].split(";\n", 1)[0].replace("<\\/", "</"))
    subs = [n for n in data["nodes"] if n["id"].startswith("s:")]
    for i, a in enumerate(subs):
        for b in subs[i + 1:]:
            wa, ha = _box(a)
            wb, hb = _box(b)
            assert abs(a["x"] - b["x"]) >= (wa + wb) / 2 - 1 or abs(a["y"] - b["y"]) >= (ha + hb) / 2 - 1
