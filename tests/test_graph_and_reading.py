"""Knowledge graph extraction/linking and 'What should I read next?'."""
from __future__ import annotations

from backend.app.services.knowledge_graph import build_graph, heuristic_extract


def _paper(pid, title, abstract, cats="cs.IR", published="2024-01-01"):
    return {"arxiv_id": pid, "title": title, "abstract": abstract,
            "categories": cats, "published": published}


A = _paper(
    "8001.00001", "Dense Passage Retrieval for Open-Domain Question Answering",
    "We propose Dense Passage Retrieval (DPR), a dual-encoder that learns dense "
    "representations. DPR outperforms BM25 and improves top-20 accuracy on "
    "Natural Questions and TriviaQA for open-domain question answering.",
)
B = _paper(
    "8001.00002", "RAG-Lite: Retrieval-Augmented Generation with Dense Retrievers",
    "We present RAG-Lite, which extends DPR with a generator. Evaluated on "
    "Natural Questions, it improves exact match over prior retrieval-augmented "
    "generation systems.",
)
C = _paper(
    "8001.00003", "Planar Shortest Paths",
    "We give a near-linear time algorithm for shortest paths in planar graphs.",
    cats="cs.DS",
)


def test_heuristic_extraction_finds_introduced_method_and_claims():
    ex = heuristic_extract(A)
    assert any("Dense Passage Retrieval" in m or m == "DPR" for m in ex.introduces)
    assert "BM25" in ex.uses
    assert {"Natural Questions", "TriviaQA"} <= set(ex.datasets)
    assert "Open-Domain QA" in ex.tasks
    assert "Accuracy" in ex.improves
    assert ex.aliases.get("DPR") == "Dense Passage Retrieval"


def test_title_name_is_introduced():
    assert "RAG-Lite" in heuristic_extract(B).introduces


def test_graph_links_paper_that_extends_another_papers_method():
    g = build_graph([A, B, C])
    nodes = {n["id"]: n for n in g["nodes"]}
    dpr = next(n for n in g["nodes"] if n["type"] == "method" and n["introduced_by"] == A["arxiv_id"])
    rels = {(e["source"], e["target"], e["relation"]) for e in g["edges"]}
    assert ("paper:8001.00001", dpr["id"], "introduces") in rels
    # B (or the method B introduces, RAG-Lite) extends A's method
    b_sources = {"paper:8001.00002"} | {
        n["id"] for n in g["nodes"] if n.get("introduced_by") == B["arxiv_id"]
    }
    assert any((src, dpr["id"], "extends") in rels for src in b_sources)
    nq = next(n for n in g["nodes"] if n["label"] == "Natural Questions")
    assert set(nq["papers"]) == {A["arxiv_id"], B["arxiv_id"]}
    assert any("evaluated on" == e["relation"] and e["target"] == nq["id"] for e in g["edges"])
    assert "paper:8001.00003" in nodes
    assert any("Natural Questions" in i for i in g["insights"])


def test_graph_edges_are_not_duplicated_across_relations():
    g = build_graph([A, B])
    pairs = [(e["source"], e["target"]) for e in g["edges"]
             if e["relation"] in ("uses", "extends")]
    assert len(pairs) == len(set(pairs))


def _ingest(pipeline, papers):
    from backend.app.services.arxiv_client import PaperMeta

    pipeline.ingest_metadata([
        PaperMeta(arxiv_id=p["arxiv_id"], title=p["title"], abstract=p["abstract"],
                  categories=p["categories"], published=p["published"])
        for p in papers
    ])


BRIDGE = _paper(
    "8001.00004", "Joint Training of Dense Passage Retrieval and Retrieval-Augmented Generation",
    "We connect dense passage retrieval with retrieval-augmented generation, "
    "training the DPR retriever end-to-end with the generator for open-domain "
    "question answering.",
)
NEAR_DUP = _paper(
    "8001.00005", "Dense Passage Retrieval for Open-Domain Question Answering Revisited",
    "We propose Dense Passage Retrieval (DPR), a dual-encoder that learns dense "
    "representations. DPR outperforms BM25 on Natural Questions.",
)


def test_read_next_prefers_the_bridging_paper(pipeline):
    _ingest(pipeline, [A, B, C, BRIDGE, NEAR_DUP])
    out = pipeline.read_next([A["arxiv_id"], B["arxiv_id"]], top_k=4, fetch_arxiv=False)
    recs = out["recommendations"]
    ids = [r["arxiv_id"] for r in recs]
    assert A["arxiv_id"] not in ids and B["arxiv_id"] not in ids
    assert BRIDGE["arxiv_id"] in ids
    bridge = next(r for r in recs if r["arxiv_id"] == BRIDGE["arxiv_id"])
    assert bridge["kind"] in ("Bridge", "Foundation")
    assert {b["from_id"] for b in bridge["bridges"]} == {A["arxiv_id"], B["arxiv_id"]}
    assert bridge["headline"].startswith("Read this")
    if C["arxiv_id"] in ids:  # unrelated paper must rank below the bridge
        assert ids.index(BRIDGE["arxiv_id"]) < ids.index(C["arxiv_id"])
    assert [k["arxiv_id"] for k in out["you_know"]] == [A["arxiv_id"], B["arxiv_id"]]


def test_read_next_marks_foundations(pipeline):
    _ingest(pipeline, [A, B])
    out = pipeline.read_next([B["arxiv_id"]], top_k=5, fetch_arxiv=False)
    # A (or its near-duplicate, which diversification may prefer) introduces
    # DPR, the method RAG-Lite extends -> recommended as its foundation
    dpr = next((r for r in out["recommendations"]
                if r["arxiv_id"] in (A["arxiv_id"], NEAR_DUP["arxiv_id"])), None)
    assert dpr is not None and dpr["kind"] == "Foundation"
    assert "RAG-Lite" in dpr["headline"] and "DPR" in dpr["headline"]


def test_read_next_unknown_ids_are_reported(pipeline):
    out = pipeline.read_next(["0000.00000"], fetch_arxiv=False)
    assert out["recommendations"] == [] and "0000.00000" in out["notice"]


def test_graph_and_read_next_api_routes(pipeline, monkeypatch):
    import backend.app.api.routes.api as api_mod
    from fastapi.testclient import TestClient

    from backend.app.main import app

    _ingest(pipeline, [A, B, BRIDGE])
    monkeypatch.setattr(api_mod, "get_pipeline", lambda: pipeline)
    client = TestClient(app)
    g = client.post("/api/v1/graph", json={"paper_ids": [A["arxiv_id"], B["arxiv_id"]]})
    assert g.status_code == 200 and g.json()["nodes"]
    r = client.post("/api/v1/read-next", json={"read_ids": [A["arxiv_id"], B["arxiv_id"]],
                                               "fetch_arxiv": False})
    assert r.status_code == 200 and r.json()["recommendations"]
    bad = client.post("/api/v1/read-next", json={"read_ids": []})
    assert bad.status_code == 422
