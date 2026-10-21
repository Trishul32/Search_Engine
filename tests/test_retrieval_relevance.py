"""Regression coverage for retrieval requiring all major query concepts."""
from __future__ import annotations


def _add_concept_fixture_papers(pipeline):
    from backend.app.services.arxiv_client import PaperMeta

    papers = [
        PaperMeta(
            arxiv_id="9901.00001",
            title="Machine Learning for Computer Networking",
            abstract=(
                "We apply machine learning to network traffic, routing, and "
                "congestion control in software defined networks."
            ),
            categories="cs.NI,cs.LG",
        ),
        PaperMeta(
            arxiv_id="9901.00002",
            title="Learning Cross-Lingual Sentence Representations via a Multi-task Dual-Encoder Model",
            abstract=(
                "This machine learning model studies multilingual NLP, sentence "
                "embeddings, natural language inference, and semantic textual similarity."
            ),
            categories="cs.CL",
        ),
        PaperMeta(
            arxiv_id="9901.00007",
            title="DOME: Recommendations for Supervised Machine Learning Validation in Biology",
            abstract=(
                "Recommendations for supervised machine learning validation, data "
                "splitting, optimization, evaluation, and reproducibility in biology."
            ),
            categories="q-bio.QM",
        ),
        PaperMeta(
            arxiv_id="9901.00003",
            title="Computer Network Routing and Protocol Performance",
            abstract="We analyze routing protocols, wireless networks, and network traffic.",
            categories="cs.NI",
        ),
        PaperMeta(
            arxiv_id="9901.00004",
            title="Deep Learning for Network Intrusion Detection",
            abstract=(
                "Deep neural networks classify malicious network traffic and detect "
                "intrusions in network security monitoring."
            ),
            categories="cs.CR,cs.LG",
        ),
        PaperMeta(
            arxiv_id="9901.00005",
            title="Deep Learning for Cross-Lingual Sentence Similarity",
            abstract="Deep learning and neural networks improve multilingual NLP embeddings.",
            categories="cs.CL",
        ),
        PaperMeta(
            arxiv_id="9901.00006",
            title="Reinforcement Learning for Congestion Control",
            abstract=(
                "A reinforcement learning policy adjusts TCP congestion control "
                "and network transmission rates."
            ),
            categories="cs.NI,cs.LG",
        ),
    ]
    pipeline.ingest_metadata(papers)
    return [paper.arxiv_id for paper in papers]


def _rank(pipeline, monkeypatch, query, ids):
    from backend.app.services.query_processor import process_query

    monkeypatch.setattr(
        pipeline.hybrid,
        "retrieve",
        lambda _pq: [(paper_id, 1.0 - index * 0.01) for index, paper_id in enumerate(ids)],
    )
    return pipeline._retrieve_and_rerank(process_query(query), top_k=len(ids))


def test_multidomain_ml_network_query_rejects_nlp_false_positive(pipeline, monkeypatch):
    ids = _add_concept_fixture_papers(pipeline)
    ranked = _rank(pipeline, monkeypatch, "machine learning in computer networking", ids)
    ranked_ids = [paper["arxiv_id"] for paper in ranked]

    assert "9901.00001" in ranked_ids
    assert "9901.00002" not in ranked_ids
    assert "9901.00003" not in ranked_ids
    assert "9901.00007" not in ranked_ids


def test_single_concept_queries_do_not_require_another_domain(pipeline, monkeypatch):
    ids = _add_concept_fixture_papers(pipeline)

    ml_ranked = _rank(pipeline, monkeypatch, "machine learning", ids)
    network_ranked = _rank(pipeline, monkeypatch, "computer networking", ids)

    assert "9901.00002" in [paper["arxiv_id"] for paper in ml_ranked]
    assert "9901.00003" in [paper["arxiv_id"] for paper in network_ranked]


def test_document_evidence_improves_single_concept_ranking(pipeline, monkeypatch):
    from backend.app.services.arxiv_client import PaperMeta
    from backend.app.services.query_processor import process_query

    papers = [
        PaperMeta(
            arxiv_id="9901.00010",
            title="Quantum Vector Database Synthesis",
            abstract="A synthesis of quantum vector database algorithms.",
            categories="quant-ph,cs.DS",
        ),
        PaperMeta(
            arxiv_id="9901.00011",
            title="Computer Network Routing and Protocol Performance",
            abstract="We analyze routing protocols, wireless networks, and network traffic.",
            categories="cs.NI",
        ),
    ]
    pipeline.ingest_metadata(papers)
    monkeypatch.setattr(
        pipeline.hybrid,
        "retrieve",
        lambda _pq: [(paper.arxiv_id, 1.0) for paper in papers],
    )

    def rerank(_query, candidates, top_k):
        for candidate, score in zip(candidates, (0.9, 0.8)):
            candidate["rerank_score"] = score
        return candidates[:top_k]

    monkeypatch.setattr(pipeline.reranker, "rerank", rerank)
    ranked = pipeline._retrieve_and_rerank(
        process_query("computer networking"), top_k=2
    )

    assert ranked[0]["arxiv_id"] == "9901.00011"
    assert ranked[0]["final_relevance_score"] > ranked[1]["final_relevance_score"]


def test_deep_learning_intrusion_query_requires_both_concepts(pipeline, monkeypatch):
    ids = _add_concept_fixture_papers(pipeline)
    ranked = _rank(pipeline, monkeypatch, "deep learning for network intrusion detection", ids)
    ranked_ids = [paper["arxiv_id"] for paper in ranked]

    assert "9901.00004" in ranked_ids
    assert "9901.00005" not in ranked_ids
    assert ranked[0]["concept_coverage"]["minimum"] == 1.0


def test_reinforcement_learning_congestion_query_requires_both_concepts(pipeline, monkeypatch):
    ids = _add_concept_fixture_papers(pipeline)
    ranked = _rank(pipeline, monkeypatch, "reinforcement learning for congestion control", ids)
    ranked_ids = [paper["arxiv_id"] for paper in ranked]

    assert "9901.00006" in ranked_ids
    assert ranked_ids[0] == "9901.00006"
    assert ranked[0]["concept_coverage"]["minimum"] == 1.0


def test_no_match_when_only_partial_multidomain_candidate_exists(pipeline, monkeypatch):
    from backend.app.core.config import settings

    ids = _add_concept_fixture_papers(pipeline)
    monkeypatch.setattr(settings, "corrective_retrieval", False)
    monkeypatch.setattr(
        pipeline.hybrid, "retrieve", lambda _pq: [(ids[1], 1.0)]
    )

    result = pipeline.search("machine learning in computer networking", fetch_arxiv=False)

    assert result["results"] == []


def test_concept_coverage_is_generic_and_based_on_text_evidence():
    from backend.app.services.relevance import concept_coverage

    concepts = [["machine", "learning"], ["healthcare"]]
    biology = concept_coverage(
        concepts,
        "Supervised machine learning validation in biology.",
        "q-bio.QM",
        title="DOME recommendations for machine learning validation in biology",
    )
    unrelated = concept_coverage(
        concepts,
        "Sentence representations and multilingual NLP embeddings.",
        "cs.CL",
    )

    assert biology["coverage"] == [1.0, 0.0]
    assert unrelated["minimum"] == 0.0


def test_arxiv_categories_corroborate_text_without_replacing_it():
    from backend.app.services.relevance import concept_coverage

    concepts = [["computer", "networking"]]
    relevant_network_paper = concept_coverage(
        concepts,
        "Federated learning over device-to-device fog networks.",
        "cs.NI,cs.LG",
        title="Multi-Stage Hybrid Federated Learning over Fog Networks",
    )
    category_only = concept_coverage(
        concepts,
        "Sentence representations and multilingual embeddings.",
        "cs.NI",
    )
    unrelated_term_with_category = concept_coverage(
        concepts,
        "Computer vision methods for image representation.",
        "cs.NI",
    )

    assert relevant_network_paper["minimum"] == 0.75
    assert category_only["minimum"] == 0.0
    assert unrelated_term_with_category["minimum"] < 0.75


def test_non_domain_concept_pairs_get_separate_query_groups():
    from backend.app.services.query_processor import process_query

    query = process_query("machine learning for biology")

    assert query.concepts == [["machine", "learning"], ["biology"]]
    assert query.requires_intersection


def test_query_parser_tracks_comparison_and_temporal_constraints():
    from backend.app.services.query_processor import process_query

    query = process_query("transformers vs RNNs for text classification in 2024")

    assert query.comparison
    assert query.requires_intersection
    assert query.requested_year == 2024
    assert query.concepts == [["transformers"], ["rnns"], ["text", "classification"]]


def test_focused_technical_phrase_requires_its_important_terms():
    from backend.app.services.query_processor import process_query

    query = process_query("transformer positional encoding")

    assert query.concepts == [["transformer", "positional", "encoding"]]
    assert query.requires_intersection


def test_natural_language_query_ignores_filler_words():
    from backend.app.services.query_processor import process_query

    query = process_query("I need recent papers about using neural networks to detect cyber attacks")

    assert query.recent
    assert query.concepts == [["neural", "networks"], ["cyber", "attacks"]]


def test_requested_year_filters_using_published_metadata(pipeline, monkeypatch):
    from backend.app.services.arxiv_client import PaperMeta
    from backend.app.services.query_processor import process_query

    papers = [
        PaperMeta(
            arxiv_id="9901.00008", title="Transformers for Classification",
            abstract="Transformer classification models.", published="2023-08-01",
        ),
        PaperMeta(
            arxiv_id="9901.00009", title="Transformers for Classification",
            abstract="Transformer classification models.", published="2024-03-01",
        ),
    ]
    pipeline.ingest_metadata(papers)
    monkeypatch.setattr(
        pipeline.hybrid, "retrieve", lambda _pq: [(paper.arxiv_id, 1.0) for paper in papers]
    )

    ranked = pipeline._retrieve_and_rerank(
        process_query("transformers for classification papers from 2024"), top_k=5
    )

    assert [paper["arxiv_id"] for paper in ranked] == ["9901.00009"]