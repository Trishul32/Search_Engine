"""Query processing: normalization, optional expansion, and trait analysis.

Trait analysis feeds the dynamic-alpha fusion step (DAT-lite): keyword-heavy
technical queries lean on BM25, natural-language conceptual queries lean on
dense retrieval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "is", "are",
    "was", "were", "what", "which", "who", "how", "does", "do", "can", "with",
    "about", "that", "this", "these", "those", "be", "as", "at", "by", "from",
    "it", "its", "into", "their", "there", "than", "then", "when", "why",
    "papers", "paper", "research",
}

# Small curated synonym map for scientific CS vocabulary. Expansion adds
# recall for BM25 without hurting dense retrieval (expanded terms are only
# appended to the sparse query).
_EXPANSIONS: dict[str, list[str]] = {
    "llm": ["large language model"],
    "llms": ["large language models"],
    "rag": ["retrieval augmented generation"],
    "ir": ["information retrieval"],
    "nn": ["neural network"],
    "cnn": ["convolutional neural network"],
    "rnn": ["recurrent neural network"],
    "rl": ["reinforcement learning"],
    "nlp": ["natural language processing"],
    "cv": ["computer vision"],
    "ann": ["approximate nearest neighbor"],
    "kg": ["knowledge graph"],
    "qa": ["question answering"],
    "sota": ["state of the art"],
    "vector database": ["vector store", "ann index"],
    "embedding": ["dense representation"],
    "reranking": ["re-ranking", "rerank"],
    "hallucination": ["factual error", "faithfulness"],
    "fine-tuning": ["finetuning", "fine tuning"],
    "transformer": ["attention model"],
}

_MATH_OR_CODE = re.compile(r"[=+^_{}\\$]|->|\b[a-zA-Z]+\(\)")
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")


@dataclass
class ProcessedQuery:
    original: str
    normalized: str
    sparse_query: str            # normalized + expansions (for BM25)
    dense_query: str             # normalized (for embedding model)
    keywords: list[str] = field(default_factory=list)
    expansions: list[str] = field(default_factory=list)
    # traits for dynamic alpha
    technicality: float = 0.0    # 0 = conversational, 1 = keyword/technical
    is_question: bool = False


def normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _keywords(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9\-']+", text.lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 1]


def _technicality(original: str, keywords: list[str]) -> float:
    """Heuristic in [0,1]: how much the query looks like exact-term search."""
    if not keywords:
        return 0.5
    score = 0.0
    n_words = len(original.split())
    if n_words <= 4:
        score += 0.35                      # short queries ~ keyword lookups
    if _ACRONYM.search(original):
        score += 0.25                      # acronyms need lexical matching
    if _MATH_OR_CODE.search(original):
        score += 0.2
    if '"' in original:
        score += 0.2                       # quoted phrase = exact intent
    hyphenated = sum(1 for k in keywords if "-" in k or any(c.isdigit() for c in k))
    score += min(0.2, 0.1 * hyphenated)    # model names like gpt-4, bm25
    return min(1.0, score)


def process_query(query: str, expand: bool = True) -> ProcessedQuery:
    norm = normalize(query)
    kws = _keywords(norm)
    expansions: list[str] = []
    if expand:
        low = " " + norm.lower() + " "
        for term, alts in _EXPANSIONS.items():
            if f" {term} " in low or term in kws:
                expansions.extend(a for a in alts if a.lower() not in low)
    sparse_query = norm if not expansions else f"{norm} {' '.join(expansions)}"
    return ProcessedQuery(
        original=query,
        normalized=norm,
        sparse_query=sparse_query,
        dense_query=norm,
        keywords=kws,
        expansions=expansions,
        technicality=_technicality(norm, kws),
        is_question=norm.rstrip().endswith("?")
        or bool(re.match(r"(?i)^(what|why|how|when|which|who|does|do|can|compare|explain)\b", norm)),
    )
