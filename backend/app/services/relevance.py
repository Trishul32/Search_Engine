"""Relevance evidence for paper candidates.

* concept coverage for multi-concept queries,
* hard exclusion / research-field filters from query understanding,
* best-matching abstract sentence ("why this paper matched").
"""
from __future__ import annotations

import re

from backend.app.core.config import settings


_TOKEN = re.compile(r"[a-z0-9]+")

# Broad arXiv category descriptions provide corroborating evidence only;
# paper text must still contain at least one term from the query concept.
_ARXIV_CATEGORY_TERMS: dict[str, set[str]] = {
    "cs.ni": {"network", "internet", "wireless", "communication"},
    "cs.lg": {"machine", "learning", "reinforcement"},
    "cs.cr": {"security", "cryptography", "intrusion"},
    "cs.dc": {"distributed", "computing", "network"},
    "cs.cv": {"vision", "visual", "image"},
    "cs.cl": {"language", "linguistics", "nlp"},
    "cs.ir": {"retrieval", "search", "information"},
    "q-bio": {"biology", "biomedical", "bioinformatics"},
}


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("-", " "))


def _stem(token: str) -> str:
    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]
        if len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]
        return token
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("tion") and len(token) > 6:
        return token[:-3]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def concept_coverage(
    concepts: list[list[str]],
    text: str,
    categories: str = "",
    *,
    title: str = "",
) -> dict:
    """Measure per-concept lexical support from title and abstract text.

    Categories are intentionally not decisive: arXiv codes are not a stable
    vocabulary for arbitrary user concepts and cannot replace paper evidence.
    """
    abstract_stems = {_stem(token) for token in _tokens(text)}
    title_stems = {_stem(token) for token in _tokens(title)}
    category_set = {item.strip().lower() for item in categories.split(",") if item.strip()}
    coverage: list[float] = []
    title_coverage: list[float] = []
    for concept in concepts:
        terms = list(dict.fromkeys(_stem(token) for token in concept if len(token) > 1))
        if not terms:
            coverage.append(1.0)
            title_coverage.append(0.0)
            continue

        weights = {term: 1 for term in terms}
        text_stems = abstract_stems | title_stems
        matched = {term for term in terms if term in text_stems}
        matched_weight = sum(weights[term] for term in matched)
        total_weight = sum(weights.values())
        raw_coverage = matched_weight / total_weight

        category_terms = set().union(*(
            {_stem(term) for term in terms_for_category}
            for category in category_set
            for prefix, terms_for_category in _ARXIV_CATEGORY_TERMS.items()
            if category == prefix or category.startswith(f"{prefix}.")
        )) if category_set else set()
        category_support = bool(matched.intersection(category_terms))
        category_boost = (
            settings.retrieval_category_evidence_boost if category_support else 0.0
        )
        coverage.append(min(1.0, raw_coverage + category_boost))
        title_coverage.append(
            sum(weights[term] for term in terms if term in title_stems) / total_weight
        )

    return {
        "concepts": [" ".join(concept) for concept in concepts],
        "coverage": coverage,
        "score": sum(coverage) / len(coverage) if coverage else 1.0,
        "minimum": min(coverage) if coverage else 1.0,
        "title_score": sum(title_coverage) / len(title_coverage) if title_coverage else 0.0,
        "complete": all(score == 1.0 for score in coverage),
    }

# ---------------------------------------------------------------- filters
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def _phrase_pattern(phrase: str) -> re.Pattern:
    """Word-bounded phrase matcher. Short all-caps acronyms (AI, ML, RL)
    match case-sensitively so e.g. the word 'ai' inside names or units
    never triggers an exclusion; everything else is case-insensitive and
    tolerant to hyphen/space variation ("retrieval-augmented")."""
    words = [re.escape(w) for w in re.split(r"[\s\-]+", phrase.strip()) if w]
    body = r"[\s\-]+".join(words)
    if phrase.isupper() and len(phrase) <= 5:
        return re.compile(rf"(?<![A-Za-z0-9]){body}s?(?![A-Za-z0-9])")
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.I)


_PATTERN_CACHE: dict[str, re.Pattern] = {}


def mentions(phrase: str, text: str) -> bool:
    pat = _PATTERN_CACHE.get(phrase)
    if pat is None:
        pat = _PATTERN_CACHE[phrase] = _phrase_pattern(phrase)
    return bool(pat.search(text))


def _category_matches(category: str, pattern: str) -> bool:
    category, pattern = category.strip().lower(), pattern.strip().lower()
    if pattern.endswith(".*"):
        return category.startswith(pattern[:-1]) or category == pattern[:-2]
    return category == pattern


def exclusion_hit(exclusions, paper: dict) -> str | None:
    """Return a human-readable reason if the paper violates an exclusion.

    A paper is excluded when its arXiv categories fall in an excluded
    concept's home categories, or its title/abstract mentions the concept
    (or a synonym). Strict by design: "without AI" means without AI.
    """
    if not exclusions:
        return None
    cats = [c for c in (paper.get("categories") or "").split(",") if c.strip()]
    text = f"{paper.get('title', '')}. {paper.get('abstract', '')}"
    for exc in exclusions:
        for cat in cats:
            if any(_category_matches(cat, pattern) for pattern in exc.categories):
                return f"category {cat.strip()} ({exc.label})"
        for phrase in exc.phrases:
            if mentions(phrase, text):
                return f"mentions '{phrase}' ({exc.label})"
    return None


def field_match(fields, paper: dict) -> bool:
    """True when the paper belongs to at least one requested field."""
    if not fields:
        return True
    cats = [c for c in (paper.get("categories") or "").split(",") if c.strip()]
    return any(
        _category_matches(cat, pattern)
        for f in fields for pattern in f.categories for cat in cats
    )


def best_evidence_sentence(query_terms: list[str], abstract: str) -> str:
    """Lexical fallback for the abstract sentence that best supports a
    query (used when no embedder is passed)."""
    sentences = [s.strip() for s in _SENTENCE.split(abstract or "") if len(s.strip()) > 25]
    if not sentences:
        return ""
    stems = {_stem(t) for t in query_terms}
    def score(sentence: str) -> float:
        toks = {_stem(t) for t in _tokens(sentence)}
        return len(stems & toks) + 0.001 * len(toks)
    return max(sentences, key=score)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text or "") if len(s.strip()) > 25]
