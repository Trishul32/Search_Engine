"""LLM query rewriting: plain-language requests -> academic vocabulary.

arXiv's search API (and BM25) match *words*. Users describe problems in
everyday language ("stop language models from making up facts") while
abstracts use field terminology ("hallucination mitigation"). For such
conversational queries we ask the LLM for:

* ``keywords``       — the standard technical phrases, used for a second,
                        structured arXiv query (title/abstract fields);
* ``academic_query`` — a short reformulation shown to the user;
* ``pseudo_abstract``— two sentences an ideal paper's abstract would
                        contain. Embedding this (HyDE, Gao et al. 2022)
                        matches real abstracts far better than embedding
                        a short question.

The rewrite never overrides the user's intent: excluded concepts are
stripped from it, it only *adds* candidates, and the cross-encoder still
scores every paper against the user's own words. Any failure (no LLM,
timeout, bad JSON) simply means no rewrite.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

_PROMPT = """You turn plain-language research requests into the technical terms used in arXiv paper abstracts.

Examples:
Request: "teaching computers to recognize pictures with very few examples"
{{"academic_query": "few-shot image classification", "keywords": ["few-shot learning", "image classification", "meta-learning"], "pseudo_abstract": "We propose a few-shot learning method for image classification that generalizes to novel classes from only a handful of labeled examples. Our meta-learning approach outperforms prior work on miniImageNet."}}

Request: "making neural networks smaller so they run on phones"
{{"academic_query": "model compression for on-device inference", "keywords": ["model compression", "quantization", "pruning", "edge devices"], "pseudo_abstract": "We study model compression techniques, including quantization and pruning, to deploy deep neural networks on resource-constrained mobile devices. Our approach reduces model size and latency with minimal accuracy loss."}}

Now do the same. Reply with ONLY the JSON object.
Request: "{query}"
"""


@dataclass
class Rewrite:
    academic_query: str
    keywords: list[str] = field(default_factory=list)
    pseudo_abstract: str = ""

    def as_dict(self) -> dict:
        return {
            "academic_query": self.academic_query,
            "keywords": self.keywords,
            "pseudo_abstract": self.pseudo_abstract,
        }


# Everyday phrasing that signals a plain-language request. Noun-phrase
# queries ("dense retrieval for question answering") already use field
# vocabulary and are not worth an LLM call (~15-25s on a CPU).
_CONVERSATIONAL = re.compile(
    r"\b(?:how (?:can|do|does|to|could|should)|why|what (?:is|are|makes|happens)|ways? to|"
    r"methods? to|techniques? to|stop|prevent|avoid|make|making|help|helps|teach|teaching|"
    r"get rid of|deal with|figure out|tell|understand|without forgetting|so that|"
    r"computers? that|machines? that|robots? that|ai that|models? that)\b",
    re.I,
)


def needs_rewrite(pq) -> bool:
    """Rewrite only conversational, multi-word requests; technical keyword
    queries ("BM25 dense hybrid retrieval") already use field vocabulary."""
    mode = settings.query_rewrite
    if mode == "off" or not pq.keywords or pq.browse_only:
        return False
    if mode == "always":
        return True
    return (
        len(pq.keywords) >= 3
        and '"' not in pq.original
        and bool(_CONVERSATIONAL.search(pq.topic or pq.original))
    )


def _parse(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


@lru_cache(maxsize=256)
def _rewrite_cached(topic: str) -> Rewrite | None:
    from backend.app.services.llm.generator import get_llm

    llm = get_llm()
    if not llm.generative:
        return None
    try:
        # short output cap: the JSON needs ~120 tokens; generation is the
        # slow part on CPU (~10 tokens/s)
        raw = "".join(llm.generate_stream(_PROMPT.format(query=topic.replace('"', "'")), "", 200))
    except Exception as exc:
        logger.info("query rewrite failed: %s", exc)
        return None
    data = _parse(raw)
    if not data:
        return None
    keywords: list[str] = []
    for k in data.get("keywords", []):
        if not isinstance(k, str):
            continue
        # "LSH (Locality-Sensitive Hashing)" -> "LSH"; the parenthetical is
        # often an invented expansion, the head term is what papers use
        k = re.sub(r"\s*\(.*?\)\s*", " ", k).strip(" .,;")
        if 2 < len(k) <= 60 and k.lower() not in {x.lower() for x in keywords}:
            keywords.append(k)
    keywords = keywords[:5]
    academic = str(data.get("academic_query", "")).strip()[:120]
    pseudo = str(data.get("pseudo_abstract", "")).strip()[:600]
    if not keywords and not academic:
        return None
    return Rewrite(academic_query=academic, keywords=keywords, pseudo_abstract=pseudo)


def rewrite_query(topic: str, exclusions=None) -> Rewrite | None:
    """Rewrite a topic; drop any rewritten term that hits an exclusion."""
    rw = _rewrite_cached(topic.strip())
    if rw is None or not exclusions:
        return rw
    from backend.app.services.relevance import mentions

    def allowed(text: str) -> bool:
        return not any(mentions(p, text) for e in exclusions for p in e.phrases)

    return Rewrite(
        academic_query=rw.academic_query if allowed(rw.academic_query) else "",
        keywords=[k for k in rw.keywords if allowed(k)],
        pseudo_abstract=rw.pseudo_abstract if allowed(rw.pseudo_abstract) else "",
    )
