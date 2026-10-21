"""Query processing: normalization, optional expansion, and trait analysis.

Trait analysis feeds the dynamic-alpha fusion step (DAT-lite): keyword-heavy
technical queries lean on BM25, natural-language conceptual queries lean on
dense retrieval.

Query understanding also separates three things the retrievers must never
mix up:

* the **positive topic** ("graph algorithms") -> embedded, BM25-matched and
  sent to the cross-encoder;
* **exclusions** ("without AI", "not about LLMs", "-blockchain") -> expanded
  into synonym phrases + arXiv categories and applied as hard filters both
  in the live arXiv query (``ANDNOT``) and on local candidates;
* **field filters** ("computer science", "in physics") -> arXiv category
  prefixes (``cat:cs.*``) instead of words that abstracts rarely contain.

Without this split, "computer science without AI" embeds *closer* to AI
papers (the word "AI" is in the query) and returns exactly what the user
asked to avoid.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "is", "are",
    "was", "were", "what", "which", "who", "how", "does", "do", "can", "with",
    "about", "that", "this", "these", "those", "be", "as", "at", "by", "from",
    "it", "its", "into", "their", "there", "than", "then", "when", "why",
    "papers", "paper", "research", "scientific", "i", "need", "want", "find", "show", "me",
    "looking", "using", "use", "recent", "latest", "new", "please",
    "without", "not", "no", "excluding", "except", "articles", "article",
    "publications", "related", "regarding", "give", "list", "some", "any",
    "works", "field", "area", "topic", "topics", "based", "approaches",
    "approach", "techniques", "technique", "methods", "method", "ones",
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
class Exclusion:
    """A concept the user asked to exclude, expanded for robust filtering."""
    label: str                      # as the user wrote it, e.g. "AI"
    phrases: list[str]              # text evidence that triggers exclusion
    categories: list[str]           # arXiv categories that trigger exclusion


@dataclass
class FieldFilter:
    """A research field mapped to arXiv category patterns."""
    label: str                      # e.g. "computer science"
    categories: list[str]           # e.g. ["cs.*"]


@dataclass
class ProcessedQuery:
    original: str
    normalized: str
    sparse_query: str            # normalized + expansions (for BM25)
    dense_query: str             # normalized (for embedding model)
    keywords: list[str] = field(default_factory=list)
    concepts: list[list[str]] = field(default_factory=list)
    requires_intersection: bool = False
    comparison: bool = False
    requested_year: int | None = None
    recent: bool = False
    expansions: list[str] = field(default_factory=list)
    # traits for dynamic alpha
    technicality: float = 0.0    # 0 = conversational, 1 = keyword/technical
    is_question: bool = False
    # query understanding: what to exclude / which fields to restrict to
    topic: str = ""                                  # positive topic text
    exclusions: list[Exclusion] = field(default_factory=list)
    fields: list[FieldFilter] = field(default_factory=list)
    # LLM rewrite into academic vocabulary (query_rewriter.Rewrite)
    rewrite: object | None = None
    # candidate pool size for hybrid retrieval (0 -> settings default);
    # widened when hard filters will discard part of the pool
    pool_size: int = 0

    @property
    def has_filters(self) -> bool:
        return bool(self.exclusions or self.fields)

    @property
    def browse_only(self) -> bool:
        """Field filter but no topic, e.g. 'computer science without AI'."""
        return bool(self.fields) and not self.keywords

    def interpretation(self) -> dict:
        """Human-readable account of how the query was understood (for UI)."""
        return {
            "topic": self.topic,
            "fields": [
                {"label": f.label, "categories": f.categories} for f in self.fields
            ],
            "excluded": [
                {
                    "label": e.label,
                    "phrases": e.phrases[:12],
                    "categories": e.categories,
                }
                for e in self.exclusions
            ],
            "year": self.requested_year,
            "recent": self.recent,
            "concepts": [" ".join(c) for c in self.concepts],
            "rewrite": self.rewrite.as_dict() if self.rewrite is not None else None,
            "browse": self.browse_only,
        }


def normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _keywords(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9\-']+", text.lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 1]


def _concept_groups(text: str) -> list[list[str]]:
    """Extract variable query concepts from relation and comparison structure."""
    topic = re.sub(
        r"(?i)^(?:i need|i want|find me|show me|looking for)\s+",
        "",
        text.strip(),
    )
    topic = re.sub(r"(?i)\b(?:recent|latest|newest)\s+(?:papers?|research)\b", " ", topic)
    topic = re.sub(r"\b(?:19|20)\d{2}\b", " ", topic)
    parts = re.split(
        r"\b(?:in|for|within|across|using|and|versus|vs\.?|"
        r"compared\s+(?:to|with)|relative\s+to|to|improve[sd]?|"
        r"improving|detect(?:s|ed|ing)?|identify(?:ing|ies)?|"
        r"classif(?:y|ies|ied|ying)|predict(?:s|ed|ing)?|"
        r"optimiz(?:e|es|ed|ing))\b|[,;]",
        topic,
        flags=re.I,
    )
    groups = [_keywords(part) for part in parts]
    groups = [group for group in groups if group]
    return groups or [_keywords(topic)]


def _query_constraints(text: str) -> tuple[bool, bool, int | None, bool]:
    comparison = bool(
        re.search(r"\b(?:vs\.?|versus|compared\s+(?:to|with))\b", text, re.I)
    )
    requires_intersection = bool(
        comparison
        or re.search(
            r"\b(?:in|for|within|across|using|and|to|"
            r"improve[sd]?|improving|detect(?:s|ed|ing)?|"
            r"identify(?:ing|ies)?|classif(?:y|ies|ied|ying)|"
            r"predict(?:s|ed|ing)?|optimiz(?:e|es|ed|ing))\b|[,;]",
            text,
            re.I,
        )
    )
    if len(_keywords(text)) >= 3 and not comparison:
        requires_intersection = True
    year_match = re.search(r"\b(?:19|20)\d{2}\b", text)
    requested_year = int(year_match.group()) if year_match else None
    recent = bool(re.search(r"\b(?:recent|latest|newest)\b", text, re.I))
    return requires_intersection, comparison, requested_year, recent


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


# --------------------------------------------------------------------------
# Negation / exclusion understanding
# --------------------------------------------------------------------------
# Excluding a concept must also exclude its close synonyms and its arXiv
# home categories: "without AI" has to drop deep-learning and LLM papers
# too, otherwise the filter is cosmetic. Keys are lowercase surface forms.
_AI_PHRASES = [
    "artificial intelligence", "AI", "machine learning", "deep learning",
    "neural network", "neural networks", "large language model",
    "large language models", "language model", "language models", "LLM",
    "LLMs", "reinforcement learning", "generative AI", "GPT", "ChatGPT",
    "deep neural", "supervised learning", "unsupervised learning",
]
_AI_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "cs.MA", "stat.ML"]
_ML_PHRASES = [
    "machine learning", "deep learning", "neural network", "neural networks",
    "supervised learning", "unsupervised learning", "reinforcement learning",
    "deep neural",
]
_LLM_PHRASES = [
    "large language model", "large language models", "LLM", "LLMs", "GPT",
    "ChatGPT",
]
_EXCLUSION_CONCEPTS: dict[str, tuple[list[str], list[str]]] = {
    "ai": (_AI_PHRASES, _AI_CATEGORIES),
    "a.i.": (_AI_PHRASES, _AI_CATEGORIES),
    "artificial intelligence": (_AI_PHRASES, _AI_CATEGORIES),
    "machine learning": (_ML_PHRASES, ["cs.LG", "stat.ML"]),
    "ml": (_ML_PHRASES + ["ML"], ["cs.LG", "stat.ML"]),
    "deep learning": (
        ["deep learning", "neural network", "neural networks", "deep neural",
         "convolutional"], ["cs.NE"]),
    "dl": (["deep learning", "neural network", "neural networks", "deep neural"], ["cs.NE"]),
    "neural networks": (
        ["neural network", "neural networks", "deep learning", "deep neural"], ["cs.NE"]),
    "neural network": (
        ["neural network", "neural networks", "deep learning", "deep neural"], ["cs.NE"]),
    "neural": (["neural network", "neural networks", "deep learning", "deep neural"], ["cs.NE"]),
    "llm": (_LLM_PHRASES, []),
    "llms": (_LLM_PHRASES, []),
    "large language model": (_LLM_PHRASES, []),
    "large language models": (_LLM_PHRASES, []),
    "language models": (["language model", "language models"] + _LLM_PHRASES, []),
    "nlp": (["natural language processing", "NLP", "language model", "language models"], ["cs.CL"]),
    "natural language processing": (
        ["natural language processing", "NLP", "language model"], ["cs.CL"]),
    "computer vision": (
        ["computer vision", "image recognition", "object detection",
         "image classification"], ["cs.CV"]),
    "cv": (["computer vision", "image recognition", "object detection"], ["cs.CV"]),
    "reinforcement learning": (["reinforcement learning", "RL"], []),
    "rl": (["reinforcement learning", "RL"], []),
    "blockchain": (
        ["blockchain", "blockchains", "cryptocurrency", "smart contract",
         "smart contracts", "bitcoin", "ethereum"], []),
    "quantum": (["quantum"], ["quant-ph"]),
    "quantum computing": (["quantum computing", "quantum computer", "qubit", "qubits"], ["quant-ph"]),
    "robotics": (["robot", "robots", "robotics", "robotic"], ["cs.RO"]),
    "cryptography": (["cryptography", "cryptographic", "encryption"], ["cs.CR"]),
    "rag": (["retrieval augmented generation", "retrieval-augmented generation", "RAG"], []),
    "transformers": (["transformer", "transformers", "self-attention"], []),
    "transformer": (["transformer", "transformers", "self-attention"], []),
}

# Words that introduce an excluded concept.
# "Weak" triggers (without / with no / and no) are ambiguous: "graph
# algorithms without machine learning" excludes a topic, but "learn new
# skills without forgetting" or "training without labels" DESCRIBES the
# method being searched for. Weak triggers exclude only known topics or in
# paper-framed queries ("papers on X without Y"); strong triggers always do.
_WEAK_NEG = r"without(?:\s+any)?|with\s+no|and\s+no"
_NEG_TRIGGER = (
    rf"(?:{_WEAK_NEG}|excluding|exclude|except(?:\s+for)?|"
    r"other\s+than|but\s+not|and\s+not|minus|"
    r"not\s+(?:about|on|related\s+to|relating\s+to|involving|including|using|"
    r"based\s+on|covering|focused\s+on))"
)
_WEAK_NEG_RE = re.compile(rf"^(?:{_WEAK_NEG})$", re.I)
_PAPER_FRAMED = re.compile(
    r"^\s*(?:(?:find|show|give|list)\s+(?:me\s+)?|i\s+(?:need|want)\s+)?"
    r"(?:(?:some|any|recent|latest|new)\s+)?"
    r"(?:papers?|research|articles?|works|publications|studies|literature)\b",
    re.I,
)
# A bare "not"/"no" is a negation only at a clause start ("X, no AI",
# "not AI: ..."); mid-sentence ("how do LLMs not hallucinate") it is part
# of the question and must stay in the topic.
_BARE_NEG = re.compile(r"(^|[,;:(])\s*(?:not|no)\s+(?=[A-Za-z0-9])", re.I)
# Optional filler between the trigger and the excluded concept.
_NEG_FILLER = (
    r"(?:\s+(?:any|the|use\s+of|using|related\s+to|involving|mention\s+of|"
    r"papers?\s+(?:on|about)|an?)(?=\s))*"
)
# Connector words that end an excluded concept.
_NEG_STOP = (
    r"(?:and|but|in|for|with|that|which|from|published|since|after|before|"
    r"during|between|by|to|on|about|focus(?:ing|ed)?|while|where|or|nor|"
    r"papers?|please)"
)
_WORD = r"[A-Za-z0-9][\w\-\.\+]*"
# An excluded concept: 1-4 words, stopping at punctuation or connector words.
_NEG_CONCEPT = rf"(?!{_NEG_STOP}\b){_WORD}(?:\s+(?!{_NEG_STOP}\b){_WORD}){{0,3}}"
_NEGATION = re.compile(
    rf"\b({_NEG_TRIGGER}){_NEG_FILLER}\s+({_NEG_CONCEPT})"
    rf"((?:\s*(?:,?\s*or|,?\s*nor|/)\s+{_NEG_CONCEPT})*)",
    re.I,
)
_NEG_LIST_ITEM = re.compile(rf"(?:or|nor|/)\s+({_NEG_CONCEPT})", re.I)
# "-blockchain" (search-engine style) and "NOT blockchain" (boolean style)
_MINUS_TERM = re.compile(r"(?:(?<=\s)|^)-([A-Za-z][\w\-]*)")
_BOOL_NOT = re.compile(r"\bNOT\s+([A-Za-z][\w\-]*)")
_NON_PREFIX = re.compile(
    r"\bnon[- ](ai|ml|llms?|neural|deep[- ]learning|quantum|machine[- ]learning)\b", re.I
)


def _expand_exclusion(label: str) -> Exclusion:
    key = re.sub(r"\s+", " ", label.lower().strip(" .,-;"))
    # drop trailing generic words: "AI techniques", "LLM-based methods"
    trimmed = re.sub(
        r"(?:[\s-](?:based|techniques?|methods?|approaches?|models?|papers?|"
        r"research|stuff|related|things|tools?|systems?|algorithms?|works?))+$",
        "",
        key,
    ).strip()
    key = trimmed or key
    key = key.replace("-", " ") if key.replace("-", " ") in _EXCLUSION_CONCEPTS else key
    if key in _EXCLUSION_CONCEPTS:
        phrases, cats = _EXCLUSION_CONCEPTS[key]
        return Exclusion(label=label.strip(), phrases=list(phrases), categories=list(cats))
    phrases = [key]
    if key.endswith("s") and len(key) > 4:
        phrases.append(key[:-1])
    elif len(key) > 3 and not key.endswith(("ed", "ing", "al", "ic", "ive", "y")):
        phrases.append(key + "s")
    phrases += _EXPANSIONS.get(key, [])
    return Exclusion(label=label.strip(), phrases=phrases, categories=[])


def _is_known_topic(concept: str) -> bool:
    """Is this a recognisable research topic (vs. a descriptive phrase)?"""
    first = concept.split()[0] if concept.split() else ""
    if first.lower().endswith("ing") and first.lower() not in {"computing", "learning"}:
        return False                       # "without forgetting", "without using"
    key = _expand_exclusion(concept).phrases[0].lower()
    if key in _EXCLUSION_CONCEPTS or concept.lower().strip() in _EXCLUSION_CONCEPTS:
        return True
    if key in _FIELDS_MULTI or key in _FIELDS_SINGLE:
        return True
    return bool(re.fullmatch(r"[A-Z][A-Z0-9\-]{1,6}s?", concept.strip()))  # GPU, LLMs


def extract_exclusions(text: str) -> tuple[str, list[Exclusion]]:
    """Split a query into (positive text, exclusions).

    Handles natural phrasing ("papers on X without Y", "X but not Y or Z",
    "X, excluding the use of Y", "non-AI X") and boolean style ("X -Y",
    "X NOT Y"). The negated spans are removed from the positive text so
    they can never act as *positive* evidence in BM25 / dense retrieval /
    the cross-encoder (embedding "X without AI" lands next to AI papers).
    """
    labels: list[str] = []

    def _collect(group: int):
        def _sub(match: re.Match) -> str:
            labels.append(match.group(group))
            return " , "
        return _sub

    paper_framed = bool(_PAPER_FRAMED.match(text))

    def _negation(match: re.Match) -> str:
        trigger, concept, more = match.group(1), match.group(2), match.group(3)
        if _WEAK_NEG_RE.match(trigger.strip()) and not (
            paper_framed or _is_known_topic(concept)
        ):
            return match.group(0)          # descriptive: keep in the topic
        labels.append(concept)
        labels.extend(_NEG_LIST_ITEM.findall(more or ""))
        return " , "

    positive = _BARE_NEG.sub(lambda m: m.group(1) + " without ", text)
    positive = _NON_PREFIX.sub(_collect(1), positive)
    positive = _BOOL_NOT.sub(_collect(1), positive)
    positive = _MINUS_TERM.sub(_collect(1), positive)
    positive = _NEGATION.sub(_negation, positive)

    exclusions: list[Exclusion] = []
    seen: set[str] = set()
    for label in labels:
        label = (label or "").strip(" ,.;")
        if not label or label.lower() in seen or label.lower() in _STOPWORDS:
            continue
        seen.add(label.lower())
        exclusions.append(_expand_exclusion(label))
    positive = re.sub(r"(?:\s*,)+", ",", positive)
    positive = re.sub(r"\s+", " ", positive).strip(" ,;")
    return positive, exclusions


def add_exclusions(pq: "ProcessedQuery", labels: list[str]) -> None:
    """Merge explicit exclusions from the UI ('Exclude topics' box)."""
    known = {e.label.lower() for e in pq.exclusions}
    for label in labels:
        label = label.strip()
        if label and label.lower() not in known:
            pq.exclusions.append(_expand_exclusion(label))
            known.add(label.lower())


# --------------------------------------------------------------------------
# Research-field understanding -> arXiv categories
# --------------------------------------------------------------------------
# Multi-word field names are unambiguous and always mapped. Single-word
# names ("physics", "biology") are mapped only when used *as a field*
# ("papers in physics", or the whole topic): "physics-informed neural
# networks" must stay a topic, not become a category filter.
_FIELDS_MULTI: dict[str, list[str]] = {
    "theoretical computer science": ["cs.CC", "cs.DS", "cs.LO", "cs.FL", "cs.DM"],
    "computer science": ["cs.*"],
    "electrical engineering": ["eess.*"],
    "quantitative biology": ["q-bio.*"],
    "computational biology": ["q-bio.*"],
    "quantitative finance": ["q-fin.*"],
    "condensed matter": ["cond-mat.*"],
    "high energy physics": ["hep-th", "hep-ph", "hep-ex", "hep-lat"],
    "quantum physics": ["quant-ph"],
}
_FIELDS_SINGLE: dict[str, list[str]] = {
    "mathematics": ["math.*"],
    "maths": ["math.*"],
    "math": ["math.*"],
    "physics": ["physics.*", "cond-mat.*", "quant-ph", "hep-th", "hep-ph",
                "astro-ph.*", "gr-qc", "nucl-th"],
    "statistics": ["stat.*"],
    "biology": ["q-bio.*"],
    "economics": ["econ.*", "q-fin.*"],
    "finance": ["q-fin.*"],
    "astrophysics": ["astro-ph.*"],
    "astronomy": ["astro-ph.*"],
}
_FIELD_INTRO = (
    r"(?:in|on|from|within|across)\s+(?:the\s+)?"
    r"(?:(?:field|domain|area)\s+of\s+)?"
)
_FIELD_ALONE = r"\s*(?:(?:papers?|research|articles?)\s+(?:on|in|about|from)\s+)?{}\s*(?:papers?|research)?\s*"


def extract_fields(text: str) -> tuple[str, list[FieldFilter]]:
    """Map research-field names to arXiv category filters."""
    fields: list[FieldFilter] = []
    out = text
    for name, cats in _FIELDS_MULTI.items():
        pat = re.compile(rf"(?:\b{_FIELD_INTRO})?\b{name}\b", re.I)
        if pat.search(out):
            fields.append(FieldFilter(label=name, categories=cats))
            out = pat.sub(" ", out)
    for name, cats in _FIELDS_SINGLE.items():
        intro = re.compile(rf"\b{_FIELD_INTRO}{name}\b(?![\w-])", re.I)
        if intro.search(out):
            fields.append(FieldFilter(label=name, categories=cats))
            out = intro.sub(" ", out)
        elif re.fullmatch(_FIELD_ALONE.format(name), out, re.I):
            fields.append(FieldFilter(label=name, categories=cats))
            out = ""
    out = re.sub(r"\s+", " ", out).strip(" ,;")
    return out, fields


# --------------------------------------------------------------------------
# Structured arXiv API query
# --------------------------------------------------------------------------
def _q(term: str) -> str:
    term = re.sub(r'["()]', " ", term).strip()
    return f'"{term}"' if term else ""


def _group_clause(group: list[str], relaxed: bool = False) -> str:
    """Match one concept group against title OR abstract.

    Multi-word groups match the exact phrase OR all of their words inside
    the abstract (word order and inflection differ between a query and an
    abstract), so papers whose *abstract* matches are found even when
    their title shares no words with the query.
    """
    terms = [t for t in group if t]
    if not terms:
        return ""
    alts = [a for t in terms for a in _EXPANSIONS.get(t, [])]
    if len(terms) == 1:
        parts = [f"ti:{_q(terms[0])}", f"abs:{_q(terms[0])}"]
    else:
        phrase = _q(" ".join(terms))
        parts = [
            f"ti:{phrase}",
            f"abs:{phrase}",
            "(" + " AND ".join(f"abs:{_q(t)}" for t in terms) + ")",
        ]
        if relaxed and len(terms) > 3:
            # long descriptive groups rarely match word-for-word: accept
            # any adjacent word pair as a phrase
            parts += [f"abs:{_q(a + ' ' + b)}" for a, b in zip(terms, terms[1:])]
    parts += [f"abs:{_q(a)}" for a in alts]
    return "(" + " OR ".join(parts) + ")"


def _field_clause(fields: list[FieldFilter]) -> str:
    cats = list(dict.fromkeys(c for f in fields for c in f.categories))
    if not cats:
        return ""
    return "(" + " OR ".join(f"cat:{c}" for c in cats) + ")"


def _exclusion_clause(exclusions: list[Exclusion]) -> str:
    parts: list[str] = []
    for e in exclusions:
        parts += [f"cat:{c}" for c in e.categories]
        for p in e.phrases:
            if _q(p):
                parts += [f"ti:{_q(p)}", f"abs:{_q(p)}"]
    parts = list(dict.fromkeys(parts))
    return "(" + " OR ".join(parts) + ")" if parts else ""


def build_arxiv_query(
    pq: "ProcessedQuery",
    relaxed: bool = False,
    concepts: list[list[str]] | None = None,
) -> str:
    """Build an arXiv API ``search_query`` from the understood query.

    strict : every concept group must match the title or abstract
    relaxed: any concept group may match (fallback when strict is thin)
    Both apply field filters (``cat:``) and exclusions (``ANDNOT``).
    ``concepts`` overrides the query's own concept groups (used for the
    LLM-rewritten keywords, which are alternatives -> pass relaxed=True).
    Returns "" when there is nothing positive to search for.
    """
    source = concepts if concepts is not None else (pq.concepts if pq.keywords else [])
    groups = [g for g in (_group_clause(c, relaxed) for c in source) if g]
    positive = ""
    if groups:
        joiner = " OR " if relaxed else " AND "
        positive = groups[0] if len(groups) == 1 else "(" + joiner.join(groups) + ")"
    query = " AND ".join(x for x in (positive, _field_clause(pq.fields)) if x)
    if not query:
        return ""
    if pq.requested_year:
        y = pq.requested_year
        query = f"{query} AND submittedDate:[{y}01010000 TO {y}12312359]"
    excl = _exclusion_clause(pq.exclusions)
    return f"{query} ANDNOT {excl}" if excl else query


def process_query(
    query: str,
    expand: bool = True,
    exclude: list[str] | None = None,
    field_filter: str | None = None,
) -> ProcessedQuery:
    norm = normalize(query)
    # Understand the query first: pull out exclusions and field filters so
    # only the positive topic drives retrieval and reranking.
    positive, exclusions = extract_exclusions(norm)
    topic, fields = extract_fields(positive)
    if field_filter:
        _, extra = extract_fields(f"in {field_filter}")
        known = {f.label for f in fields}
        fields += [f for f in extra if f.label not in known]
    requires_intersection, comparison, requested_year, recent = _query_constraints(topic)
    if requested_year is not None:
        # the year is a filter, not a topic word: "graph algorithms, 2024"
        # must not show or match "2024" as part of the topic
        stripped = re.sub(
            r"(?:\b(?:in|from|since|during|of|published(?:\s+in)?)\s+)?\b(?:19|20)\d{2}\b",
            " ", topic, flags=re.I,
        )
        stripped = re.sub(r"\s+", " ", stripped).strip(" ,;")
        if _keywords(stripped):
            topic = stripped
    kws = _keywords(topic)
    expansions: list[str] = []
    if expand:
        low = " " + topic.lower() + " "
        for term, alts in _EXPANSIONS.items():
            if f" {term} " in low or term in kws:
                expansions.extend(a for a in alts if a.lower() not in low)
    # What the retrievers see. With no topic left (pure field browse, e.g.
    # "computer science without AI") fall back to the field names.
    search_text = topic if kws else (" ".join(f.label for f in fields) or topic)
    sparse_query = search_text if not expansions else f"{search_text} {' '.join(expansions)}"
    pq = ProcessedQuery(
        original=query,
        normalized=search_text,
        sparse_query=sparse_query,
        dense_query=search_text,
        keywords=kws,
        concepts=_concept_groups(topic) if kws else [],
        requires_intersection=requires_intersection,
        comparison=comparison,
        requested_year=requested_year,
        recent=recent,
        expansions=expansions,
        technicality=_technicality(search_text, kws),
        is_question=norm.rstrip().endswith("?")
        or bool(re.match(r"(?i)^(what|why|how|when|which|who|does|do|can|compare|explain)\b", norm)),
        topic=topic if kws else "",
        exclusions=exclusions,
        fields=fields,
    )
    if exclude:
        add_exclusions(pq, exclude)
    return pq
