"""Concept map for ONE paper: the topics you need to understand it.

Given a paper, this builds a two-level map:

    paper ─┬─ major topic (most important, closest)
           │     ├─ subtopic
           │     └─ subtopic
           └─ major topic …

How it works (no extra dependencies; runs on CPU in a couple of seconds):

1. Text: title, abstract and every indexed full-text passage of the paper
   (ligatures and PDF hyphenation repaired). Without full text, the
   abstract alone is used.
2. Candidates: key phrases found by splitting sentences at stop words and
   punctuation (RAKE-style), plus known methods / datasets / metrics /
   tasks from the knowledge-graph gazetteers. Acronyms are merged with
   their long form ("RAG" -> "retrieval-augmented generation").
3. Importance of each phrase, in [0, 1]:
     0.35 centrality   - embedding similarity to the paper's title+abstract
     0.25 frequency    - how often the full text uses it
     0.15 spread       - in how many parts of the paper it appears
     0.15 prominence   - in the title (1.0) or abstract (0.6)
     0.10 specificity  - how rare it is across the rest of the corpus
4. Major topics: the most important phrases, skipping near-duplicates of
   ones already chosen. Subtopics: remaining phrases attached to the most
   similar major topic, most important first.
5. For every topic: the sentence that best explains it in the paper
   (definitions preferred), where it appears, and its ROLE in the paper
   (background / core idea / method / contribution / evaluation) inferred
   from cue words around its mentions and where in the paper they occur.
6. Optionally, the LLM writes one plain-language explanation per topic.
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
from sqlalchemy import select

from backend.app.db.models import Paper, PaperEntities
from backend.app.db.session import db_session
from backend.app.services.knowledge_graph import (
    _DATASETS,
    _METHODS,
    _METRICS,
    _TASKS,
    canonical,
    heuristic_extract,
    norm_key,
)

logger = logging.getLogger(__name__)

VERSION = "conceptmap-v2"
MAX_MAJOR = 6
MAX_SUB = 5

_STOP = set("""
a an the and or but if then else of to in on at by for with from as into onto about over under
between among through during before after above below up down out off again further once here there
when where why how all any both each few more most other some such no nor not only own same so than
too very can will just don should now is are was were be been being have has had having do does did
doing this that these those i me my we our ours you your he him his she her it its they them their
what which who whom whose am also via per within without upon while whereas thus hence therefore
however moreover furthermore although though because since unless until whether either neither yet
still even ever every much many several various e.g i.e etc et al us let shall could would should
might must may cannot often usually typically mainly mostly first second third finally
someone something anyone anything everyone everything nothing whereby thereby
""".split())

_GENERIC = set("""
paper papers work works study studies approach approaches method methods methodology result results
propose proposed proposes proposing show shows shown showing use used uses using based novel new
different well make makes made making number numbers set sets way ways order part parts important
significant significantly problem problems task tasks data model models system systems framework
frameworks technique techniques analysis performance example examples case cases section sections
figure figures fig table tables appendix respectively include includes including included provide
provides provided present presents presented given give gives obtain obtained achieve achieves
achieved found find finds consider considered considering existing current previous recent prior
better best good main key overall general specific particular related able one two three four five
six ten type types kind kinds level levels form forms term terms point points fact aspect aspects
issue issues process processes article authors author researchers research field fields area areas
literature review reviews question questions answer answers lot lots thing things time times year
years number total average value values amount rate step steps stage stages end means effect effects
impact impacts role roles need needs use cases goal goals aim aims purpose idea ideas
""".split())

# Verbs end a noun phrase ("neural networks learn" -> "neural networks").
# Words that are just as often nouns (access, study, design...) are kept
# out of this list and only dropped when they stand alone.
_VERBS = set("""
argue lack learn generate explore demonstrate outperform require allow enable improve achieve rely
produce cause lead help suggest indicate seem appear become remain take propose show make give find
consider obtain perform compute evaluate compare predict apply construct encode retrieve combine
contain include involve introduce present describe discuss address seek tend hold depend differ vary
increase decrease reduce exceed fail succeed deliver offer enhance capture affect continue begin want
know think believe note observe reveal yield highlight emphasize illustrate identify investigate
examine analyze analyse assess measure explain define determine establish ensure maintain replace
extend modify adapt raise operate build develop implement integrate leverage utilize employ adopt
follow outline summarize mitigate prevent avoid create provide serve represent remove select treat
drop hallucinate pair store ground fall rise grow emerge occur exist arise matter fit
""".split())
# trailing generic nouns that still form real terms ("language models",
# "knowledge-intensive tasks"); others ("methods", "results") are noise
_KEEP_TRAILING = set("""
model models task tasks system systems data framework frameworks analysis problem problems
performance process processes network networks
""".split())
_AMBIGUOUS = set("""
access study test design need support report focus change update aim claim limit use score rank
prompt train share run start review answer question work result process control
""".split())


def _verb_form(word: str) -> str | None:
    w = word.lower()
    if w in _AMBIGUOUS:
        return None
    if w in _VERBS:
        return "base"
    def base_in(b: str) -> bool:
        return b in _VERBS or b + "e" in _VERBS or (len(b) > 2 and b[-1] == b[-2] and b[:-1] in _VERBS)
    if w.endswith("ing") and base_in(w[:-3]):
        return "ing"
    if w.endswith("ied") and w[:-3] + "y" in _VERBS:
        return "ed"
    if w.endswith("ed") and (base_in(w[:-2]) or w[:-1] in _VERBS):
        return "ed"
    if w.endswith("ies") and w[:-3] + "y" in _VERBS:
        return "s"
    if w.endswith("es") and w[:-2] in _VERBS or w.endswith("s") and w[:-1] in _VERBS:
        return "s"
    return None


_ROLE_CUES = {
    "contribution": re.compile(
        r"\b(?:we (?:propose|introduce|present|develop|design|contribute)|our (?:method|approach|"
        r"framework|model|system|contribution)|this (?:paper|work|study) (?:proposes|introduces|presents)|"
        r"novel|first to)\b", re.I),
    "evaluation": re.compile(
        r"\b(?:evaluat\w*|experiment\w*|benchmark\w*|dataset\w*|accuracy|outperform\w*|baseline\w*|"
        r"metric\w*|score\w*|results? (?:show|indicate)|we (?:test|measure|compare)|table \d|"
        r"participants|survey(?:ed)? \d|sample)\b", re.I),
    "background": re.compile(
        r"\b(?:prior work|previous(?:ly)?|existing|traditional(?:ly)?|conventional|widely used|"
        r"has been|have been|et al\.?|literature|recent(?:ly)? (?:work|studies|advances)|commonly|"
        r"well[- ]known|in the past)\b", re.I),
    "method": re.compile(
        r"\b(?:consists? of|composed of|module|component|step|stage|algorithm|architecture|pipeline|"
        r"we (?:use|apply|train|compute|define|construct|encode|model)|is computed|is defined|"
        r"objective|loss|layer|procedure|mechanism)\b", re.I),
}
_DEFINITION = r"(?:is|are|refers to|denotes|means|describes|represents|is defined as)"
_ACRONYM_DEF = re.compile(
    r"\b([A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){1,5})\s*\(([A-Z][A-Za-z0-9\-]{1,9}?)s?\)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[])")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-']*|\d[\w\-.]*|[^\sA-Za-z0-9]")
_ROLE_ORDER = ["background", "concept", "method", "contribution", "evaluation"]
ROLE_LABEL = {
    "background": "Background you need",
    "concept": "Core idea",
    "method": "How it works",
    "contribution": "What's new",
    "evaluation": "How it's tested",
}


# ----------------------------------------------------------------- text
def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)          # PDF line-break hyphenation
    text = re.sub(r"\[\d+(?:[,–\-]\s*\d+)*\]", " ", text)  # [12], [3-5]
    text = re.sub(r"https?://\S+|\S+@\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _stem(word: str) -> str:
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _key(phrase: str) -> str:
    return " ".join(_stem(w) for w in re.findall(r"[A-Za-z0-9]+", phrase))


def _trim(words: list[str]) -> list[str]:
    """Drop generic/verb words at the edges: "proposed method" -> "" and
    "outperforming parametric seq2seq" -> "parametric seq2seq". A trailing
    generic noun after a content word is kept: "language models" and
    "knowledge-intensive tasks" are real terms."""
    while words and (words[0].lower() in _GENERIC or words[0].lower() in _STOP
                     or (len(words) > 1 and _verb_form(words[0]) in ("ing", "base", "s"))):
        words = words[1:]
    while words and (words[-1].lower() in _STOP or (
            words[-1].lower() in _GENERIC
            and (len(words) == 1 or words[-1].lower() not in _KEEP_TRAILING))):
        words = words[:-1]
    return words


def _phrases(sentence: str) -> list[str]:
    """RAKE-style candidates: runs of content words between stop words."""
    out: list[str] = []
    run: list[str] = []

    def flush():
        nonlocal run
        words = _trim(run)
        run = []
        if not words:
            return
        if len(words) <= 4:
            out.append(" ".join(words))
        elif len(words) <= 6:                       # long runs: keep 2-3 word windows
            for n in (3, 2):
                for i in range(len(words) - n + 1):
                    w = _trim(words[i:i + n])
                    if len(w) >= 2:
                        out.append(" ".join(w))

    for tok in _TOKEN.findall(sentence):
        low = tok.lower()
        if not tok[0].isalpha() or low in _STOP or len(tok) == 1:
            flush()
            continue
        form = _verb_form(tok)
        if form in ("base", "s") or (form == "ed" and run) or (low.endswith("ly") and len(low) > 4):
            flush()                       # a verb ends the noun phrase
            continue
        run.append(tok)
    flush()
    return out


def _acceptable(phrase: str, count: int, in_front: bool) -> bool:
    words = phrase.split()
    if not words or len(phrase) < 3 or len(phrase) > 48:
        return False
    if all(w.lower() in _GENERIC for w in words):
        return False
    if any(len(w) > 25 for w in words):
        return False
    if len(words) == 1:
        w = words[0]
        acronym = w.isupper() and 2 <= len(w) <= 8
        if not acronym and (len(w) < 4 or w.lower().endswith(("ly", "ed"))
                            or w.lower() in _AMBIGUOUS or _verb_form(w)
                            # lone adjectives ("cognitive", "parametric") are not topics
                            or w.lower().endswith(("ive", "ic", "ous", "ful", "less", "able", "ible"))):
            return False
        return count >= 3 or (in_front and count >= 2)
    return count >= 2 or in_front


# --------------------------------------------------------------- mapper
class ConceptMapper:
    def __init__(self, pipeline):
        self.pipe = pipeline

    def _passages(self, arxiv_id: str) -> list[tuple[str, str, float]]:
        """(section, text, relative position 0..1) for the paper's full text."""
        items = self.pipe.chunk_store.items_where(lambda m: m.get("arxiv_id") == arxiv_id)
        items.sort(key=lambda it: it[1].get("chunk_index", 0))
        n = max(1, len(items) - 1)
        return [(m.get("section", "body"), _clean(m.get("text", "")), i / n)
                for i, (_, m) in enumerate(items)]

    def build(self, arxiv_id: str, explain: bool = False) -> dict:
        cache_key = VERSION + ("-llm" if explain else "")
        cached = _cache_get(arxiv_id, cache_key)
        if cached:
            return cached

        with db_session() as s:
            p = s.execute(select(Paper).where(Paper.arxiv_id == arxiv_id)).scalar_one_or_none()
            if p is None:
                raise ValueError(f"unknown paper {arxiv_id}")
            paper = p.to_dict()
            corpus = [a.lower() for (a,) in s.execute(
                select(Paper.abstract).where(Paper.arxiv_id != arxiv_id)).all() if a]

        title, abstract = _clean(paper["title"]), _clean(paper["abstract"])
        passages = self._passages(arxiv_id)
        full_text = bool(passages)
        sentences: list[dict] = [{"text": title, "section": "title", "pos": 0.0}]
        sentences += [{"text": s, "section": "abstract", "pos": 0.0} for s in _SENT_SPLIT.split(abstract) if s]
        for section, text, pos in passages:
            for s in _SENT_SPLIT.split(text):
                if 25 <= len(s) <= 600:
                    sentences.append({"text": s, "section": section, "pos": pos})

        # acronyms -> long forms
        acronyms: dict[str, str] = {}
        for sent in sentences[:400]:
            for m in _ACRONYM_DEF.finditer(sent["text"]):
                words, acr = m.group(1).split(), m.group(2)
                initials = [c.lower() for c in acr if c.isupper()]
                # take the shortest suffix of words whose (hyphen-split)
                # initials spell the acronym
                for start in range(len(words) - 1, -1, -1):
                    parts = [x for w in words[start:] for x in w.split("-") if x]
                    if len(parts) >= len(initials):
                        if [x[0].lower() for x in parts] == initials:
                            acronyms[acr.lower()] = " ".join(words[start:])
                        break

        # candidate phrases with counts, surface forms, places
        counts: Counter = Counter()
        surface: dict[str, Counter] = defaultdict(Counter)
        where: dict[str, list[int]] = defaultdict(list)
        front: set[str] = set()
        for idx, sent in enumerate(sentences):
            seen_here = set()
            for ph in _phrases(sent["text"]):
                low = ph.lower()
                if low in acronyms:                      # "RAG" counts for its long form
                    ph = acronyms[low]
                k = _key(ph)
                if not k or k in seen_here:
                    continue
                seen_here.add(k)
                counts[k] += 1
                surface[k][ph] += 1
                where[k].append(idx)
                if sent["section"] in ("title", "abstract"):
                    front.add(k)
        # known entities from the gazetteers count as candidates too
        known_type: dict[str, str] = {}
        for kind, table in (("method", _METHODS), ("dataset", _DATASETS),
                            ("metric", _METRICS), ("task", _TASKS)):
            for display in table:
                k = _key(display)
                if k in counts:
                    known_type[k] = kind
        acr_of = {_key(long): acr.upper() for acr, long in acronyms.items()}

        cands = [k for k, c in counts.items() if _acceptable(_display(surface[k]), c, k in front)]
        cands.sort(key=lambda k: (k in front, counts[k]), reverse=True)
        cands = cands[:140]
        if not cands:
            return {"paper": _paper_head(paper), "topics": [], "learning_path": [],
                    "source": "full text" if full_text else "abstract only",
                    "notice": "Not enough text to build a concept map for this paper."}

        names = [_display(surface[k]) for k in cands]
        emb = self.pipe.embedder
        vecs = emb.encode(names)
        doc_vec = emb.encode_one(f"{title}. {abstract}")
        centrality = np.clip(vecs @ doc_vec, 0, 1)
        maxc = max(counts[k] for k in cands)
        title_keys = {_key(p) for p in _phrases(title)} | {_key(acronyms.get(p.lower(), p)) for p in _phrases(title)}
        n_corpus = max(1, len(corpus))

        scored = []
        for i, k in enumerate(cands):
            idxs = where[k]
            parts = {round(sentences[j]["pos"] * 4) for j in idxs if sentences[j]["section"] not in ("title", "abstract")}
            spread = len(parts) / 5 if full_text else min(1.0, len(idxs) / 3)
            prominence = 1.0 if (k in title_keys or any(k in tk or tk in k for tk in title_keys if len(tk) > 3)) \
                else (0.6 if k in front else 0.0)
            phrase_low = names[i].lower()
            df = sum(1 for a in corpus if phrase_low in a)
            specificity = 1 - min(1.0, df / max(5, 0.15 * n_corpus))
            freq = math.log1p(counts[k]) / math.log1p(maxc)
            imp = (0.35 * float(centrality[i]) + 0.25 * freq + 0.15 * spread
                   + 0.15 * prominence + 0.10 * specificity)
            scored.append({"key": k, "i": i, "name": names[i], "importance": imp})
        scored.sort(key=lambda c: c["importance"], reverse=True)

        sim = vecs @ vecs.T

        def expand(key: str) -> set[str]:
            """Words of a phrase, with an acronym replaced by its long form,
            so "RAG models" relates to "retrieval-augmented generation"."""
            words = set()
            for w in key.split():
                long = acronyms.get(w)
                words |= set(_key(long).split()) if long else {w}
            return words

        def related(a: dict, b: dict) -> float:
            s = float(sim[a["i"], b["i"]])
            wa, wb = expand(a["key"]), expand(b["key"])
            if wa <= wb or wb <= wa:
                s = max(s, 0.9 if (wa == wb) else 0.7)
            return s

        majors: list[dict] = []
        for c in scored:
            if len(majors) == MAX_MAJOR:
                break
            if any(related(c, m) > 0.62 for m in majors):
                continue
            majors.append(c)
        chosen = {m["key"] for m in majors}
        subs: dict[str, list[dict]] = {m["key"]: [] for m in majors}
        for c in scored[:60]:
            if c["key"] in chosen:
                continue
            best, best_s = None, 0.0
            for m in majors:
                s = related(c, m)
                if s > best_s:
                    best, best_s = m, s
            if best is None or best_s < 0.32 or best_s > 0.88 or len(subs[best["key"]]) >= MAX_SUB:
                continue
            if any(related(c, o) > 0.8 for o in subs[best["key"]]):
                continue                                   # near-duplicate subtopic
            subs[best["key"]].append(c)
            chosen.add(c["key"])

        intro = heuristic_extract({"arxiv_id": arxiv_id, "title": title, "abstract": abstract})
        introduced = {_key(x) for x in intro.introduces}
        top = majors[0]["importance"] if majors else 1.0

        def describe(c: dict, level: int, rank: int = 99) -> dict:
            k = c["key"]
            tier = ("Core" if rank < 2 else "Major") if level == 1 else "Supporting"
            idxs = where[k]
            role = _role(k, idxs, sentences, known_type.get(k), introduced, full_text)
            sections = Counter(sentences[j]["section"] for j in idxs
                               if sentences[j]["section"] not in ("title",))
            name = c["name"]
            if k in acr_of and acr_of[k].lower() not in name.lower():
                name = f"{name} ({acr_of[k]})"
            return {
                "id": k, "name": _pretty(name), "level": level, "role": role,
                "role_label": ROLE_LABEL[role],
                "importance": round(c["importance"], 3),
                "tier": tier,
                "mentions": counts[k],
                "sections": [s for s, _ in sections.most_common(4)],
                "evidence": _evidence(c["name"], acr_of.get(k), idxs, sentences),
                "kind": known_type.get(k),
            }

        topics = []
        for rank, m in enumerate(majors):
            node = describe(m, 1, rank)
            node["subtopics"] = [describe(s, 2) for s in subs[m["key"]]]
            topics.append(node)

        result = {
            "paper": _paper_head(paper),
            "topics": topics,
            "learning_path": _learning_path(topics),
            "source": "full text" if full_text else "abstract only",
            "notice": None if full_text else
                      "Built from the abstract only: the full text of this paper is not indexed.",
        }
        if explain:
            _add_explanations(result, title, abstract)
        _cache_put(arxiv_id, cache_key, result)
        return result


# ----------------------------------------------------------- helpers
def _display(forms: Counter) -> str:
    # prefer the most common form; break ties toward forms with capitals (acronyms, names)
    return max(forms.items(), key=lambda kv: (kv[1], sum(ch.isupper() for ch in kv[0])))[0]


def _pretty(name: str) -> str:
    return name[0].upper() + name[1:] if name and name[0].islower() else name


def _paper_head(paper: dict) -> dict:
    from backend.app.services.citations import _cached

    tldr = (_cached([paper["arxiv_id"]]).get(paper["arxiv_id"]) or {}).get("tldr") or ""
    first = _SENT_SPLIT.split(_clean(paper.get("abstract", "")))
    return {
        "arxiv_id": paper["arxiv_id"], "title": paper["title"],
        "authors": paper.get("authors", ""), "published": paper.get("published", ""),
        "summary": tldr or (first[0] if first else ""),
    }


def _role(key: str, idxs: list[int], sentences: list[dict], kind: str | None,
          introduced: set[str], full_text: bool) -> str:
    if key in introduced or any(key in i or i in key for i in introduced if len(i) > 3):
        return "contribution"
    if kind in ("dataset", "metric"):
        return "evaluation"
    votes: Counter = Counter()
    positions = []
    for j in idxs:
        s = sentences[j]
        for role, pat in _ROLE_CUES.items():
            if pat.search(s["text"]):
                votes[role] += 1
        if s["section"] not in ("title", "abstract"):
            positions.append(s["pos"])
            sec = s["section"]
            if sec in ("introduction", "related_work"):
                votes["background"] += 0.5
            elif sec == "methodology":
                votes["method"] += 0.5
            elif sec in ("experiments", "results"):
                votes["evaluation"] += 0.5
    if full_text and positions:
        avg = sum(positions) / len(positions)
        if avg < 0.25:
            votes["background"] += 0.6
        elif avg > 0.65:
            votes["evaluation"] += 0.6
        else:
            votes["method"] += 0.4
    n = len(idxs) or 1
    if votes["contribution"] / n >= 0.25:
        return "contribution"
    if not votes:
        return "concept"
    role, v = votes.most_common(1)[0]
    if v / n < 0.2:
        return "concept"
    if kind == "method" and role == "contribution":
        return "method"
    return role


def _evidence(name: str, acronym: str | None, idxs: list[int], sentences: list[dict]) -> str:
    """The sentence that best explains the topic, preferring definitions."""
    names = [name] + ([acronym] if acronym else [])
    pats = [re.compile(re.escape(n).replace(r"\ ", r"[\s\-]+"), re.I) for n in names]
    best, best_score = "", -1.0
    for j in idxs[:60]:
        s = sentences[j]
        text = re.sub(r"^\d+(?:\.\d+)*\s+", "", s["text"])     # "3.4 Fact Verification ..."
        if s["section"] == "title" or not any(p.search(text) for p in pats):
            continue
        score = 0.0
        for n in names:
            if re.search(re.escape(n) + r"[^.]{0,20}\b" + _DEFINITION + r"\b", text, re.I):
                score += 2.0
            if re.search(r"(?:called|termed|known as|refer to)[^.]{0,10}" + re.escape(n), text, re.I):
                score += 1.5
        if s["section"] in ("abstract", "introduction"):
            score += 0.5
        length = len(text)
        score += 0.5 if 60 <= length <= 260 else -0.5
        score -= 0.1 * len(re.findall(r"\d", text)) / 5
        if score > best_score:
            best, best_score = text, score
    if len(best) > 320:
        best = best[:317].rsplit(" ", 1)[0] + "…"
    return best


def _learning_path(topics: list[dict]) -> list[dict]:
    order = {r: i for i, r in enumerate(_ROLE_ORDER)}
    path = sorted(topics, key=lambda t: (order[t["role"]], -t["importance"]))
    return [{"id": t["id"], "name": t["name"], "role": t["role"], "role_label": t["role_label"]}
            for t in path]


_EXPLAIN_PROMPT = """Paper: "{title}"
Abstract: {abstract}

For each topic below, write ONE plain-language sentence (max 30 words) that explains what it is
and why it matters for this paper. Use only the abstract and general knowledge of the field.

Topics:
{topics}

Return ONLY a JSON object mapping each topic name exactly as written to its sentence."""


def _add_explanations(result: dict, title: str, abstract: str) -> None:
    from backend.app.services.llm.generator import get_llm

    llm = get_llm()
    if not llm.generative:
        result["notice"] = (result.get("notice") or "") + " Plain-language explanations need the LLM (Ollama)."
        return
    names = [t["name"] for t in result["topics"]]
    names += [s["name"] for t in result["topics"] for s in t["subtopics"][:2]]
    names = names[:14]
    prompt = _EXPLAIN_PROMPT.format(title=title, abstract=abstract[:1500],
                                    topics="\n".join(f"- {n}" for n in names))
    try:
        raw = "".join(llm.generate_stream(prompt, "", 700))
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group())
    except Exception as exc:
        logger.info("concept explanations failed: %s", exc)
        return
    lookup = {str(k).strip().lower(): str(v).strip() for k, v in data.items() if isinstance(v, str)}
    for t in result["topics"]:
        t["explanation"] = lookup.get(t["name"].lower(), "")
        for s in t["subtopics"]:
            s["explanation"] = lookup.get(s["name"].lower(), "")


def _cache_get(arxiv_id: str, key: str) -> dict | None:
    try:
        with db_session() as s:
            row = s.execute(select(PaperEntities).where(
                PaperEntities.arxiv_id == arxiv_id, PaperEntities.extractor == key)).scalar_one_or_none()
            return json.loads(row.data) if row else None
    except Exception:
        return None


def _cache_put(arxiv_id: str, key: str, data: dict) -> None:
    try:
        with db_session() as s:
            s.add(PaperEntities(arxiv_id=arxiv_id, extractor=key, data=json.dumps(data)))
    except Exception:
        logger.debug("concept map cache write failed", exc_info=True)
