"""'What should I read next?' — gap-aware reading recommendations.

Plain recommenders return the papers most *similar* to what you read, which
mostly repeats what you already know. This module instead models the
reader's knowledge and looks for the missing connection:

    YOU KNOW:      Paper A -> Dense Retrieval
                   Paper B -> Retrieval-Augmented Generation
    YOU'RE MISSING: Paper C connects Dense Retrieval and RAG
    "Read C because it bridges the concepts introduced in A and B."

Signals per candidate C (given read papers R1..Rn):

* bridge similarity — harmonic mean of cos(C, Ri): high only when C is
  related to *all* read papers, not just one of them;
* concept bridging — how many read papers' concepts (methods, tasks,
  datasets from the knowledge-graph extractor) C's abstract mentions;
* foundation — C introduces a method a read paper uses/builds on;
* novelty — concepts C adds that none of the read papers cover;
* redundancy penalty — near-duplicates of a read paper are demoted;
* diversity — greedy MMR so the list is not five versions of one paper.

Candidates come from the local index (neighbours of each read paper and of
their centroid) plus an optional live arXiv "bridge query" that ANDs the
key concepts of pairs of read papers, i.e. asks arXiv directly for papers
connecting them.
"""
from __future__ import annotations

import itertools
import logging
import re

import numpy as np

from backend.app.core.config import settings
from backend.app.services.knowledge_graph import (
    _ENTITY_TABLES,
    canonical,
    extract_papers,
    load_papers,
    norm_key,
)
from backend.app.services.query_processor import _concept_groups
from backend.app.services.relevance import mentions

logger = logging.getLogger(__name__)

_W_BRIDGE_SIM = 0.45
_W_BRIDGE_COV = 0.35
_W_NOVELTY = 0.10
_W_FOUNDATION = 0.15
_REDUNDANT_SIM = 0.92
_REDUNDANT_PENALTY = 0.40
_DIVERSITY_SIM = 0.88


def _key(concept: str) -> str:
    return norm_key(canonical(concept)[1])


def _aliases(concept: str) -> list[str]:
    kind, display = canonical(concept)
    out = [concept, display]
    if kind:
        out += _ENTITY_TABLES[kind].get(display, [])
    base = re.sub(r"\s*\(.*?\)\s*", " ", concept).strip()
    acr = re.search(r"\(([^)]+)\)", concept)
    out += [base] + ([acr.group(1)] if acr else [])
    return [a for a in dict.fromkeys(out) if len(a) >= 3 or a.isupper()]


def _mentions_concept(concept: str, text: str) -> bool:
    return any(mentions(a, text) for a in _aliases(concept))


_GENERIC_TITLE_WORDS = {
    "analysis", "comparative", "study", "studies", "survey", "review", "overview",
    "towards", "toward", "novel", "new", "efficient", "improving", "improved",
    "case", "decision", "recommendation", "recommendations", "framework",
    "validation", "reproduces", "reference", "exact", "characterization",
    "generalizations", "simplified", "critical", "benchmarking", "exploring",
    "understanding", "revisiting", "rethinking", "learning", "public", "large",
    "scale", "simple", "effective", "via", "based", "two", "users", "approach",
    "datasets", "dataset", "results", "methods",
}


def _title_concepts(title: str) -> list[str]:
    """Fallback concepts from a title: 'Dense Passage Retrieval for
    Open-Domain QA' -> ['dense passage retrieval', 'open-domain qa'].
    Generic words ('comparative analysis', 'a survey') are dropped."""
    out = []
    for group in _concept_groups(title.replace(":", ",")):
        words = [w for w in group if w not in _GENERIC_TITLE_WORDS and not w.isdigit()]
        words = words[-3:] if len(words) > 4 else words   # head noun phrase
        if 1 <= len(words) <= 4 and not (len(words) == 1 and len(words[0]) < 4):
            out.append(" ".join(words))
    out.sort(key=lambda c: -len(c.split()))       # multi-word = more specific
    return list(dict.fromkeys(out))[:3]


def _paper_concepts(paper: dict, ex) -> list[str]:
    concepts = ex.introduces + ex.tasks + ex.extends + ex.uses + ex.introduces_datasets
    concepts = [c for c in concepts if len(norm_key(c)) > 2]
    if len(concepts) < 2:
        concepts += _title_concepts(paper["title"])
    seen: set[str] = set()
    out = []
    for c in concepts:
        k = norm_key(canonical(c)[1])
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out[:8]


def _search_concepts(paper: dict, ex) -> list[str]:
    """The 2 most specific, searchable concepts of a paper (for arXiv)."""
    pool = (
        [c for c in ex.introduces if len(c) >= 4]
        + ex.tasks + ex.extends + _title_concepts(paper["title"]) + ex.uses
    )
    out: list[str] = []
    for c in pool:
        c = re.sub(r"\s*\(.*?\)\s*", " ", c).strip()
        if c and norm_key(c) not in {norm_key(x) for x in out}:
            out.append(c)
        if len(out) == 2:
            break
    return out


def _arxiv_term(c: str) -> str:
    c = re.sub(r'["()]', " ", c).strip()
    return f'(ti:"{c}" OR abs:"{c}")'


def bridge_query(read: list[dict], ext: dict) -> str:
    """arXiv query asking for papers that connect pairs of read papers."""
    per_paper = [_search_concepts(p, ext[p["arxiv_id"]]) for p in read]
    per_paper = [c for c in per_paper if c]
    if not per_paper:
        return ""

    def any_of(cs: list[str]) -> str:
        return "(" + " OR ".join(_arxiv_term(c) for c in cs) + ")"

    if len(per_paper) == 1:
        return any_of(per_paper[0])
    clauses = [
        f"({any_of(a)} AND {any_of(b)})"
        for a, b in itertools.islice(itertools.combinations(per_paper, 2), 4)
    ]
    return " OR ".join(clauses)


class ReadingAdvisor:
    def __init__(self, pipeline):
        self.pipe = pipeline

    # ----------------------------------------------------------- vectors
    def _vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        wanted = set(ids)
        out: dict[str, np.ndarray] = {}
        for vid, meta in self.pipe.paper_store.all_meta().items():
            pid = meta.get("arxiv_id")
            if pid in wanted and pid not in out:
                vec = self.pipe.paper_store.get_vector(vid)
                if vec is not None:
                    out[pid] = vec
        return out

    def _neighbours(self, vec: np.ndarray, k: int, exclude: set[str]) -> list[str]:
        hits = self.pipe.paper_store.search(
            vec, top_k=k, filter_fn=lambda m: m.get("arxiv_id") not in exclude
        )
        return [m["arxiv_id"] for _, _, m in hits]

    # -------------------------------------------------------------- main
    def recommend(
        self,
        read_ids: list[str],
        top_k: int | None = None,
        fetch_arxiv: bool = True,
        use_llm: bool = False,
    ) -> dict:
        from backend.app.services.arxiv_client import fetch_by_ids, search_arxiv, wait_for_slot

        top_k = top_k or settings.read_next_top_k
        read_ids = list(dict.fromkeys(i.strip() for i in read_ids if i.strip()))
        notices: list[str] = []

        # make sure every read paper is in the corpus (pasted ids)
        read = load_papers(read_ids)
        missing = [i for i in read_ids if i not in {p["arxiv_id"] for p in read}]
        if missing and fetch_arxiv:
            metas, err = fetch_by_ids(missing)
            if metas:
                self.pipe.ingest_metadata(metas)
                read = load_papers(read_ids)
            if err:
                notices.append(f"Could not fetch {', '.join(missing)} from arXiv ({err}).")
        unknown = [i for i in read_ids if i not in {p["arxiv_id"] for p in read}]
        if unknown:
            notices.append(f"Unknown paper id(s) skipped: {', '.join(unknown)}.")
        if not read:
            return {"you_know": [], "you_are_missing": [], "recommendations": [],
                    "bridge_query": "", "notice": " ".join(notices) or "No known papers."}

        read_set = {p["arxiv_id"] for p in read}
        ext = extract_papers(read)
        known = {p["arxiv_id"]: _paper_concepts(p, ext[p["arxiv_id"]]) for p in read}
        known_keys = {_key(c) for cs in known.values() for c in cs}
        # concepts distinctive to each read paper come first, so a bridge
        # names what is *different* about A and B, not what they share
        counts: dict[str, int] = {}
        for cs in known.values():
            for c in {_key(x) for x in cs}:
                counts[c] = counts.get(c, 0) + 1
        bridge_order = {
            rid: sorted(cs, key=lambda c: counts[_key(c)] > 1) for rid, cs in known.items()
        }

        # 1. live bridge query: papers connecting the read papers' concepts
        bq = bridge_query(read, ext) if fetch_arxiv else ""
        live_ids: list[str] = []
        if bq and wait_for_slot():
            metas, err = search_arxiv(bq)
            if metas:
                self.pipe.ingest_metadata(metas)
                live_ids = [m.arxiv_id for m in metas if m.arxiv_id not in read_set]
            elif err:
                notices.append(f"Live arXiv bridge search skipped ({err}); using the local corpus.")

        # 2. local candidates: neighbours of each read paper and of the centroid
        read_vecs = self._vectors(list(read_set))
        pool: list[str] = list(live_ids)
        if read_vecs:
            k = settings.read_next_pool
            for vec in read_vecs.values():
                pool += self._neighbours(vec, k, read_set)
            centroid = np.mean(list(read_vecs.values()), axis=0)
            centroid /= np.linalg.norm(centroid) or 1.0
            pool += self._neighbours(centroid, k, read_set)
        pool = [pid for pid in dict.fromkeys(pool) if pid not in read_set]
        if not pool:
            return {"you_know": self._you_know(read, known), "you_are_missing": [],
                    "recommendations": [], "bridge_query": bq,
                    "notice": " ".join(notices + ["No candidate papers found yet. Run a few searches to grow the corpus."])}

        cands = load_papers(pool)
        cand_vecs = self._vectors([c["arxiv_id"] for c in cands])
        cand_ext = extract_papers(cands)
        read_texts = {p["arxiv_id"]: f"{p['title']}. {p.get('abstract', '')}" for p in read}
        title = {p["arxiv_id"]: p["title"] for p in read}
        n = len(read)
        live_set = set(live_ids)

        scored = []
        for c in cands:
            cid = c["arxiv_id"]
            vec = cand_vecs.get(cid)
            if vec is None:
                continue
            sims = {rid: float(np.dot(vec, rv)) for rid, rv in read_vecs.items()}
            if not sims:
                continue
            clipped = [max(s, 0.01) for s in sims.values()]
            bridge_sim = len(clipped) / sum(1 / s for s in clipped)
            text = f"{c['title']}. {c.get('abstract', '')}"

            # which read papers' concepts does this candidate talk about?
            bridges = []
            for rid, concepts in bridge_order.items():
                hit = next((cc for cc in concepts if _mentions_concept(cc, text)), None)
                if hit:
                    bridges.append({"concept": hit, "from_id": rid, "from_title": title[rid]})
            bridge_cov = len(bridges) / n

            cx = cand_ext[cid]
            new_concepts = [
                cc for cc in _paper_concepts(c, cx)
                if _key(cc) not in known_keys
            ][:4]
            foundations = []
            for m in cx.introduces:
                users = [rid for rid, rt in read_texts.items() if _mentions_concept(m, rt)]
                if users:
                    foundations.append({"method": m, "used_by": users})

            redundant = max(sims.values()) >= _REDUNDANT_SIM
            score = (
                _W_BRIDGE_SIM * bridge_sim
                + _W_BRIDGE_COV * bridge_cov
                + _W_NOVELTY * min(1.0, len(new_concepts) / 3)
                + _W_FOUNDATION * (1.0 if foundations else 0.0)
                + (0.03 if cid in live_set else 0.0)
                - (_REDUNDANT_PENALTY if redundant else 0.0)
            )
            scored.append({
                "paper": c, "vec": vec, "score": score, "sims": sims,
                "bridges": bridges, "new_concepts": new_concepts,
                "foundations": foundations, "redundant": redundant,
            })

        picked = self._diversify(sorted(scored, key=lambda s: s["score"], reverse=True), top_k)
        recs = [self._explain(s, n, title) for s in picked]
        if use_llm:
            self._llm_headlines(recs, read)

        missing_map: dict[str, dict] = {}
        for r in recs:
            for cc in r["new_concepts"]:
                key = norm_key(canonical(cc)[1])
                entry = missing_map.setdefault(key, {"concept": canonical(cc)[1], "papers": []})
                entry["papers"].append(r["arxiv_id"])
        return {
            "you_know": self._you_know(read, known),
            "you_are_missing": sorted(missing_map.values(), key=lambda e: len(e["papers"]), reverse=True)[:8],
            "recommendations": recs,
            "bridge_query": bq,
            "notice": " ".join(notices) or None,
        }

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _you_know(read: list[dict], known: dict[str, list[str]]) -> list[dict]:
        return [
            {"arxiv_id": p["arxiv_id"], "title": p["title"],
             "concepts": [canonical(c)[1] for c in known[p["arxiv_id"]]][:6]}
            for p in read
        ]

    @staticmethod
    def _diversify(ranked: list[dict], k: int) -> list[dict]:
        picked: list[dict] = []
        for cand in ranked:
            if any(float(np.dot(cand["vec"], p["vec"])) >= _DIVERSITY_SIM for p in picked):
                continue
            picked.append(cand)
            if len(picked) == k:
                break
        return picked

    @staticmethod
    def _explain(s: dict, n: int, title: dict[str, str]) -> dict:
        p = s["paper"]

        def short(t: str) -> str:
            return t if len(t) <= 60 else t[:57].rstrip() + "…"

        reasons: list[str] = []
        distinct = {b["from_id"]: b for b in s["bridges"]}
        if s["foundations"]:
            f = s["foundations"][0]
            kind = "Foundation"
            users = " and ".join(f"“{short(title[u])}”" for u in f["used_by"][:2])
            headline = f"Read this first: it introduces {f['method']}, which {users} builds on."
            reasons.append(headline)
        elif n >= 2 and len(distinct) >= 2:
            kind = "Bridge"
            a, b = list(distinct.values())[:2]
            ca, cb = canonical(a["concept"])[1], canonical(b["concept"])[1]
            if _key(ca) != _key(cb):
                headline = (
                    f"Read this because it bridges {ca} (from “{short(a['from_title'])}”) "
                    f"and {cb} (from “{short(b['from_title'])}”)."
                )
            else:
                headline = (
                    f"Read this because it connects “{short(a['from_title'])}” and "
                    f"“{short(b['from_title'])}” through their shared focus on {ca}."
                )
            reasons.append(headline)
        else:
            kind = "Next step"
            if distinct:
                b = next(iter(distinct.values()))
                headline = (
                    f"Builds on {canonical(b['concept'])[1]} from “{short(b['from_title'])}”"
                    + (f" and takes it towards {s['new_concepts'][0]}." if s["new_concepts"] else ".")
                )
            else:
                rid, sim = max(s["sims"].items(), key=lambda kv: kv[1])
                headline = f"Closely related to “{short(title[rid])}” ({sim:.0%} content similarity)."
            reasons.append(headline)
        if s["new_concepts"]:
            reasons.append("Adds concepts you have not covered yet: " + ", ".join(s["new_concepts"][:3]) + ".")
        if s["redundant"]:
            reasons.append("Note: very similar to a paper you already read.")
        return {
            "arxiv_id": p["arxiv_id"], "title": p["title"], "abstract": p.get("abstract", ""),
            "published": p.get("published", ""), "categories": p.get("categories", ""),
            "score": round(s["score"], 4), "kind": kind, "headline": headline, "reasons": reasons,
            "bridges": s["bridges"], "new_concepts": s["new_concepts"],
            "foundations": s["foundations"],
            "similarity": {rid: round(v, 3) for rid, v in s["sims"].items()},
        }

    @staticmethod
    def _llm_headlines(recs: list[dict], read: list[dict]) -> None:
        """Optionally rewrite the top headlines in natural language."""
        from backend.app.services.llm.generator import get_llm

        llm = get_llm()
        if not llm.generative:
            return
        read_list = "\n".join(f"- {p['title']}" for p in read)
        for r in recs[:3]:
            prompt = (
                f"A researcher has read:\n{read_list}\n\nCandidate paper: {r['title']}\n"
                f"Abstract: {r['abstract'][:1200]}\n\nFacts: {' '.join(r['reasons'])}\n\n"
                "In ONE sentence (max 35 words), tell the researcher why they should read the "
                "candidate next, naming the specific concepts it connects. Do not invent facts."
            )
            try:
                text = llm.generate(prompt).strip().strip('"')
            except Exception:
                return
            if 20 <= len(text) <= 400:
                r["llm_headline"] = text
