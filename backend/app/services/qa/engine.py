"""Question answering over a paper's full text: a grounded research chatbot.

Single-paper chat: the paper's abstract plus the passages that best answer
the question, picked by exact search over *that paper's* chunks, then a
cross-encoder; follow-ups are resolved against the conversation.

Multi-paper chat: up to ``qa_multi_single_pass_max`` papers are answered
in ONE pass over per-paper evidence (abstract + best passages), which keeps
comparisons coherent and is much faster on CPU than map-reduce. Larger sets
fall back to map-reduce (map: per-paper evidence, reduce: synthesis).

Why it is built this way (failures it fixes):
* Generic questions ("what's the key difference between these papers?")
  contain no topical words, so relevance scoring rated every passage ~0
  and the old engine answered "no evidence". Question *intent* now picks
  the sections to read (abstract, intro, method, results, conclusion).
* Paper-restricted retrieval used post-filtered ANN search over the whole
  chunk index; with many papers indexed only ~1 passage per paper
  survived. Retrieval is now exact over the paper's own chunks.
* The abstract is always included, so every answer has a grounded summary.

Every generation path streams: ``*_stream`` methods yield events
(status / sources / token / final) for the chat UI; the plain methods
consume the stream and return a ``QAResult``.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from backend.app.core.config import settings
from backend.app.services.embeddings import BaseEmbedder
from backend.app.services.llm.generator import safe_generate, stream_generate
from backend.app.services.qa import prompts
from backend.app.services.qa.intent import Intent, contextualize, detect_intent
from backend.app.services.reranker import Reranker
from backend.app.services.vector_store import FaissStore

logger = logging.getLogger(__name__)

_SENT_END = re.compile(r"(?<=[.!?])\s+")
_NO_INFO_ANSWER = re.compile(
    r"(does not contain|not contain enough|does not discuss|do not discuss|"
    r"no relevant information|not (?:discussed|mentioned|covered))",
    re.I,
)


@dataclass
class SourceRef:
    n: int
    arxiv_id: str
    title: str
    section: str
    snippet: str


@dataclass
class QAResult:
    answer: str
    sources: list[SourceRef] = field(default_factory=list)
    mode: str = "single"
    papers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _trim(text: str, limit: int) -> str:
    """Cut text to ``limit`` chars at a sentence boundary when possible."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return (cut[: end + 1] if end > limit * 0.5 else cut.rstrip() + "…")


def _short_title(title: str, n: int = 48) -> str:
    head = title.split(":")[0].strip() if ":" in title and len(title.split(":")[0]) >= 4 else title
    return head if len(head) <= n else head[: n - 1].rstrip() + "…"


def _byline(meta: dict) -> str:
    authors = [a.strip() for a in (meta.get("authors") or "").split(";") if a.strip()]
    who = (authors[0].split()[-1] + (" et al." if len(authors) > 1 else "")) if authors else ""
    year = (meta.get("published") or "")[:4]
    bits = ", ".join(x for x in (who, year) if x)
    return f" ({bits})" if bits else ""


class QAEngine:
    def __init__(
        self, chunk_store: FaissStore, embedder: BaseEmbedder, reranker: Reranker
    ):
        self.chunk_store = chunk_store
        self.embedder = embedder
        self.reranker = reranker

    # --------------------------------------------------------- evidence
    def _paper_chunks(self, paper_id: str) -> list[tuple[int, dict]]:
        return self.chunk_store.items_where(lambda m: m.get("arxiv_id") == paper_id)

    def _has_indexed_chunks(self, paper_id: str) -> bool:
        return bool(self._paper_chunks(paper_id))

    def _select_chunks(
        self,
        paper_id: str,
        question: str,
        intent: Intent,
        budget_chars: int,
        abstract: str = "",
        per_chunk_chars: int = 900,
    ) -> tuple[list[dict], float]:
        """Best passages of ONE paper for a question, within a char budget.

        1. exact dense similarity over all of the paper's chunks;
        2. cross-encoder rerank of the top candidates;
        3. confident passages first; for generic questions (or when nothing
           is confidently relevant) fill from the intent's preferred
           sections, so "what's the difference?" reads abstract/method/
           results/conclusion instead of returning nothing.
        Returns (chunks, best cross-encoder score).
        """
        items = self._paper_chunks(paper_id)
        if abstract:
            # the metadata abstract is always sent; the PDF's own abstract
            # passage would only duplicate it (and cost prompt time)
            items = [it for it in items if it[1].get("section") != "abstract"]
        if not items or budget_chars <= 0:
            return [], 0.0
        query = f"{question} {intent.hint}".strip() if intent.generic else question
        vecs = self.chunk_store.vectors([vid for vid, _ in items])
        dense = vecs @ self.embedder.encode_one(query)
        order = np.argsort(-dense)
        cands = []
        for i in order[:16]:
            m = items[int(i)][1]
            cands.append({
                "arxiv_id": paper_id, "title": m.get("title", ""),
                "section": m.get("section", "body"), "text": m.get("text", ""),
                "chunk_index": m.get("chunk_index", 0), "fused_score": float(dense[int(i)]),
            })
        ranked = self.reranker.rerank(question, cands, top_k=len(cands))
        top = ranked[0].get("rerank_score", 0.0) if ranked else 0.0
        threshold = settings.qa_relevance_threshold

        picked: list[dict] = []
        used = 0
        abstract_words = set(re.findall(r"[a-z0-9]+", abstract.lower()))

        def add(c: dict) -> bool:
            nonlocal used
            text = _trim(c["text"], per_chunk_chars)
            if not text or any(p["chunk_index"] == c["chunk_index"] for p in picked):
                return False
            words = set(re.findall(r"[a-z0-9]+", text.lower()))
            # skip passages that merely repeat the abstract (already included)
            if abstract_words and words and len(words & abstract_words) / len(words) > 0.8:
                return False
            if used + len(text) > budget_chars and picked:
                return False
            picked.append({**c, "text": text})
            used += len(text)
            return True

        for c in ranked:
            if c.get("rerank_score", 0.0) >= threshold:
                add(c)
        if intent.generic or top < threshold:
            # one best passage per preferred section, in section order
            by_section: dict[str, dict] = {}
            for i in order:
                m = items[int(i)][1]
                sec = m.get("section", "body")
                if sec not in by_section:
                    by_section[sec] = {
                        "arxiv_id": paper_id, "title": m.get("title", ""), "section": sec,
                        "text": m.get("text", ""), "chunk_index": m.get("chunk_index", 0),
                        "fused_score": float(dense[int(i)]),
                    }
            for sec in intent.sections:
                if sec in by_section:
                    add(by_section[sec])
            for c in ranked:                      # then best remaining by relevance
                if used >= budget_chars:
                    break
                add(c)
        # present passages in document order: reads more naturally
        picked.sort(key=lambda c: c.get("chunk_index", 0))
        return picked, top

    @staticmethod
    def _history_block(history: list[dict] | None) -> str:
        if not history:
            return ""
        turns = history[-settings.max_conversation_turns:]
        lines = []
        for t in turns:
            who = "User" if t["role"] == "user" else "Assistant"
            content = t["content"]
            if who == "Assistant":                # keep follow-up context short
                content = _trim(re.sub(r"\[\d+\]", "", content), 400)
            lines.append(f"{who}: {content}")
        return "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    @staticmethod
    def _sources(items: list[dict]) -> list[SourceRef]:
        return [
            SourceRef(n=i, arxiv_id=c["arxiv_id"], title=c["title"],
                      section=c["section"], snippet=c["text"][:400])
            for i, c in enumerate(items, start=1)
        ]

    @staticmethod
    def _collect(events: Iterator[dict]) -> QAResult:
        result = QAResult(answer="")
        for ev in events:
            if ev["type"] == "sources":
                result.sources = [SourceRef(**s) for s in ev["sources"]]
            elif ev["type"] == "final":
                result.answer = ev["answer"]
                result.mode = ev.get("mode", result.mode)
                result.papers = ev.get("papers", result.papers)
                result.notes = ev.get("notes", [])
        return result

    # ------------------------------------------------------------- single
    def answer_single(
        self, question: str, arxiv_id: str, title: str = "",
        history: list[dict] | None = None, meta: dict | None = None,
    ) -> QAResult:
        return self._collect(self.answer_single_stream(
            question, arxiv_id, title=title, history=history, meta=meta))

    def answer_single_stream(
        self, question: str, arxiv_id: str, title: str = "",
        history: list[dict] | None = None, meta: dict | None = None,
    ) -> Iterator[dict]:
        meta = dict(meta or {})
        title = title or meta.get("title", "") or arxiv_id
        meta.setdefault("title", title)
        intent = detect_intent(question)
        retrieval_q = contextualize(question, history)
        yield {"type": "status", "text": "Finding the most relevant passages…"}

        abstract = _trim(meta.get("abstract", ""), 1500)
        chunks, _top = self._select_chunks(
            arxiv_id, retrieval_q, intent, settings.qa_context_chars, abstract=abstract)
        items = []
        if abstract:
            items.append({"arxiv_id": arxiv_id, "title": title, "section": "abstract", "text": abstract})
        items += chunks
        if not items:
            answer = (
                "I couldn't read this paper: its full text could not be downloaded "
                "or parsed and it has no abstract. Try another paper, or check its "
                f"status with GET /papers/{arxiv_id}."
            )
            yield {"type": "sources", "sources": []}
            yield {"type": "final", "answer": answer, "mode": "single", "papers": [arxiv_id]}
            return
        sources = self._sources(items)
        yield {"type": "sources", "sources": [s.__dict__ for s in sources]}

        contexts = prompts.format_contexts(
            [{"label": c["section"].capitalize(), "text": c["text"]} for c in items])
        prompt = prompts.CHAT_SINGLE_TEMPLATE.format(
            title=title, byline=_byline(meta), contexts=contexts,
            history=self._history_block(history), question=question,
            instruction=intent.instruction,
        )
        yield {"type": "status", "text": "Writing the answer…"}
        final = ""
        for kind, text in stream_generate(
            prompt, system=prompts.SYSTEM_CHAT, n_sources=len(items),
            require_citations=True, max_tokens=settings.llm_max_new_tokens,
        ):
            if kind == "token":
                yield {"type": "token", "text": text}
            else:
                final = text
        notes = []
        if not chunks:
            notes.append("Answered from the abstract: the full text of this paper is not indexed.")
        yield {"type": "final", "answer": final, "mode": "single",
               "papers": [arxiv_id], "notes": notes}

    # ------------------------------------------------------------- multi
    @staticmethod
    def _is_no_info(summary: str) -> bool:
        """True only when the map output IS the sentinel — a partial answer
        that merely mentions missing information must not be discarded."""
        up = summary.strip().upper()
        return up.startswith("NO RELEVANT INFORMATION") or (
            "NO RELEVANT INFORMATION" in up and len(summary.strip()) < 80
        )

    @staticmethod
    def _empty_multi_answer(
        skipped: dict[str, str],
        titles: dict[str, str],
        paper_status: dict[str, str],
    ) -> str:
        """Actionable diagnostics instead of a blanket 'nothing relevant'."""
        lines = ["I couldn't extract evidence for this question. Per paper:"]
        any_no_match = False
        for pid, reason in skipped.items():
            name = titles.get(pid) or pid
            name = name if len(name) <= 70 else name[:67] + "…"
            if reason == "no_chunks":
                status = paper_status.get(pid, "unknown")
                lines.append(
                    f"- {name} (arXiv:{pid}): no indexed full text — "
                    f"indexing status: {status}. Its PDF may have failed to "
                    "download or parse; try re-selecting it or check "
                    f"GET /papers/{pid}."
                )
            else:
                any_no_match = True
                lines.append(
                    f"- {name} (arXiv:{pid}): indexed, but none of its "
                    "retrieved passages matched the question."
                )
        if any_no_match:
            lines.append(
                "\nTip: short generic questions (e.g. \"compare the results\") "
                "match poorly — name what you want compared, e.g. \"Compare "
                "the retrieval accuracy results and datasets used in these "
                "papers.\""
            )
        return "\n".join(lines)

    def answer_multi(
        self, question: str, paper_ids: list[str],
        titles: dict[str, str] | None = None, history: list[dict] | None = None,
        paper_status: dict[str, str] | None = None, metas: dict[str, dict] | None = None,
    ) -> QAResult:
        return self._collect(self.answer_multi_stream(
            question, paper_ids, titles=titles, history=history,
            paper_status=paper_status, metas=metas))

    def answer_multi_stream(
        self, question: str, paper_ids: list[str],
        titles: dict[str, str] | None = None, history: list[dict] | None = None,
        paper_status: dict[str, str] | None = None, metas: dict[str, dict] | None = None,
    ) -> Iterator[dict]:
        titles = titles or {}
        paper_status = paper_status or {}
        metas = metas or {}
        if len(paper_ids) > settings.qa_multi_single_pass_max:
            yield from self._map_reduce_stream(question, paper_ids, titles, paper_status)
            return

        intent = detect_intent(question)
        retrieval_q = contextualize(question, history)
        yield {"type": "status", "text": f"Reading {len(paper_ids)} papers…"}
        per_paper = max(900, settings.qa_multi_context_chars // max(1, len(paper_ids)))
        items: list[dict] = []
        paper_lines: list[str] = []
        missing: dict[str, str] = {}
        used_pids: list[str] = []
        for k, pid in enumerate(paper_ids, start=1):
            meta = metas.get(pid, {})
            title = titles.get(pid) or meta.get("title") or pid
            abstract = _trim(meta.get("abstract", ""), min(900, per_paper // 2))
            chunks, _ = self._select_chunks(
                pid, retrieval_q, intent, per_paper - len(abstract), abstract=abstract,
                per_chunk_chars=700,
            )
            label = f"P{k} {_short_title(title)}"
            if not abstract and not chunks:
                missing[pid] = "no_chunks"
                continue
            used_pids.append(pid)
            paper_lines.append(f"P{k} = \"{title}\"{_byline(meta)}")
            if abstract:
                items.append({"arxiv_id": pid, "title": title, "section": "abstract",
                              "text": abstract, "label": f"{label} · Abstract"})
            for c in chunks:
                items.append({**c, "title": title, "label": f"{label} · {c['section'].capitalize()}"})
            if not chunks:
                missing[pid] = "abstract_only"

        if not items:
            yield {"type": "sources", "sources": []}
            yield {"type": "final", "mode": "multi", "papers": paper_ids,
                   "answer": self._empty_multi_answer(missing, titles, paper_status)}
            return
        sources = self._sources(items)
        yield {"type": "sources", "sources": [s.__dict__ for s in sources]}
        prompt = prompts.CHAT_MULTI_TEMPLATE.format(
            paper_list="\n".join(paper_lines),
            contexts=prompts.format_contexts(items),
            history=self._history_block(history), question=question,
            instruction=intent.instruction,
        )
        yield {"type": "status", "text": "Comparing the papers and writing the answer…"}
        final = ""
        for kind, text in stream_generate(
            prompt, system=prompts.SYSTEM_CHAT, n_sources=len(items),
            require_citations=True, max_tokens=max(settings.llm_max_new_tokens, 600),
        ):
            if kind == "token":
                yield {"type": "token", "text": text}
            else:
                final = text

        notes = []
        for pid, reason in missing.items():
            name = _short_title(titles.get(pid) or metas.get(pid, {}).get("title") or pid, 60)
            if reason == "no_chunks":
                notes.append(
                    f"{name} (arXiv:{pid}): no indexed full text — indexing status: "
                    f"{paper_status.get(pid, 'unknown')}. It was left out."
                )
            else:
                notes.append(f"{name}: full text unavailable, answered from its abstract.")
        if _NO_INFO_ANSWER.search(final[:300]):
            notes.append(
                "Tip: these papers may not cover this topic. Try asking about their "
                "goals, methods, results or limitations, or compare a specific aspect."
            )
        if notes:
            final = final.rstrip() + "\n\n" + "\n".join(f"- {n}" for n in notes)
        yield {"type": "final", "answer": final, "mode": "multi",
               "papers": used_pids, "notes": notes}

    # --------------------------------------------------- map-reduce (large)
    def _map_reduce_stream(
        self, question: str, paper_ids: list[str],
        titles: dict[str, str], paper_status: dict[str, str],
    ) -> Iterator[dict]:
        from backend.app.services.llm.generator import get_llm

        intent = detect_intent(question)
        yield {"type": "status", "text": f"Extracting evidence from {len(paper_ids)} papers…"}

        def _map_one(pid: str) -> tuple[str, dict | None, str]:
            chunks, _ = self._select_chunks(
                pid, question, intent, settings.map_reduce_per_doc_chunks * 900)
            if not chunks:
                return pid, None, "no_chunks"
            ctx_items = [{"label": f"section: {c['section']}", "text": c["text"]} for c in chunks]
            map_prompt = prompts.MAP_TEMPLATE.format(
                title=titles.get(pid, chunks[0]["title"]), arxiv_id=pid,
                contexts=prompts.format_contexts(ctx_items), question=question,
            )
            summary = safe_generate(map_prompt, system=prompts.SYSTEM_QA).strip()
            if self._is_no_info(summary):
                return pid, None, "no_match"
            return pid, {"arxiv_id": pid, "title": titles.get(pid, chunks[0]["title"]),
                         "summary": summary}, "ok"

        if get_llm().name != "hf":
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(4, len(paper_ids))) as ex:
                outcomes = list(ex.map(_map_one, paper_ids))
        else:
            outcomes = [_map_one(pid) for pid in paper_ids]
        per_paper = [item for _, item, _ in outcomes if item]
        skipped = {pid: reason for pid, item, reason in outcomes if not item}
        if not per_paper:
            yield {"type": "sources", "sources": []}
            yield {"type": "final", "mode": "multi", "papers": paper_ids,
                   "answer": self._empty_multi_answer(skipped, titles, paper_status)}
            return
        sources = [
            SourceRef(n=i, arxiv_id=p["arxiv_id"], title=p["title"], section="synthesis",
                      snippet=p["summary"][:400])
            for i, p in enumerate(per_paper, start=1)
        ]
        yield {"type": "sources", "sources": [s.__dict__ for s in sources]}
        ctx_items = [{"label": f"{p['title']} (arXiv:{p['arxiv_id']})", "text": p["summary"]}
                     for p in per_paper]
        reduce_prompt = prompts.REDUCE_TEMPLATE.format(
            contexts=prompts.format_contexts(ctx_items), question=question)
        final = ""
        for kind, text in stream_generate(
            reduce_prompt, system=prompts.SYSTEM_QA, n_sources=len(per_paper),
            require_citations=True,
        ):
            if kind == "token":
                yield {"type": "token", "text": text}
            else:
                final = text
        yield {"type": "final", "answer": final, "mode": "multi",
               "papers": [p["arxiv_id"] for p in per_paper]}
