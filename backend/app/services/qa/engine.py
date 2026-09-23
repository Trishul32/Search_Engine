"""Question answering over indexed chunks.

Single-document mode: chunk retrieval restricted (metadata filter) to one
paper; supports multi-turn follow-ups via conversation history.

Multi-document mode: map-reduce — the *map* stage answers the question
independently from each paper's best chunks (bounding per-paper context and
avoiding context overflow), the *reduce* stage synthesizes one coherent,
deduplicated answer across papers with bracketed citations.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from backend.app.core.config import settings
from backend.app.services.embeddings import BaseEmbedder
from backend.app.services.llm.generator import safe_generate
from backend.app.services.qa import prompts
from backend.app.services.reranker import Reranker
from backend.app.services.vector_store import FaissStore

logger = logging.getLogger(__name__)


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


class QAEngine:
    def __init__(
        self, chunk_store: FaissStore, embedder: BaseEmbedder, reranker: Reranker
    ):
        self.chunk_store = chunk_store
        self.embedder = embedder
        self.reranker = reranker

    # ------------------------------------------------------------ helpers
    def _retrieve_chunks(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        top_k: int | None = None,
        section: str | None = None,
    ) -> list[dict]:
        top_k = top_k or settings.qa_chunk_top_k
        qv = self.embedder.encode_one(question)
        allowed = set(paper_ids) if paper_ids else None

        def _filter(meta: dict) -> bool:
            if allowed is not None and meta.get("arxiv_id") not in allowed:
                return False
            if section and meta.get("section") != section:
                return False
            return True

        hits = self.chunk_store.search(
            qv, top_k=top_k * 2, filter_fn=_filter, overfetch=8
        )
        cands = [
            {
                "arxiv_id": m.get("arxiv_id", ""),
                "title": m.get("title", ""),
                "section": m.get("section", "body"),
                "text": m.get("text", ""),
                "fused_score": score,
            }
            for _, score, m in hits
        ]
        # chunk-level rerank sharpens the context fed to the LLM
        ranked = self.reranker.rerank(question, cands, top_k=top_k)
        if (
            not ranked
            or ranked[0].get("rerank_score", 0.0)
            < settings.qa_relevance_threshold
        ):
            logger.info(
                "No QA evidence above threshold %.3f for question %r "
                "(top score %.3f)",
                settings.qa_relevance_threshold,
                question,
                ranked[0].get("rerank_score", 0.0) if ranked else 0.0,
            )
            return []
        unique: list[dict] = []
        seen: set[str] = set()
        seen_words: list[set[str]] = []
        for chunk in ranked:
            key = re.sub(r"\s+", " ", chunk.get("text", "").strip().lower())
            if not key or key in seen:
                continue
            words = set(re.findall(r"[a-z0-9]+", key))
            if any(
                len(words & prior) / max(1, len(words | prior)) >= 0.75
                for prior in seen_words
            ):
                continue
            seen.add(key)
            seen_words.append(words)
            unique.append(chunk)
        return unique

    @staticmethod
    def _history_block(history: list[dict] | None) -> str:
        if not history:
            return ""
        turns = history[-settings.max_conversation_turns:]
        lines = [f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}"
                 for t in turns]
        return "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    def _has_indexed_chunks(self, paper_id: str) -> bool:
        """Distinguish low-relevance retrieval from missing full text."""
        return self.chunk_store.contains(
            lambda meta: meta.get("arxiv_id") == paper_id
        )

    @staticmethod
    def _sources(chunks: list[dict]) -> list[SourceRef]:
        return [
            SourceRef(
                n=i,
                arxiv_id=c["arxiv_id"],
                title=c["title"],
                section=c["section"],
                snippet=c["text"][:300],
            )
            for i, c in enumerate(chunks, start=1)
        ]

    # ------------------------------------------------------------- single
    def answer_single(
        self,
        question: str,
        arxiv_id: str,
        title: str = "",
        history: list[dict] | None = None,
    ) -> QAResult:
        chunks = self._retrieve_chunks(question, paper_ids=[arxiv_id])
        if not chunks:
            return QAResult(
                answer=(
                    "This paper could not be indexed for QA — its PDF may have "
                    "failed to download or parse, and no abstract fallback was "
                    "available. Check the paper's status via GET /papers/{id}."
                ),
                mode="single",
                papers=[arxiv_id],
            )
        ctx_items = [
            {"label": f"section: {c['section']}", "text": c["text"]} for c in chunks
        ]
        prompt = prompts.SINGLE_DOC_TEMPLATE.format(
            title=title or chunks[0]["title"],
            arxiv_id=arxiv_id,
            contexts=prompts.format_contexts(ctx_items),
            history=self._history_block(history),
            question=question,
        )
        answer = safe_generate(
            prompt, system=prompts.SYSTEM_QA,
            n_sources=len(chunks), require_citations=True,
        )
        return QAResult(
            answer=answer,
            sources=self._sources(chunks),
            mode="single",
            papers=[arxiv_id],
        )

    # -------------------------------------------------------- multi (M/R)
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
        self,
        question: str,
        paper_ids: list[str],
        titles: dict[str, str] | None = None,
        history: list[dict] | None = None,
        paper_status: dict[str, str] | None = None,
    ) -> QAResult:
        titles = titles or {}
        paper_status = paper_status or {}

        # ---- MAP: per-paper evidence extraction over that paper's chunks.
        # Papers are independent, so map calls run in parallel — a large
        # speedup with a served LLM (Ollama/vLLM). A local HF model stays
        # serial (concurrent generate() on one model is not safe).
        from backend.app.services.llm.generator import get_llm

        def _map_one(pid: str) -> tuple[str, dict | None, str]:
            chunks = self._retrieve_chunks(
                question, paper_ids=[pid],
                top_k=settings.map_reduce_per_doc_chunks,
            )
            if not chunks:
                reason = "no_match" if self._has_indexed_chunks(pid) else "no_chunks"
                return pid, None, reason
            ctx_items = [
                {"label": f"section: {c['section']}", "text": c["text"]}
                for c in chunks
            ]
            map_prompt = prompts.MAP_TEMPLATE.format(
                title=titles.get(pid, chunks[0]["title"]),
                arxiv_id=pid,
                contexts=prompts.format_contexts(ctx_items),
                question=question,
            )
            summary = safe_generate(map_prompt, system=prompts.SYSTEM_QA).strip()
            if self._is_no_info(summary):
                return pid, None, "no_match"
            return pid, {
                "arxiv_id": pid,
                "title": titles.get(pid, chunks[0]["title"]),
                "summary": summary,
                "chunks": chunks,
            }, "ok"

        skipped: dict[str, str] = {}
        per_paper: list[dict] = []
        if len(paper_ids) > 1 and get_llm().name != "hf":
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(4, len(paper_ids))) as ex:
                outcomes = list(ex.map(_map_one, paper_ids))
        else:
            outcomes = [_map_one(pid) for pid in paper_ids]
        for pid, item, reason in outcomes:  # preserves selection order
            if item is not None:
                per_paper.append(item)
            else:
                skipped[pid] = reason

        if not per_paper:
            return QAResult(
                answer=self._empty_multi_answer(skipped, titles, paper_status),
                mode="multi",
                papers=paper_ids,
            )

        # ---- REDUCE: cross-paper synthesis
        ctx_items = [
            {"label": f"{p['title']} (arXiv:{p['arxiv_id']})", "text": p["summary"]}
            for p in per_paper
        ]
        reduce_prompt = prompts.REDUCE_TEMPLATE.format(
            contexts=prompts.format_contexts(ctx_items),
            question=question,
        )
        answer = safe_generate(
            reduce_prompt, system=prompts.SYSTEM_QA,
            n_sources=len(per_paper), require_citations=True,
        )

        sources = [
            SourceRef(
                n=i,
                arxiv_id=p["arxiv_id"],
                title=p["title"],
                section="synthesis",
                snippet=p["summary"][:300],
            )
            for i, p in enumerate(per_paper, start=1)
        ]
        return QAResult(
            answer=answer,
            sources=sources,
            mode="multi",
            papers=[p["arxiv_id"] for p in per_paper],
        )
