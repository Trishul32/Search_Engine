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
        return self.reranker.rerank(question, cands, top_k=top_k)

    @staticmethod
    def _history_block(history: list[dict] | None) -> str:
        if not history:
            return ""
        turns = history[-settings.max_conversation_turns:]
        lines = [f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}"
                 for t in turns]
        return "Conversation so far:\n" + "\n".join(lines) + "\n\n"

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
                    "No indexed content found for this paper yet — its PDF may "
                    "not have been processed. Try re-running the search so the "
                    "lazy loader picks it up."
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
        answer = safe_generate(prompt, system=prompts.SYSTEM_QA)
        return QAResult(
            answer=answer,
            sources=self._sources(chunks),
            mode="single",
            papers=[arxiv_id],
        )

    # -------------------------------------------------------- multi (M/R)
    def answer_multi(
        self,
        question: str,
        paper_ids: list[str],
        titles: dict[str, str] | None = None,
        history: list[dict] | None = None,
    ) -> QAResult:
        titles = titles or {}
        # ---- MAP: per-paper evidence extraction over that paper's chunks
        per_paper: list[dict] = []
        for pid in paper_ids:
            chunks = self._retrieve_chunks(
                question, paper_ids=[pid],
                top_k=settings.map_reduce_per_doc_chunks,
            )
            if not chunks:
                continue
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
            summary = safe_generate(map_prompt, system=prompts.SYSTEM_QA)
            if "NO RELEVANT INFORMATION" in summary.upper():
                continue
            per_paper.append(
                {
                    "arxiv_id": pid,
                    "title": titles.get(pid, chunks[0]["title"]),
                    "summary": summary,
                    "chunks": chunks,
                }
            )

        if not per_paper:
            return QAResult(
                answer=(
                    "None of the selected papers contain indexed content "
                    "relevant to this question."
                ),
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
        answer = safe_generate(reduce_prompt, system=prompts.SYSTEM_QA)

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
