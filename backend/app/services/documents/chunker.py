"""Structure-aware chunking of parsed papers.

Instead of blind fixed-size splitting, chunking respects the paper's logical
sections (abstract, introduction, methodology, experiments, results,
conclusion, ...). Within a section, text is split at sentence boundaries
into ~chunk_size character windows with overlap, and every chunk carries its
canonical section label as metadata — enabling section-filtered retrieval
and better answer faithfulness (motivated by Amiri & Bocklitz, ref [3]).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.core.config import settings
from backend.app.services.documents.loader import ParsedDocument

# heading text -> canonical section label
_SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\babstract\b", re.I), "abstract"),
    (re.compile(r"\bintroduction\b|\bbackground\b|\bmotivation\b", re.I), "introduction"),
    (re.compile(r"\brelated\s+work\b|\bliterature\b|\bprior\s+work\b", re.I), "related_work"),
    (re.compile(r"\bmethod(olog(y|ies))?s?\b|\bapproach\b|\bmodel\b|\barchitecture\b|\bproposed\b|\bframework\b|\bsystem\s+design\b|\bdesign\b", re.I), "methodology"),
    (re.compile(r"\bexperiment(s|al)?\b|\bevaluation\b|\bsetup\b|\bimplementation\b|\bdataset(s)?\b", re.I), "experiments"),
    (re.compile(r"\bresult(s)?\b|\bfinding(s)?\b|\banalysis\b|\bablation\b|\bperformance\b", re.I), "results"),
    (re.compile(r"\bdiscussion\b|\blimitation(s)?\b|\bfuture\s+work\b", re.I), "discussion"),
    (re.compile(r"\bconclusion(s)?\b|\bsummary\b", re.I), "conclusion"),
    (re.compile(r"\breference(s)?\b|\bbibliograph", re.I), "references"),
    (re.compile(r"\bappendix\b|\bsupplementary\b|\backnowledg", re.I), "appendix"),
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
# arXiv PDF artifacts worth stripping from chunk text
_NOISE = re.compile(r"arXiv:\d{4}\.\d{4,5}(v\d+)?|^\d+$")


@dataclass
class DocChunk:
    chunk_index: int
    section: str
    heading: str
    text: str


def canonical_section(heading: str) -> str:
    h = heading.strip()
    if not h:
        return "body"
    for pat, label in _SECTION_PATTERNS:
        if pat.search(h):
            return label
    return "body"


def _sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    doc: ParsedDocument,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[DocChunk]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap
    chunks: list[DocChunk] = []
    idx = 0

    for heading, text in doc.blocks:
        section = canonical_section(heading)
        if section in ("references", "appendix"):
            continue  # citations lists add noise, not answerable content
        text = _NOISE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        sents = _sentences(text) or [text]
        window: list[str] = []
        length = 0
        fresh = 0  # sentences added since last emit (guards overlap-only residual)
        for sent in sents:
            window.append(sent)
            length += len(sent) + 1
            fresh += 1
            if length >= chunk_size:
                chunk_text = " ".join(window)
                chunks.append(DocChunk(idx, section, heading, chunk_text))
                idx += 1
                # sentence-level overlap: keep tail sentences ~overlap chars
                tail: list[str] = []
                acc = 0
                for s in reversed(window):
                    tail.insert(0, s)
                    acc += len(s) + 1
                    if acc >= overlap:
                        break
                window, length, fresh = tail, acc, 0
        residual = " ".join(window).strip()
        if residual and fresh and (
            len(residual) >= settings.min_chunk_chars or not chunks
        ):
            chunks.append(DocChunk(idx, section, heading, residual))
            idx += 1
        elif residual and fresh and chunks and chunks[-1].section == section:
            # merge small residuals into the previous chunk of the section
            chunks[-1].text += " " + residual

    return chunks
