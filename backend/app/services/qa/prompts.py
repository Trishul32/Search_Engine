"""Prompt templates. All prompts demand grounding + [n] source attribution."""

SYSTEM_QA = (
    "You are a precise research assistant answering questions about scientific "
    "papers. Answer ONLY from the provided context. Cite sources inline using "
    "their bracketed numbers, e.g. [1], after each claim. If the context does "
    "not contain the answer, say so explicitly instead of guessing."
)

SINGLE_DOC_TEMPLATE = """Context from the paper "{title}" (arXiv:{arxiv_id}):

{contexts}

{history}Question: {question}

Answer the question using only the context above. Cite the numbered excerpts you used, e.g. [2]."""

MAP_TEMPLATE = """Context from the paper "{title}" (arXiv:{arxiv_id}):

{contexts}

Question: {question}

Extract every piece of information in the context relevant to the question, as a concise, self-contained summary. If nothing is relevant, reply exactly: NO RELEVANT INFORMATION."""

REDUCE_TEMPLATE = """You are synthesizing an answer from per-paper summaries.

Per-paper evidence:

{contexts}

Question: {question}

Write one coherent answer that synthesizes the evidence across papers: highlight agreements, disagreements, and complementary findings. Cite papers inline with their bracketed numbers, e.g. [1][3]. Do not repeat the same point twice."""


def format_contexts(items: list[dict], text_key: str = "text") -> str:
    """items: [{'label': 'title/section', text_key: ...}] -> numbered blocks."""
    lines = []
    for i, it in enumerate(items, start=1):
        label = it.get("label", "")
        lines.append(f"[{i}] {label}\n{it[text_key]}")
    return "\n\n".join(lines)
