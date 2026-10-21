"""Question intent for paper QA.

Questions like "what's the key difference between these papers?" or
"summarize this paper" contain no topical words, so pure relevance
retrieval finds nothing to match (every passage scores ~0). For such
questions the right evidence is structural: the abstract, introduction,
method, results and conclusion. The intent decides which sections to read
and how the answer should be shaped (overview, comparison table, ...).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    name: str
    sections: tuple[str, ...]      # preferred sections, most useful first
    hint: str                      # added to the retrieval query
    instruction: str               # answer-shape instruction for the LLM
    generic: bool                  # True: the question has little topical content


_INTENTS: list[tuple[re.Pattern, Intent]] = [
    (re.compile(r"\b(differen\w*|differ|compar\w*|contrast|versus|vs\.?|similar\w*|common|"
                r"in common|which (?:one|paper) is better|same|distinguish)\b", re.I),
     Intent("compare", ("abstract", "introduction", "methodology", "results", "experiments", "conclusion"),
            "main goal approach method data evaluation results findings conclusion",
            "Start with a one-sentence answer. Then give a markdown table with the header "
            "| Paper | Goal | Approach | Evidence | Key findings | and one row per paper "
            "(short cells, no citations inside the table). End with 2-4 bullets on the key "
            "differences, each with citations.", True)),
    (re.compile(r"\b(summar\w*|overview|main idea|gist|tl;?dr|explain (?:this|the) paper|"
                r"what(?: is|'s) (?:this|the) paper|(?:this|the) paper about|abstract|"
                r"describe (?:this|the) paper)\b", re.I),
     Intent("overview", ("abstract", "introduction", "conclusion"),
            "main goal problem approach key findings contribution",
            "Give a short overview: the problem, the approach, the key findings, and why it "
            "matters. Use 3-5 bullets.", True)),
    (re.compile(r"\b(contribution\w*|novel\w*|what'?s new|new about|key idea|innovation|propose)\b", re.I),
     Intent("contribution", ("abstract", "introduction", "conclusion"),
            "we propose contribution novel main idea",
            "List the main contributions as bullets, most important first.", True)),
    (re.compile(r"\b(limitation\w*|weakness\w*|drawback\w*|shortcoming\w*|future work|"
                r"open (?:problem|question)s?|challenge\w*|gaps?)\b", re.I),
     Intent("limitation", ("discussion", "conclusion", "results"),
            "limitations future work challenges drawbacks",
            "List the limitations and open problems the paper states. If it states none, "
            "say so.", True)),
    (re.compile(r"\b(result\w*|finding\w*|perform\w*|accura\w*|outcome\w*|how well|"
                r"evaluat\w*|experiment\w*|metric\w*|score\w*|effective\w*)\b", re.I),
     Intent("results", ("results", "experiments", "discussion", "conclusion", "abstract"),
            "results findings performance evaluation experiments",
            "Report the key results with the numbers the excerpts give.", False)),
    (re.compile(r"\b(dataset\w*|data|benchmark\w*|corpus|participants?|sample size|cohort)\b", re.I),
     Intent("data", ("experiments", "methodology", "results"),
            "dataset data collection participants benchmark evaluation setup",
            "Name the datasets, participants or benchmarks and how they were used.", False)),
    (re.compile(r"\b(method\w*|approach\w*|how does|how do|how is|architecture|algorithm\w*|"
                r"technique\w*|framework|pipeline|model|work\w*|implement\w*|design)\b", re.I),
     Intent("method", ("methodology", "introduction", "abstract"),
            "method approach architecture how it works",
            "Explain how the method works, step by step where that helps.", False)),
    (re.compile(r"\b(conclu\w*|takeaway\w*|implication\w*|lesson\w*|recommend\w*)\b", re.I),
     Intent("conclusion", ("conclusion", "discussion", "abstract"),
            "conclusion implications recommendations",
            "State the conclusions and their implications.", True)),
]

GENERAL = Intent("general", ("abstract", "introduction", "conclusion"), "",
                 "Answer directly, then explain briefly.", False)

_CONTENT = re.compile(r"[a-z0-9]{3,}")
_FILLER = set(
    "what whats which how does do the this that these those paper papers about between "
    "key main their them they its and are was were with from for can you please tell "
    "give explain describe compare difference differences study studies article work "
    "both each other two three".split()
)


def detect_intent(question: str) -> Intent:
    for pattern, intent in _INTENTS:
        if pattern.search(question):
            if intent.generic:
                return intent
            # a 'method'/'results' question with no topical words is generic too
            topical = [w for w in _CONTENT.findall(question.lower()) if w not in _FILLER]
            return intent if len(topical) > 1 else Intent(
                intent.name, intent.sections, intent.hint, intent.instruction, True)
    topical = [w for w in _CONTENT.findall(question.lower()) if w not in _FILLER]
    if not topical:
        return Intent("overview", ("abstract", "introduction", "conclusion"),
                      "main goal approach key findings", GENERAL.instruction, True)
    return GENERAL


_PRONOUN = re.compile(r"\b(it|its|this|that|they|them|their|these|those|he|she|the same|above)\b", re.I)


def contextualize(question: str, history: list[dict] | None) -> str:
    """Make a follow-up self-contained for retrieval: 'what about its
    limitations?' -> previous question + this one."""
    if not history:
        return question
    previous = next((t["content"] for t in reversed(history) if t.get("role") == "user"), "")
    if previous and (len(question.split()) <= 6 or _PRONOUN.search(question)):
        return f"{previous} {question}"
    return question
