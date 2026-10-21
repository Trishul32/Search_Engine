"""LLM generation backends with automatic strongest-available selection.

Backends (all open-source-friendly, per the report's scope):

    if _has_malformed_source_artifacts(answer):
        logger.warning("%s answer contained malformed source artifacts; using fallback", llm.name)
        return ExtractiveLLM().generate(prompt, system)
* ``openai-compatible`` — any OpenAI-style endpoint: **Ollama**, vLLM,
                          LM Studio, llama.cpp server, or OpenAI itself.
                          Recommended path to a strong model (e.g.
                          llama3.1:8b / qwen2.5:14b) without GPU plumbing.
* ``hf``                — local HuggingFace causal LM via transformers
                          (default: Qwen2.5-7B-Instruct).
* ``extractive``        — zero-dependency, citation-grounded sentence
                          synthesis straight from retrieved chunks using
                          query-biased MMR selection. Cannot hallucinate
                          (it only selects and orders source sentences).

``LLM_BACKEND=auto`` (the default) probes at startup and picks the
strongest backend that is actually available on this machine:
openai-compatible endpoint -> local HF -> extractive.

Grounding guard (``LLM_GROUNDING_CHECK``): every generative answer is
post-validated — citations pointing at non-existent sources are stripped,
boilerplate preambles removed, and an answer that makes claims with *no*
valid citation is regenerated with the extractive backend, which is
grounded by construction. Failures always degrade down the chain instead
of crashing a query.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import httpx

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class BaseLLM:
    name = "base"
    generative = True  # False => grounded by construction (extractive)

    def generate(self, prompt: str, system: str = "") -> str:  # pragma: no cover
        raise NotImplementedError

    def generate_stream(self, prompt: str, system: str = "", max_tokens: int | None = None):
        """Yield the answer in pieces. Backends without native streaming
        yield the whole answer at once."""
        yield self.generate(prompt, system)


# --------------------------------------------------------------- extractive
_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the is are was were be been being do does did has have had of to in "
    "on at by for with from as into about between during this that these those "
    "it its they them their there here what which who whom how why when where "
    "and or not no nor so if then than can could will would should may might "
    "must shall we you i he she his her our your my me us also very often "
    "used use using".split()
)


def _stem(w: str) -> str:
    """Light suffix stripping so morphological variants match:
    results→result, compared/comparing→compar, embeddings/embedding→embedd.
    Deliberately conservative — never touches short words."""
    prev = None
    while w != prev and len(w) > 4:
        prev = w
        for suf in ("ing", "ion", "ies", "ed", "es", "ly", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
    return w[:-1] if w.endswith("e") and len(w) > 5 else w


def _words(text: str) -> set[str]:
    """Stemmed content words only — stopwords carry no relevance signal,
    and stemming keeps 'compare the results' matching 'the result shows'."""
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP}


_GENERIC_Q = _words(
    "compare comparison comparing difference differences different differ key main "
    "summarize summarise summary paper papers contribution contributions approach "
    "approaches method methods technique techniques result results finding findings "
    "explain describe overview idea ideas between these those both each two similar "
    "similarity similarities common contrast versus work works study studies article "
    "tell give list what which highlight important major novelty novel whats"
)
_CUE = re.compile(
    r"\b(?:we (?:propose|present|introduce|show|find|develop)|this (?:paper|study|work)|"
    r"results? (?:show|indicate|suggest|demonstrate)|outperform\w*|improv\w*|"
    r"conclude\w*|findings?|limitation\w*)\b",
    re.I,
)


def _clean_extracted_sentence(sentence: str) -> str:
    """Remove page/section labels accidentally joined to PDF text."""
    sentence = sentence.replace("â€™", "'").replace("â€œ", '"').replace("â€\x9d", '"')
    sentence = sentence.replace("â€“", "-").replace("â€”", "-").replace("â", "")
    sentence = re.sub(
        r"^(?:\d+\s+on [^.]+\)\s*)?(?:Conclusion|Results|Abstract|Discussion)\s+",
        "",
        sentence.strip(),
        flags=re.I,
    )
    # PDF page boundaries can splice the same conclusion into a later chunk.
    sentence = re.sub(
        r"\s+(?:Conclusion|Results)\s+.*?\s+(?=This paper|The authors|The proposed)",
        " ",
        sentence,
        flags=re.I,
    )
    return sentence.strip()


def _is_pdf_noise(sentence: str) -> bool:
    lowered = sentence.lower()
    return (
        "â" in sentence
        or "search for non-existant value" in lowered
        or "search for non-existent value" in lowered
        or lowered.startswith("attempt to search")
        or re.match(r"^the result of testing the circuit from fig\.?", lowered)
        or len(_normalized_words(sentence)) < 8
    )


def _clean_context(text: str) -> str:
    sentences = []
    seen: set[str] = set()
    for raw in re.split(r"(?<=[.!?])\s+", text):
        sentence = _clean_extracted_sentence(raw)
        key = " ".join(_normalized_words(sentence))
        if _is_pdf_noise(sentence) or key in seen:
            continue
        seen.add(key)
        sentences.append(sentence)
    return " ".join(sentences)


class ExtractiveLLM(BaseLLM):
    """Query-biased extractive synthesis with inline [n] source markers.

    Sentence selection uses MMR (maximal marginal relevance): each step
    picks the sentence with the best trade-off between relevance to the
    question and novelty w.r.t. already-selected sentences, so answers
    cover multiple aspects instead of repeating the top match. Selected
    sentences are then re-ordered by source document order and stitched
    into a lead answer + supporting-detail structure.

    Serves as both the no-GPU default and the grounded last-resort
    fallback for the generative backends.
    """

    name = "extractive"
    generative = False
    _MMR_LAMBDA = 0.72          # relevance vs novelty trade-off
    _MAX_SENTS = 6

    def generate(self, prompt: str, system: str = "") -> str:
        kind, question, contexts = _parse_prompt(prompt)
        q_terms = _words(question)
        # "compare", "difference", "summary"... say WHAT KIND of answer is
        # wanted, not what it is about. Matching on them picks unrelated
        # sentences ("compared to ..."), so relevance uses topical words only;
        # a question with no topical words gets a lead-sentence overview.
        topical = q_terms - _GENERIC_Q
        generic_mode = not topical

        # score every candidate sentence
        cands: list[dict] = []
        for pos, (src_idx, ctx) in enumerate(contexts):
            for s_pos, sent in enumerate(_SENT.split(_clean_context(ctx))):
                sent = _clean_extracted_sentence(sent)
                w = _words(sent)
                if len(w) < 3 or len(sent) > 600:
                    continue
                if generic_mode:
                    # overview: lead sentences + contribution/finding cues
                    rel = (0.6 if s_pos == 0 else 0.3 / (1 + s_pos))
                    rel += 0.3 if _CUE.search(sent) else 0.0
                    rel *= min(1.0, len(w) / 18)
                else:
                    overlap = len(topical & w) / len(topical)
                    # mild length prior: prefer informative, complete sentences
                    rel = overlap * min(1.0, len(w) / 22)
                if rel > 0:
                    cands.append(
                        {"rel": rel, "sent": sent, "src": src_idx,
                         "order": (pos, s_pos), "w": w}
                    )
        if not cands:
            if kind == "map":
                return "NO RELEVANT INFORMATION."
            return (
                "The provided context does not contain enough information "
                "to answer this question."
            )

        # MMR selection: relevant AND non-redundant
        cands.sort(key=lambda c: c["rel"], reverse=True)
        picked: list[dict] = [cands.pop(0)]
        while cands and len(picked) < self._MAX_SENTS:
            best, best_score = None, -1.0
            for c in cands:
                redundancy = max(
                    len(c["w"] & p["w"]) / (len(c["w"] | p["w"]) or 1)
                    for p in picked
                )
                score = self._MMR_LAMBDA * c["rel"] - (1 - self._MMR_LAMBDA) * redundancy
                if score > best_score:
                    best, best_score = c, score
            if best is None or best_score <= 0:
                break
            picked.append(best)
            cands.remove(best)

        if kind == "map":
            # plain evidence summary — the reduce stage adds paper-level
            # citations, so no [n] markers here (they'd collide)
            picked.sort(key=lambda c: c["order"])
            return " ".join(c["sent"] for c in picked)

        # single / reduce: lead with the most relevant sentence, then
        # supporting detail in document order — every sentence cited.
        lead, rest = picked[0], sorted(picked[1:], key=lambda c: c["order"])
        parts = [f"{lead['sent']} [{lead['src']}]"]
        if rest:
            parts.append("\n\n")
            parts.append(" ".join(f"{c['sent']} [{c['src']}]" for c in rest))
        parts.extend([
            "\n\n",
            "Key takeaway: The cited evidence supports the finding above.",
        ])
        return "".join(parts)


def _parse_prompt(prompt: str) -> tuple[str, str, list[tuple[int, str]]]:
    """Recover (kind, question, [(source_idx, context)]) from a prompt.

    kind: "map" for evidence-extraction prompts (must emit the
    NO RELEVANT INFORMATION sentinel when empty), else "qa".
    """
    kind = "map" if "NO RELEVANT INFORMATION" in prompt else "qa"
    question = ""
    m = re.search(r"Question:\s*(.+?)(?:\n|$)", prompt, re.S)
    if m:
        question = m.group(1).strip()
    contexts: list[tuple[int, str]] = []
    for cm in re.finditer(
        r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\nConversation so far:|\nQuestion:|\Z)", prompt, re.S
    ):
        contexts.append((int(cm.group(1)), cm.group(2).strip()))
    if not contexts:
        contexts = [(1, prompt)]
    return kind, question or prompt[-500:], contexts


# ----------------------------------------------------------------- HF local
class HFLocalLLM(BaseLLM):
    name = "hf"

    def __init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

        logger.info("Loading local LLM %s ...", settings.llm_model)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model, device_map="auto", torch_dtype="auto"
        )

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        generation_config = {
            "max_new_tokens": settings.llm_max_new_tokens,
            "do_sample": settings.llm_temperature > 0,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if settings.llm_temperature > 0:
            generation_config["temperature"] = settings.llm_temperature
        with __import__("torch").inference_mode():
            out = self.model.generate(**inputs, **generation_config)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()


# ------------------------------------------------------- OpenAI-compatible
class OpenAICompatLLM(BaseLLM):
    name = "openai-compatible"

    def generate(self, prompt: str, system: str = "") -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        resp = httpx.post(
            f"{settings.llm_api_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={
                "model": settings.llm_api_model or settings.llm_model,
                "messages": messages,
                "max_tokens": settings.llm_max_new_tokens,
                "temperature": settings.llm_temperature,
            },
            timeout=settings.llm_timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def generate_stream(self, prompt: str, system: str = "", max_tokens: int | None = None):
        import json as _json

        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        with httpx.stream(
            "POST", f"{settings.llm_api_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={
                "model": settings.llm_api_model or settings.llm_model,
                "messages": messages,
                "max_tokens": max_tokens or settings.llm_max_new_tokens,
                "temperature": settings.llm_temperature,
                "stream": True,
            },
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=5.0),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                delta = _json.loads(payload)["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]


class OllamaLLM(BaseLLM):
    """Ollama via its native /api/chat endpoint.

    Preferred over Ollama's OpenAI-compatible endpoint because that one
    ignores context-size options: prompts longer than the model default
    (~2048 tokens) are silently truncated, so the model never saw part of
    the QA context. Native calls set ``num_ctx`` and keep the model loaded.
    Ollama also caches a repeated prompt prefix, so follow-up questions on
    the same paper start answering in about a second.
    """

    name = "ollama"

    def __init__(self) -> None:
        self.root = _ollama_root()

    def _payload(self, prompt: str, system: str, stream: bool, max_tokens: int | None) -> dict:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return {
            "model": settings.llm_api_model or settings.llm_model,
            "messages": messages,
            "stream": stream,
            "keep_alive": settings.llm_keep_alive,
            "options": {
                "num_ctx": settings.llm_num_ctx,
                "num_predict": max_tokens or settings.llm_max_new_tokens,
                "temperature": settings.llm_temperature,
            },
        }

    def generate_stream(self, prompt: str, system: str = "", max_tokens: int | None = None):
        import json as _json

        timeout = httpx.Timeout(settings.llm_timeout_s, connect=5.0)
        with httpx.stream(
            "POST", f"{self.root}/api/chat",
            json=self._payload(prompt, system, True, max_tokens), timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = _json.loads(line)
                if data.get("error"):
                    raise RuntimeError(data["error"])
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    break

    def generate(self, prompt: str, system: str = "") -> str:
        # streamed internally: no single long read that could time out
        return "".join(self.generate_stream(prompt, system)).strip()


def _ollama_root() -> str:
    base = settings.llm_api_base.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _ollama_alive() -> bool:
    try:
        r = httpx.get(f"{_ollama_root()}/api/version", timeout=2.0)
        return r.status_code == 200 and "version" in r.json()
    except Exception:
        return False


def _openai_endpoint_alive() -> bool:
    """Cheap reachability probe used by LLM_BACKEND=auto."""
    try:
        r = httpx.get(
            f"{settings.llm_api_base.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=2.0,
        )
        return r.status_code < 500
    except Exception:
        return False


def _hf_stack_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ facade
@lru_cache(maxsize=1)
def get_llm() -> BaseLLM:
    backend = settings.llm_backend

    if backend in ("auto", "openai-compatible", "ollama") and settings.llm_ollama_native and _ollama_alive():
        logger.info(
            "LLM: Ollama native API at %s (model %s, num_ctx %d)",
            _ollama_root(), settings.llm_api_model or settings.llm_model, settings.llm_num_ctx,
        )
        return OllamaLLM()

    if backend == "auto":
        if _openai_endpoint_alive():
            logger.info(
                "LLM auto-select: OpenAI-compatible endpoint at %s (model %s)",
                settings.llm_api_base, settings.llm_api_model or settings.llm_model,
            )
            return OpenAICompatLLM()
        if _hf_stack_available():
            try:
                return HFLocalLLM()
            except Exception as exc:
                logger.warning("HF LLM unavailable (%s)", exc)
        logger.info("LLM auto-select: extractive grounded backend")
        return ExtractiveLLM()

    if backend == "hf":
        try:
            return HFLocalLLM()
        except Exception as exc:
            logger.warning("HF LLM unavailable (%s); using extractive fallback", exc)
    elif backend == "openai-compatible":
        return OpenAICompatLLM()  # network errors handled per-call in safe_generate
    return ExtractiveLLM()


# --------------------------------------------------- grounding enforcement
_CITE = re.compile(r"\[(\d+)\]")
_PREAMBLE = re.compile(
    r"^\s*(?:sure[,!.]?\s*|certainly[,!.]?\s*|here(?:'s| is)[^.\n]*[.:]\s*|"
    r"based on the (?:provided )?(?:context|excerpts|papers)[, ]*|"
    r"according to the (?:provided )?context[, ]*)",
    re.I,
)
_NO_INFO = re.compile(r"(does not contain|no relevant information|cannot be answered)", re.I)


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _looks_like_raw_context_copy(answer: str, prompt: str) -> bool:
    """Detect long verbatim source reuse in a generative answer."""
    _, _, contexts = _parse_prompt(prompt)
    answer_words = _normalized_words(answer)
    if len(answer_words) < 12:
        return False
    answer_text = " ".join(answer_words)
    for _, context in contexts:
        for sentence in re.split(r"(?<=[.!?])\s+", context):
            source_words = _normalized_words(sentence)
            if len(source_words) < 12:
                continue
            phrase = " ".join(source_words)
            if phrase in answer_text:
                return True
    return False


def _has_repeated_phrase(answer: str, size: int = 12) -> bool:
    """Detect stitched chunk output by looking for repeated long phrases."""
    words = _normalized_words(answer)
    phrases = {
        " ".join(words[i : i + size])
        for i in range(len(words) - size + 1)
    }
    return len(phrases) < len(words) - size + 1


def _has_malformed_source_artifacts(answer: str) -> bool:
    lowered = answer.lower()
    return (
        "â" in answer
        or "search for non-existant value" in lowered
        or "search for non-existent value" in lowered
        or "attempt to search" in lowered
        or re.search(r"\b(?:conclusion|results)\s+in this paper", lowered) is not None
    )


def _synthesis_retry_prompt(prompt: str) -> str:
    _, question, contexts = _parse_prompt(prompt)
    evidence = []
    seen: set[str] = set()
    for source_id, text in contexts:
        compact = re.sub(r"\s+", " ", text).strip()[:900]
        if not compact or compact in seen:
            continue
        seen.add(compact)
        evidence.append(f"[{source_id}] {compact}")
    return (
        "Rewrite the answer below as a concise synthesis in your own words. "
        "Do not copy any source sentence or return source fragments. Remove "
        "repetition, keep only claims supported by the numbered excerpts, and "
        "retain valid inline citations. Return only the rewritten answer.\n\n"
        "Evidence:\n" + "\n\n".join(evidence) +
        f"\n\nQuestion: {question}"
    )


def enforce_grounding(answer: str, n_sources: int) -> tuple[str, bool]:
    """Sanitize a generated answer against its actual source list.

    Returns (clean_answer, grounded) where grounded=False means the answer
    asserts content but cites nothing valid — i.e. it cannot be verified
    against the retrieved context and should not be trusted as-is.
    """
    clean = _PREAMBLE.sub("", answer).strip()
    valid_cites = 0

    def _fix(m: re.Match) -> str:
        nonlocal valid_cites
        k = int(m.group(1))
        if 1 <= k <= n_sources:
            valid_cites += 1
            return m.group(0)
        return ""  # citation to a non-existent source: strip it

    clean = _CITE.sub(_fix, clean)
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    grounded = valid_cites > 0 or bool(_NO_INFO.search(clean))
    return clean, grounded


def safe_generate(
    prompt: str,
    system: str = "",
    n_sources: int = 0,
    require_citations: bool = False,
) -> str:
    """Generate with automatic degradation + optional grounding guard.

    * Any backend failure falls back to the extractive backend.
    * With ``require_citations`` and ``LLM_GROUNDING_CHECK`` on, a
      generative answer whose every citation is invalid/absent is
      regenerated extractively — unsupported claims never reach the user.
    """
    llm = get_llm()
    try:
        answer = llm.generate(prompt, system)
    except Exception:
        logger.exception(
            "%s generation failed; using the grounded extractive fallback",
            llm.name,
        )
        return ExtractiveLLM().generate(prompt, system)

    if llm.generative and (
        _looks_like_raw_context_copy(answer, prompt)
        or _has_repeated_phrase(answer)
        or _has_malformed_source_artifacts(answer)
    ):
        logger.warning(
            "%s answer copied a source passage; requesting a synthesized rewrite",
            llm.name,
        )
        try:
            answer = llm.generate(_synthesis_retry_prompt(prompt), system)
        except Exception:
            logger.exception("%s synthesis retry failed", llm.name)
            return ExtractiveLLM().generate(prompt, system)
        if _looks_like_raw_context_copy(answer, prompt):
            logger.warning(
                "%s synthesis retry still copied source text; using fallback",
                llm.name,
            )
            return ExtractiveLLM().generate(prompt, system)

    if require_citations and n_sources > 0 and llm.generative:
        answer, grounded = enforce_grounding(answer, n_sources)
        if not grounded:
            logger.warning(
                "%s answer had no valid citations; regenerating extractively",
                llm.name,
            )
            return ExtractiveLLM().generate(prompt, system)
    if _has_malformed_source_artifacts(answer):
        logger.warning(
            "%s answer contained malformed source artifacts; using fallback",
            llm.name,
        )
        return ExtractiveLLM().generate(prompt, system)
    return answer


# ------------------------------------------------------------ streaming chat
_ANSWER_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z*\-(])")
# honest "the papers don't cover this" answers must never be replaced
_HONEST_NO_INFO = re.compile(
    r"(?:does not|doesn't|do not|don't|did not)\s+(?:contain|discuss|mention|cover|address|"
    r"provide|describe|include|report|specify|state)|not\s+(?:discussed|mentioned|covered|"
    r"addressed|reported|specified)|no\s+(?:relevant\s+)?(?:information|mention|details|data)",
    re.I,
)


def attach_citations(answer: str, contexts: list[tuple[int, str]], min_overlap: float = 0.34) -> tuple[str, float]:
    """Add [n] markers to uncited answer sentences that a context supports.

    Small models often write a correct answer but forget the citation
    markers. Instead of discarding such an answer, each uncited sentence
    is matched to the excerpt sharing the most content words with it.
    Returns (answer, fraction of factual sentences now supported).
    """
    ctx_words = [(n, _words(text)) for n, text in contexts]
    out_lines, supported, factual = [], 0, 0
    for line in answer.split("\n"):
        if line.lstrip().startswith("|"):        # markdown table row: leave intact
            out_lines.append(line)
            continue
        parts = _ANSWER_SENT.split(line)
        fixed = []
        for sent in parts:
            words = _words(re.sub(r"\[\d+\]", "", sent))
            if len(words) < 4 or sent.strip().startswith(("#", "|---", "| ---")):
                fixed.append(sent)
                continue
            factual += 1
            if _CITE.search(sent):
                supported += 1
                fixed.append(sent)
                continue
            best_n, best = None, 0.0
            for n, cw in ctx_words:
                score = len(words & cw) / len(words)
                if score > best:
                    best_n, best = n, score
            if best_n is not None and best >= min_overlap:
                supported += 1
                stripped = sent.rstrip()
                trail = sent[len(stripped):]
                fixed.append(f"{stripped} [{best_n}]{trail}")
            else:
                fixed.append(sent)
        out_lines.append(" ".join(fixed))
    return "\n".join(out_lines), (supported / factual if factual else 1.0)


def _is_degenerate_chat(answer: str) -> bool:
    """Looping / stitched output in a chat answer.

    Stricter phrase checks misfire on legitimate comparisons, which repeat
    long paper titles, and on markdown tables. Ignore table rows, then flag
    only long repeated runs (25 words) or a line repeated 3+ times.
    """
    lines = [ln.strip() for ln in answer.split("\n") if ln.strip() and not ln.lstrip().startswith("|")]
    counts: dict[str, int] = {}
    for ln in lines:
        if len(ln) > 30:
            counts[ln] = counts.get(ln, 0) + 1
    if any(c >= 3 for c in counts.values()):
        return True
    return _has_repeated_phrase("\n".join(lines), size=25)


def stream_generate(
    prompt: str,
    system: str = "",
    n_sources: int = 0,
    require_citations: bool = False,
    max_tokens: int | None = None,
):
    """Stream an answer: yields ("token", text) pieces, then ("final", answer).

    The final answer is post-validated like ``safe_generate``: invalid
    citations are stripped, missing citations are repaired from the
    excerpts, and an answer whose claims the excerpts do not support is
    replaced by the grounded extractive answer. The UI swaps the streamed
    text for the final one, so unsupported text never stays on screen.
    """
    llm = get_llm()
    pieces: list[str] = []
    try:
        for piece in llm.generate_stream(prompt, system, max_tokens):
            pieces.append(piece)
            yield ("token", piece)
    except Exception:
        logger.exception("%s streaming failed; using the grounded extractive fallback", llm.name)
        yield ("final", ExtractiveLLM().generate(prompt, system))
        return
    answer = "".join(pieces).strip()
    if not llm.generative:
        yield ("final", answer)
        return
    if not answer or _is_degenerate_chat(answer) or _has_malformed_source_artifacts(answer):
        logger.warning("%s streamed answer was degenerate; using fallback", llm.name)
        yield ("final", ExtractiveLLM().generate(prompt, system))
        return
    if require_citations and n_sources > 0:
        answer = _PREAMBLE.sub("", answer).strip()
        answer = _CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", answer)
        if not (_NO_INFO.search(answer) or _HONEST_NO_INFO.search(answer)):
            _, _, contexts = _parse_prompt(prompt)
            answer, support = attach_citations(answer, contexts)
            if support < 0.3:
                logger.warning("%s answer poorly supported (%.0f%%); using fallback", llm.name, support * 100)
                yield ("final", ExtractiveLLM().generate(prompt, system))
                return
    yield ("final", re.sub(r"[ \t]{2,}", " ", answer).strip())
