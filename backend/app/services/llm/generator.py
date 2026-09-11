"""LLM generation backends (all open-source-friendly, per the report's scope).

* ``extractive``        — zero-dependency, citation-grounded sentence
                          synthesis straight from retrieved chunks. Cannot
                          hallucinate (it only reorders source sentences);
                          the default so the system runs anywhere.
* ``hf``                — local HuggingFace causal LM (LLaMA, Qwen, Mistral,
                          TinyLlama...) via transformers.
* ``openai-compatible`` — any OpenAI-style endpoint: Ollama, vLLM,
                          LM Studio, llama.cpp server, or OpenAI itself.

The active backend is chosen by LLM_BACKEND; failures degrade down the
chain hf/api -> extractive, never crashing a query.
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

    def generate(self, prompt: str, system: str = "") -> str:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------- extractive
_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")


class ExtractiveLLM(BaseLLM):
    """Query-biased extractive synthesis with inline [n] source markers.

    Scores each context sentence by unigram/bigram overlap with the
    question, picks the best-supported sentences across sources, and stitches
    them into a short grounded answer. Serves as both the no-GPU default and
    the last-resort fallback for the generative backends.
    """

    name = "extractive"

    def generate(self, prompt: str, system: str = "") -> str:
        question, contexts = _parse_prompt(prompt)
        q_terms = set(_WORD.findall(question.lower()))
        scored: list[tuple[float, str, int]] = []
        for src_idx, ctx in contexts:
            for sent in _SENT.split(ctx):
                sent = sent.strip()
                words = _WORD.findall(sent.lower())
                if len(words) < 6 or len(sent) > 500:
                    continue
                overlap = len(q_terms & set(words)) / (len(q_terms) or 1)
                density = overlap * min(1.0, len(words) / 25)
                if overlap > 0:
                    scored.append((density, sent, src_idx))
        if not scored:
            return (
                "The retrieved context does not contain enough information to "
                "answer this question confidently."
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        picked: list[tuple[str, int]] = []
        seen: set[str] = set()
        for _, sent, src in scored:
            key = " ".join(_WORD.findall(sent.lower()))[:120]
            if key in seen:
                continue
            seen.add(key)
            picked.append((sent, src))
            if len(picked) >= 5:
                break
        return " ".join(f"{s} [{i}]" for s, i in picked)


def _parse_prompt(prompt: str) -> tuple[str, list[tuple[int, str]]]:
    """Recover (question, [(source_idx, context)]) from the standard prompt."""
    question = ""
    m = re.search(r"Question:\s*(.+?)(?:\n|$)", prompt, re.S)
    if m:
        question = m.group(1).strip()
    contexts: list[tuple[int, str]] = []
    for cm in re.finditer(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\nQuestion:|\Z)", prompt, re.S):
        contexts.append((int(cm.group(1)), cm.group(2).strip()))
    if not contexts:
        contexts = [(1, prompt)]
    return question or prompt[-500:], contexts


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
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(
            inputs,
            max_new_tokens=settings.llm_max_new_tokens,
            temperature=max(0.01, settings.llm_temperature),
            do_sample=settings.llm_temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
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
                "model": settings.llm_model,
                "messages": messages,
                "max_tokens": settings.llm_max_new_tokens,
                "temperature": settings.llm_temperature,
            },
            timeout=settings.request_timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ------------------------------------------------------------------ facade
@lru_cache(maxsize=1)
def get_llm() -> BaseLLM:
    backend = settings.llm_backend
    if backend == "hf":
        try:
            return HFLocalLLM()
        except Exception as exc:
            logger.warning("HF LLM unavailable (%s); using extractive fallback", exc)
    elif backend == "openai-compatible":
        return OpenAICompatLLM()  # network errors handled per-call below
    return ExtractiveLLM()


def safe_generate(prompt: str, system: str = "") -> str:
    """Generate with automatic degradation to the extractive backend."""
    llm = get_llm()
    try:
        return llm.generate(prompt, system)
    except Exception as exc:
        logger.warning("%s generation failed (%s); extractive fallback", llm.name, exc)
        return ExtractiveLLM().generate(prompt, system)
