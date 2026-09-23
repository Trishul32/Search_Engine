"""Central application configuration.

Every tunable in the pipeline (models, fusion weights, chunk sizes, DB URL,
LLM backend) is controlled here and overridable via environment variables or
a `.env` file, e.g. ``EMBEDDING_MODEL=BAAI/bge-base-en-v1.5``.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ------------------------------------------------------------------ app
    app_name: str = "arXiv Semantic Search Engine"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # ------------------------------------------------------------ database
    # SQLite by default (zero-setup); point at PostgreSQL in production:
    #   DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/arxiv_rag
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'app.db'}"

    # ------------------------------------------------------------ storage
    data_dir: Path = PROJECT_ROOT / "data"
    pdf_dir: Path = PROJECT_ROOT / "data" / "pdfs"
    index_dir: Path = PROJECT_ROOT / "data" / "indexes"

    # ------------------------------------------------------------- arXiv
    # Live candidates pulled per query. Kept small on purpose: this is a
    # synchronous call inside /search, and 15 fresh candidates per query
    # accumulate quickly across searches (each also costs an embedding).
    arxiv_max_results: int = 15
    arxiv_api_url: str = "https://arxiv.org/api/query"
    # Hard cap on the live-fetch HTTP call. Search must never hang on
    # a slow/blocked route to the arXiv API — on timeout it degrades to
    # the local corpus and reports why.
    arxiv_timeout_s: float = 8.0
    # Politeness controls (arXiv asks for ~1 request / 3s): identical
    # queries are cached, and uncached fetches are spaced out.
    arxiv_min_interval_s: float = 3.0
    arxiv_cache_ttl_s: float = 300.0
    # Cold-start seeding: when the corpus is empty at startup, ingest
    # the bundled seed corpus (~45 canonical IR/RAG/NLP papers) so the
    # first search always has something to rank — no dependence on the
    # live arXiv API being reachable. Abstracts in the seed file are
    # condensed summaries; full texts are fetched on QA selection.
    auto_seed_if_empty: bool = True
    seed_file: str = ""   # empty -> PROJECT_ROOT/seed/arxiv_seed.jsonl
    arxiv_categories: str = "cs.AI,cs.CL,cs.LG,cs.CV,cs.IR,cs.DC"

    # -------------------------------------------------------- embeddings
    # backend: "sentence-transformers" (production) | "hash" (dependency-free
    # deterministic fallback used for tests / environments without torch)
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384             # used by the hash fallback; ST models
    embedding_batch_size: int = 64       # report their own dim at runtime.

    # ---------------------------------------------------------- retrieval
    retrieval_top_k: int = 30            # candidates from each retriever
    fusion_method: str = "rrf"           # "rrf" | "weighted"
    fusion_alpha: float = 0.5            # dense weight for "weighted" fusion
    dynamic_alpha: bool = True           # per-query alpha tuning (DAT-lite)
    rrf_k: int = 60

    # --------------------------------------------------------- reranking
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 10               # papers surviving the reranker
    # Corrective retrieval: if the best reranked score is below this
    # confidence, expand the query and retry once (Corrective-RAG style).
    corrective_retrieval: bool = True
    corrective_score_threshold: float = 0.15

    # ---------------------------------------------------------- chunking
    chunk_size: int = 1000               # characters per chunk (target)
    chunk_overlap: int = 150
    min_chunk_chars: int = 200
    # Papers auto-selected (and only then downloaded/indexed) when
    # Multi-Paper QA is asked without an explicit paper list. Search
    # itself never downloads PDFs.
    lazy_load_top_n: int = 3

    # ----------------------------------------------------------------- QA
    qa_chunk_top_k: int = 8              # chunks fed to the LLM per answer
    map_reduce_per_doc_chunks: int = 4   # chunks per paper in map stage
    # Minimum top chunk score accepted as evidence for QA. The cross-encoder
    # returns sigmoid scores in [0, 1]; fixture measurements separate an
    # unrelated query at 0.0 from a valid generic match at 0.077.
    qa_relevance_threshold: float = 0.05
    # llm backend:
    #   "auto"              — pick the strongest available at startup:
    #                         openai-compatible endpoint (if reachable)
    #                         -> local HF model (if torch+transformers)
    #                         -> extractive
    #   "extractive"        — citation-grounded sentence synthesis, no model
    #   "hf"                — local HuggingFace model (llm_model)
    #   "openai-compatible" — Ollama / vLLM / LM Studio / OpenAI-style API
    # auto: Ollama/OpenAI-compatible endpoint -> local HF model -> extractive.
    # Set LLM_BACKEND=openai-compatible for Ollama, vLLM, LM Studio, etc.
    llm_backend: str = "auto"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"        # local HF default
    llm_api_model: str = "llama3.1:8b"                 # served-model name (Ollama etc.)
    llm_api_base: str = "http://localhost:11434/v1"    # Ollama default
    llm_api_key: str = "not-needed"
    llm_max_new_tokens: int = 700
    llm_temperature: float = 0.2
    # Grounding guard: post-validate generated answers — strip citations
    # pointing at non-existent sources and regenerate extractively when a
    # generative answer carries no valid citations at all (unsupported).
    llm_grounding_check: bool = True

    # -------------------------------------------------------------- misc
    max_conversation_turns: int = 6      # follow-up context window (turns)
    request_timeout_s: int = 60


settings = Settings()
settings.pdf_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
