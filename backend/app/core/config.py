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
    arxiv_max_results: int = 50          # metadata candidates pulled per query
    arxiv_page_size: int = 100
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
    lazy_load_top_n: int = 3             # papers whose PDFs are lazily loaded

    # ----------------------------------------------------------------- QA
    qa_chunk_top_k: int = 8              # chunks fed to the LLM per answer
    map_reduce_per_doc_chunks: int = 4   # chunks per paper in map stage
    # llm backend: "extractive" (no external model, citation-grounded
    # sentence synthesis) | "hf" (local HuggingFace model) |
    # "openai-compatible" (Ollama / vLLM / LM Studio / OpenAI)
    llm_backend: str = "extractive"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_api_base: str = "http://localhost:11434/v1"   # Ollama default
    llm_api_key: str = "not-needed"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.2

    # -------------------------------------------------------------- misc
    max_conversation_turns: int = 6      # follow-up context window (turns)
    request_timeout_s: int = 60


settings = Settings()
settings.pdf_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
