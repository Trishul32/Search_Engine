# Real-Time Semantic Search Engine for Research Papers

A hybrid Retrieval-Augmented Generation (RAG) platform for arXiv research papers.
It combines **BM25 sparse retrieval** and **dense semantic retrieval** with
**cross-encoder reranking**, **lazy PDF loading**, **structure-aware chunking**,
a **FAISS vector database**, and **single- / multi-document question answering**
with map-reduce synthesis and source attribution — behind a **FastAPI** backend
and **Streamlit** frontend, with **PostgreSQL/SQLite** metadata storage and a
built-in **Recall@k / MRR / nDCG** evaluation harness.

```
                 ┌────────────────────────────────────────────────────────────┐
 user query ───▶ │ Query Processor: normalize · expand · analyze traits        │
                 └───────────────┬────────────────────────────────────────────┘
                                 ▼            (real-time arXiv API ingest, dedup)
                 ┌───────────────────────────────┐
                 │        HYBRID RETRIEVAL        │   BM25 (rank-bm25)  ─┐
                 │   run in parallel, then fuse   │   Dense (FAISS)     ─┤ RRF /
                 └───────────────┬───────────────┘   weighted fusion  ◀─┘ dynamic α
                                 ▼
                 ┌───────────────────────────────┐   low confidence?
                 │   Cross-Encoder RERANKER       │──▶ corrective re-query (1x)
                 └───────────────┬───────────────┘
                                 ▼  top-N papers only
                 ┌───────────────────────────────┐
                 │ LAZY PDF PIPELINE               │  download → PyMuPDF parse →
                 │                                 │  structure-aware chunking →
                 │                                 │  embed → FAISS chunk index
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │  QA ENGINE                     │  single-doc: filtered chunk
                 │  (LLM w/ source attribution)   │  retrieval + grounded answer
                 │                                │  multi-doc: MAP per paper →
                 │                                │  REDUCE cross-paper synthesis
                 └───────────────┬───────────────┘
                                 ▼
                 Streamlit UI  ◀──  FastAPI  ──▶  PostgreSQL / SQLite metadata
                                                  FAISS indexes (papers + chunks)
```

## Requirement traceability

| Report / Review-2 requirement | Where implemented |
|---|---|
| Natural-language queries, normalization + expansion | `services/query_processor.py` |
| Real-time arXiv API ingestion, metadata management | `services/arxiv_client.py`, `scripts/ingest.py` |
| Hybrid retrieval (BM25 + dense embeddings), parallel | `services/retrieval/` (`bm25_retriever`, `dense_retriever`, `hybrid`) |
| Adaptive fusion (not fixed weights) | `retrieval/fusion.py` — RRF + weighted fusion with **dynamic alpha tuning** (per-query, DAT-inspired, ref [1]) |
| Cross-encoder reranking | `services/reranker.py` (ms-marco cross-encoder; lexical fallback) |
| Corrective/adaptive retrieval (re-query on low confidence) | `services/pipeline.py::search` (Corrective-RAG-style, ref [9]) |
| Lazy PDF loading (top-ranked papers only) | `services/documents/loader.py`, `pipeline.ensure_paper_indexed` |
| Structure-aware chunking with section metadata | `services/documents/chunker.py` (font-size heading detection → canonical sections → sentence-boundary chunks w/ overlap) |
| FAISS vector database + metadata filtering | `services/vector_store.py` (IndexFlatIP → IVF at scale; **post-filtering** per ref [14]) |
| Single-document QA (focused, multi-turn follow-ups) | `services/qa/engine.py::answer_single` |
| Multi-document QA with **map-reduce** synthesis | `services/qa/engine.py::answer_multi` |
| LLM generation with source attribution, open-source LLMs | `services/llm/generator.py` (extractive / HuggingFace local / OpenAI-compatible e.g. Ollama) |
| Recommendations for related papers | `services/recommender.py` |
| REST APIs for retrieval & query operations | `backend/app/api/routes/api.py` (`/search`, `/qa/*`, `/papers/*`, `/admin/*`) |
| Streamlit web interface | `frontend/streamlit_app.py` (Search, Single-QA chat, Multi-QA, Recommendations, Admin) |
| PostgreSQL metadata storage | SQLAlchemy models (`db/models.py`); `DATABASE_URL` switches SQLite↔PostgreSQL |
| No duplicate indexing | unique `arxiv_id` constraint + dedup in retrievers + idempotent chunk indexing |
| Graceful PDF-failure handling | loader never raises; abstract-only fallback, error stored on the paper row |
| Evaluation: Recall@k, MRR, nDCG; sparse vs dense vs hybrid | `evaluation/evaluate.py` |
| Admin: rebuild indexes, stats, query history | `/api/v1/admin/*` + Admin tab in UI |

## Quickstart

```bash
# 1. install  (minimal — runs anywhere, lexical fallbacks)
pip install -r requirements.txt
#    or the full ML stack (real embeddings, cross-encoder, local LLMs):
pip install -r requirements-ml.txt

# 2. start the backend
uvicorn backend.app.main:app --reload          # http://localhost:8000/docs

# 3. start the frontend (new terminal)
streamlit run frontend/streamlit_app.py        # http://localhost:8501

# 4. (optional) pre-ingest a corpus
python -m scripts.ingest --arxiv-query "retrieval augmented generation" --max 200
python -m scripts.ingest --categories cs.IR cs.CL --max 300
# or from the Kaggle arXiv snapshot (the deck's primary dataset):
python -m scripts.ingest --kaggle-json arxiv-metadata-oai-snapshot.json --max 5000
```

Searching in the UI with "Fetch fresh papers from arXiv" enabled ingests new
papers in real time; you don't have to pre-ingest anything.

### Choosing an LLM backend

| `LLM_BACKEND` | What it does | Needs |
|---|---|---|
| `extractive` *(default)* | Citation-grounded sentence synthesis from retrieved chunks. Cannot hallucinate. | nothing |
| `hf` | Local open-source LLM (LLaMA, Qwen, Mistral, …) via transformers | `requirements-ml.txt`, GPU recommended |
| `openai-compatible` | Any OpenAI-style server: **Ollama**, vLLM, LM Studio, llama.cpp | a running endpoint |

Ollama example: `ollama run llama3.1`, then set
`LLM_BACKEND=openai-compatible  LLM_MODEL=llama3.1  LLM_API_BASE=http://localhost:11434/v1`.

## API overview (`/api/v1`)

| Endpoint | Purpose |
|---|---|
| `POST /search` | Hybrid retrieve → rerank → lazy-index; returns ranked papers |
| `POST /qa/single` | Question answering over one paper (multi-turn `history` supported) |
| `POST /qa/multi` | Map-reduce QA across selected papers (or let retrieval pick) |
| `GET /papers/{id}` | Paper metadata + indexing status |
| `POST /papers/{id}/index` | Force lazy load+chunk+embed for one paper |
| `GET /papers/{id}/recommendations` | Related papers (embedding + category similarity) |
| `GET /admin/stats` · `POST /admin/rebuild` · `GET /admin/history` | Operations |

Interactive docs: `http://localhost:8000/docs`.

## Evaluation

```bash
python -m evaluation.evaluate --eval-file evaluation/datasets/sample_eval.jsonl --k 1 5 10
```

Prints Recall@k, MRR, and nDCG@k for `bm25`, `dense`, `hybrid`, and
`hybrid+rerank` side-by-side. Eval sets are JSONL:
`{"query": "...", "relevant": ["<arxiv_id>", ...]}` — build one for your
ingested corpus to reproduce the report's Phase-2 comparison.

## Tests

```bash
python -m pytest tests/ -q        # 21 tests, fully offline
```

Covers: query processing, dynamic-alpha fusion, BM25/dense/hybrid retrieval,
reranking, PDF parsing + structure-aware chunking (against a generated PDF),
graceful PDF failure, FAISS filtering/persistence, dedup, lazy-index
idempotency, single & multi-doc QA, conversation history, recommendations,
all API routes, and the metric implementations.

## Docker deployment (with PostgreSQL)

```bash
docker compose up --build
# frontend :8501 · api :8000 · postgres :5432
```

## Architecture notes & design decisions

* **Two-level retrieval.** Paper-level hybrid retrieval over title+abstract
  decides *which papers matter* before any PDF is touched (lazy loading);
  chunk-level FAISS retrieval with metadata filters serves QA. Chunk
  candidates are additionally reranked before hitting the LLM.
* **Fusion.** RRF is the default (scale-free, robust). Weighted fusion with
  **dynamic alpha** is available: query traits (acronyms, quoted phrases,
  length, math tokens) shift weight toward BM25 for technical lookups and
  toward dense retrieval for conceptual questions — the adaptive behavior the
  report's research-gap analysis calls for.
* **Structure-aware chunking.** Headings are detected from PDF font
  geometry at line level, mapped to canonical sections (abstract,
  introduction, methodology, experiments, results, …); references/appendix
  are dropped; chunks respect sentence boundaries with sentence-level
  overlap and carry their section as filterable metadata.
* **Graceful degradation everywhere.** No torch → hash embedder + lexical
  reranker; no LLM → extractive grounded answers; unreachable/broken PDF →
  abstract-only indexing; arXiv API down → local corpus. A query can always
  be answered; components upgrade quality when their dependencies exist.
* **Scaling path.** Flat FAISS indexes are exact and fastest below ~1M
  vectors; `maybe_train_ivf()` upgrades to IVF beyond a threshold, and
  metadata filtering uses post-filtering to preserve index efficiency.

## Project layout

```
backend/app/
  core/config.py            all settings (env-overridable)
  db/                       SQLAlchemy models + session (SQLite/PostgreSQL)
  api/                      FastAPI schemas + routes
  services/
    arxiv_client.py         real-time arXiv ingestion
    query_processor.py      normalization, expansion, trait analysis
    retrieval/              bm25, dense, fusion (RRF/weighted/dynamic α), hybrid
    reranker.py             cross-encoder + fallback, confidence gate
    documents/              lazy loader (PyMuPDF) + structure-aware chunker
    embeddings.py           sentence-transformers + hash fallback
    vector_store.py         FAISS (papers + chunks), filtering, persistence, IVF
    llm/generator.py        extractive / HF local / OpenAI-compatible
    qa/                     prompts + single-doc & map-reduce multi-doc engine
    recommender.py          related-paper suggestions
    pipeline.py             end-to-end orchestrator + admin ops + query log
frontend/streamlit_app.py   5-tab UI
evaluation/evaluate.py      Recall@k, MRR, nDCG across 4 strategies
scripts/ingest.py           arXiv API / Kaggle-snapshot bulk ingestion
tests/                      21 offline tests
```
