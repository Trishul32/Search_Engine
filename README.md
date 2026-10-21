# ScholarLens: Real-Time Semantic Search Engine for Research Papers

A hybrid Retrieval-Augmented Generation (RAG) platform for arXiv research papers.
It combines **BM25 sparse retrieval** and **dense semantic retrieval** with
**cross-encoder reranking**, **lazy PDF loading**, **structure-aware chunking**,
a **FAISS vector database**, and **single- / multi-document question answering**
with map-reduce synthesis and source attribution — behind a **FastAPI** backend
and **Streamlit** frontend, with **PostgreSQL/SQLite** metadata storage and a
built-in **Recall@k / MRR / nDCG** evaluation harness.

On top of search and QA it adds three research-assistant capabilities:
**query understanding** (negation such as "computer science *without AI*",
research-field filters, and LLM rewriting of plain-language questions into
academic vocabulary), a **paper → knowledge graph** of the research
landscape, and **"What should I read next?"** recommendations that find the
missing conceptual connection between papers you have already read.

```
                 ┌────────────────────────────────────────────────────────────┐
 user query ───▶ │ Query Understanding: topic · exclusions ("without AI") ·    │
                 │ fields (cs.*) · year · LLM rewrite (plain words → terms)    │
                 └───────────────┬────────────────────────────────────────────┘
                                 ▼   structured live arXiv query: ti:/abs: fields,
                                     cat: filters, ANDNOT exclusions (+ relaxed retry)
                 ┌───────────────────────────────┐
                 │        HYBRID RETRIEVAL        │   BM25 (rank-bm25)  ─┐
                 │   run in parallel, then fuse   │   Dense (FAISS)     ─┤ RRF /
                 └───────────────┬───────────────┘   weighted fusion  ◀─┘ dynamic α
                                 ▼
                    hard filters: exclusions + fields (before reranking)
                                 ▼
                 ┌───────────────────────────────┐   low confidence?
                 │   Cross-Encoder RERANKER       │──▶ corrective re-query (1x)
                 └───────────────┬───────────────┘
                                 ▼
                     ranked results + "why it matched" abstract sentence
                     (metadata only — search never downloads PDFs)
                                 │
                                 ▼  user selects paper(s) for QA
                 ┌───────────────────────────────┐
                 │ ON-DEMAND PDF PIPELINE          │  download → PyMuPDF parse →
                 │ (only the selected papers;      │  structure-aware chunking →
                 │  idempotent, cached forever)    │  embed → FAISS chunk index
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │  QA ENGINE                     │  single-doc: filtered chunk
                 │  (grounded LLM + citation      │  retrieval + grounded answer
                 │   enforcement)                 │  multi-doc: MAP per paper →
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
| Real-time arXiv API ingestion (direct Atom API, hard 8s timeout — a slow arXiv route can never hang search), metadata management | `services/arxiv_client.py`, `scripts/ingest.py` |
| Hybrid retrieval (BM25 + dense embeddings), parallel | `services/retrieval/` (`bm25_retriever`, `dense_retriever`, `hybrid`) |
| Adaptive fusion (not fixed weights) | `retrieval/fusion.py` — RRF + weighted fusion with **dynamic alpha tuning** (per-query, DAT-inspired, ref [1]) |
| Cross-encoder reranking | `services/reranker.py` (ms-marco cross-encoder; lexical fallback) |
| Corrective/adaptive retrieval (re-query on low confidence) | `services/pipeline.py::search` (Corrective-RAG-style, ref [9]) |
| Lazy PDF loading — download & index **only when a paper is selected for QA**; search stays metadata-only | `services/documents/loader.py`, `pipeline.ensure_paper_indexed` (called from `ask_single`/`ask_multi`, never from `search`) |
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
| Negation / exclusion queries ("without AI", "not about LLMs", "-blockchain", "non-AI") with synonym + category expansion | `services/query_processor.py::extract_exclusions`, `services/relevance.py::exclusion_hit` |
| Research-field filters ("computer science" → `cat:cs.*`) | `services/query_processor.py::extract_fields` |
| Abstract-level matching: structured arXiv query over title **and abstract**, live matches always reranked, best-matching abstract sentence returned | `query_processor.build_arxiv_query`, `pipeline._retrieve_and_rerank`, `pipeline._attach_evidence` |
| LLM query rewriting + HyDE pseudo-abstract for plain-language queries | `services/query_rewriter.py` |
| Paper → knowledge graph (introduces / uses / extends / evaluated on / improves) + landscape insights | `services/knowledge_graph.py`, `POST /graph` (API) |
| Understand one paper: concept map of major topics → subtopics ranked by importance, each with its role, the paper's own explanation and where it appears, plus a suggested learning order | `services/paper_concepts.py`, `POST /papers/{id}/concept-map`, Research Map tab |
| "What should I read next?" (bridge / foundation / next-step recommendations) | `services/reading_path.py`, `POST /read-next`, Read Next tab |
| Research map: ResearchRabbit-style citation timeline (year × citations, who-cites-whom, shared foundations, TL;DRs) | `services/citations.py` (Semantic Scholar → OpenAlex fallback, cached), `POST /graph/citations`, `frontend/graph_view.py::timeline_html` |
| Grounded paper chatbot: intent-aware evidence (summary / compare / method / results / limitations), follow-up memory, streaming answers, citation repair | `services/qa/engine.py`, `services/qa/intent.py`, `llm/generator.py::stream_generate`, `POST /qa/*/stream` |
| Embedding-model change safety (indexes from another embedder are rebuilt, not mixed) | `vector_store.FaissStore(signature=...)`, `pipeline._reset_chunk_state` |

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
#    Cold start note: on first run with an empty corpus, the app
#    auto-ingests the bundled seed corpus (seed/arxiv_seed.jsonl — 46
#    canonical IR/RAG/NLP papers with condensed summary abstracts), so
#    search works immediately even if the arXiv API is unreachable or
#    rate-limiting you. Re-run it any time with:
python -m scripts.ingest --seed
python -m scripts.ingest --arxiv-query "retrieval augmented generation" --max 200
python -m scripts.ingest --categories cs.IR cs.CL --max 300
# or from the Kaggle arXiv snapshot (the deck's primary dataset):
python -m scripts.ingest --kaggle-json arxiv-metadata-oai-snapshot.json --max 5000
```

Searching in the UI with "Fetch fresh papers from arXiv" enabled ingests new
papers in real time; you don't have to pre-ingest anything.

**Use the real models.** Set `EMBEDDING_BACKEND=sentence-transformers` and
`RERANKER_ENABLED=true` (the defaults in `.env`). The `hash` embedder and a
disabled reranker match *words* only, so papers whose abstract answers the
query in different wording are missed. Switching embedders is safe: indexes
built by a different embedder are detected and rebuilt automatically.

### LLM configuration (auto-selects the strongest available)

`LLM_BACKEND=auto` (default) probes at startup and picks the best backend
present on your machine:

| Priority | Backend | What it gives you | Needs |
|---|---|---|---|
| 1 | `openai-compatible` | Strong generative answers from **Ollama** / vLLM / LM Studio (default model `llama3.1:8b`) | endpoint at `LLM_API_BASE` |
| 2 | `hf` | Local **Qwen2.5-7B-Instruct** via transformers | `requirements-ml.txt`, GPU recommended |
| 3 | `extractive` | Citation-grounded sentence synthesis (MMR selection). Cannot hallucinate. | nothing |

Recommended: install [Ollama](https://ollama.com), run `ollama pull llama3.1:8b`,
start the backend — it is detected automatically. Use `qwen2.5:14b` or
`llama3.3` via `LLM_API_MODEL` for even stronger answers.

**Grounding guard** (`LLM_GROUNDING_CHECK=true`): every generative answer is
post-validated against its retrieved sources — citations to non-existent
sources are stripped, and an answer making claims with *no* valid citation is
regenerated with the extractive backend, so unsupported claims never reach
the user. All prompts enforce a strict contract: answer only from numbered
excerpts, cite every claim as `[n]`, state explicitly when the context lacks
the answer, and structure responses as a direct answer followed by
explanation. `GET /admin/stats` shows which backend/model was resolved.

## What makes this project distinctive

### 1. It understands what you mean, and what you do not want

| You type | What happens |
|---|---|
| `paper on computer science without AI` | Field → `cat:cs.*`; **AI** excluded with its synonyms (machine learning, deep learning, neural networks, LLMs, …) and home categories (cs.AI, cs.LG, cs.CL, cs.CV, cs.NE, stat.ML). Filtered both in the live arXiv query (`ANDNOT`) and on every local candidate. |
| `graph algorithms without machine learning or deep learning` | Topic *graph algorithms*; both concepts excluded. |
| `how can robots learn new skills without forgetting old ones` | Not an exclusion: "without forgetting" **describes** the method, so it stays in the topic. |
| `methods to stop language models from making up facts` | LLM rewrite → *fact verification / factuality* terms; a pseudo-abstract (HyDE) is embedded to match real abstracts by meaning. |

The UI shows exactly how the query was understood (topic, field, exclusions,
rewrite), how many papers were filtered out and why, the arXiv query used,
and for every result the abstract sentence that made it match.

### 2. Understand a paper: concept map

Pick one paper. The map puts it in the centre, the **major topics** you need
to understand it on an inner ring (closer and bigger = more important) and
their **subtopics** branching outward. Topics come from the paper's full
text: key phrases between stop words and verbs, known methods / datasets /
metrics, and acronyms merged with their long form. Importance combines
similarity to the abstract, frequency, spread across the paper, presence in
the title or abstract, and rarity in the rest of the corpus. Each topic shows
its **role** (background you need, core idea, how it works, what's new, how
it's tested), the sentence where the paper explains it, and where it appears.
A **suggested learning order** and a study outline sit under the map, and the
LLM can add a one-line plain-language explanation per topic.

The multi-paper knowledge graph (who introduces / uses / extends which
method) remains available through `POST /graph`.

### 3. "What should I read next?"

Mark papers as read (📌). Instead of "more of the same", candidates are
scored for **bridging** your papers (harmonic-mean similarity to all of them
+ mentioning each paper's distinctive concepts), being a **foundation** (it
introduces a method your papers build on), and **novelty** (concepts you
have not covered), with near-duplicates penalised and results diversified.
A live arXiv "bridge query" asks directly for papers combining your papers'
key concepts. Every suggestion explains itself, e.g. *"Read this because it
bridges Multi-hop QA (from A) and Open-Domain QA (from B)"*, and the reading
map shows the connections.

### 4. Research map: a citation timeline

Like ResearchRabbit, but built into the search workflow: every paper is a
circle placed by **publication date** (left to right) and **citation count**
(bottom to top), sized by citations. Arrows show which paper cites which,
and dashed circles are **shared foundations**, earlier works that several
of the papers cite (e.g. BERT, Natural Questions). Clicking a paper shows
its one-line TL;DR, authors and how it connects. Data comes from Semantic
Scholar (with an OpenAlex fallback) and is cached for a week.

### 5. A grounded chatbot for one paper or several

Pick papers **by title**, click a suggestion (Summary, Contributions,
Method, Results, Limitations, Key differences…) or type any question.
Answers stream in as they are written, cite their sources as badges, keep
the conversation for follow-ups ("what about its limitations?"), and say
plainly when a paper does not cover something. Generic questions work:
the question's intent decides which sections to read, and the abstract is
always included. Comparisons come back as a table plus key differences.

### Roadmap ideas (not implemented yet)

* Research-trend timeline: how a method's usage grows over years.
* Contradiction finder: papers reporting opposite results on the same benchmark.
* Comparison tables: method × dataset × metric extracted from full texts.
* Saved searches with alerts for new arXiv papers matching a query.
* Explanation levels (beginner / expert) for QA answers; BibTeX export.

## API overview (`/api/v1`)

| Endpoint | Purpose |
|---|---|
| `POST /search` | Understand → structured arXiv fetch → hybrid retrieve → filter → rerank. Body: `query`, `top_k`, `fetch_arxiv`, `exclude` (list), `field`, `smart` (LLM rewrite). Returns results with `evidence`, plus `interpretation`, `excluded_count`, `live_query` |
| `POST /graph` | Knowledge graph for `paper_ids` or a `query` (`top_n`, `use_llm`) → nodes, edges, insights |
| `POST /graph/citations` | Citation timeline for `paper_ids` or a `query` → papers (year, citations, TL;DR), cites-edges, shared foundations |
| `POST /papers/{id}/concept-map?explain=` | Concept map of one paper → topics (importance, tier, role, evidence, sections, subtopics) and a learning order |
| `POST /qa/single/stream` · `POST /qa/multi/stream` | Streaming chat (NDJSON events: `status`, `sources`, `token`, `final`, `error`) |
| `GET /papers?q=` | Papers in the corpus for pick-by-title selectors (indexed first) |
| `POST /read-next` | `read_ids` → `you_know`, `you_are_missing`, `recommendations` (kind, headline, reasons, bridges) |
| `POST /qa/single` | Question answering over one paper (multi-turn `history` supported) |
| `POST /qa/multi` | Map-reduce QA across selected papers (or let retrieval pick) |
| `GET /papers/{id}` | Paper metadata + indexing status |
| `POST /papers/{id}/index` | On-demand download+parse+chunk+embed (auto-invoked when a paper is selected for QA) |
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
python -m pytest tests/ -q        # 122 tests, fully offline
```

Covers: query processing, dynamic-alpha fusion, BM25/dense/hybrid retrieval,
reranking, PDF parsing + structure-aware chunking (against a generated PDF),
graceful PDF failure, FAISS filtering/persistence, dedup, lazy-index
idempotency, single & multi-doc QA, conversation history, recommendations,
all API routes, the metric implementations, plus the on-demand-indexing contract (search never downloads; QA selection triggers indexing) and the citation/grounding guard.
New: negation/exclusion parsing (incl. descriptive "without"), field filters,
the structured arXiv query, "without AI" never returning AI papers, live
matches always reaching the reranker, evidence sentences, the LLM-rewrite
hook, knowledge-graph extraction and cross-paper linking, read-next
bridge/foundation ranking, and the embedder-change index reset.
Chat and map: intent detection, generic comparison questions answered (the
"key difference" failure), exact per-paper retrieval, streaming events,
citation repair, honest "not covered" answers, degenerate-output checks,
the Ollama context window, the streaming endpoints, and the citation map's
edges, shared foundations and HTML escaping.

## Docker deployment (with PostgreSQL)

```bash
docker compose up --build
# frontend :8501 · api :8000 · postgres :5432
```

## Architecture notes & design decisions

* **Two-level retrieval, strictly deferred full text.** Paper-level hybrid
  retrieval over title+abstract answers *search* from metadata alone —
  PDFs are downloaded, parsed, chunked, and embedded only at the moment a
  paper is selected for Single- or Multi-Paper QA (idempotently, so the
  cost is paid once per paper). Chunk-level FAISS retrieval with metadata
  filters then serves QA, with chunk candidates reranked before the LLM.
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
* **Fast search, visible costs.** Models (embedder + cross-encoder) are
  warmed at startup so the first query doesn't pay model-load time; the
  live arXiv fetch is capped (30 structured candidates; a relaxed second
  request only when the strict one is thin) and
  the corrective retry re-queries local indexes only. Search responses
  include a `timings_ms` breakdown (arXiv fetch vs retrieval+rerank),
  shown in the UI, so any slowness is immediately attributable. When
  the live fetch fails or times out, the response carries a `notice`
  explaining what happened and what to do (shown as a warning banner),
  and search continues over the local corpus. The live fetch is also
  polite by design: identical queries are cached for 5 minutes,
  uncached fetches are spaced ≥3s apart, and HTTP 429/503 responses
  are reported explicitly as arXiv rate limiting — repeated searches
  can no longer trigger or be stalled by rate limits.
* **Why the chatbot reads papers the way it does.** Three failures shaped
  it. (1) Relevance scoring rates every passage ~0 for questions like
  "what's the key difference?", so question *intent* now selects the
  sections to read and the abstract is always included. (2) Paper-restricted
  retrieval used post-filtered ANN search over all chunks; with hundreds of
  chunks indexed only one passage per paper survived, so retrieval is now
  exact over that paper's own chunks. (3) Ollama's OpenAI-compatible
  endpoint silently truncated prompts to ~2k tokens; the app now uses the
  native API with `num_ctx`. Prompts are kept lean (CPU prompt processing
  is ~50 tokens/s) and put the stable part first, so Ollama's prefix cache
  makes follow-up answers start in seconds.
* **Robust QA matching + diagnostics.** Extractive evidence matching
  uses stemmed content words ("compare the results" matches "the
  result shows"), map calls across papers run in parallel, and when no
  evidence is found the answer names each paper's reason (indexing
  failed vs. nothing matched) with rephrasing guidance instead of a
  blanket failure message.
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
    query_processor.py      normalization, expansion, traits, exclusions, fields, arXiv query
    query_rewriter.py       LLM rewrite of plain-language queries (+ HyDE pseudo-abstract)
    relevance.py            concept coverage, exclusion/field filters, evidence sentences
    knowledge_graph.py      entity extraction + paper knowledge graph + insights
    paper_concepts.py       single-paper concept map (topics, importance, roles, evidence)
    citations.py            citation timeline data (Semantic Scholar / OpenAlex, cached)
    reading_path.py         "What should I read next?" bridge recommendations
    retrieval/              bm25, dense, fusion (RRF/weighted/dynamic α), hybrid
    reranker.py             cross-encoder + fallback, confidence gate
    documents/              lazy loader (PyMuPDF) + structure-aware chunker
    embeddings.py           sentence-transformers + hash fallback
    vector_store.py         FAISS (papers + chunks), filtering, persistence, IVF
    llm/generator.py        extractive / HF local / OpenAI-compatible
    qa/                     intent detection, chat prompts, streaming single/multi-paper engine
    recommender.py          related-paper suggestions
    pipeline.py             end-to-end orchestrator + admin ops + query log
frontend/streamlit_app.py   7-tab UI (+ reading-list sidebar), "Aurora" theme
frontend/graph_view.py      citation timeline, concept graph, reading map (vis-network)
.streamlit/config.toml      theme (run Streamlit from the project folder to apply it)
evaluation/evaluate.py      Recall@k, MRR, nDCG across 4 strategies
scripts/ingest.py           arXiv API / Kaggle-snapshot bulk ingestion
tests/                      122 offline tests
```
