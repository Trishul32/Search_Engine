"""Streamlit frontend for the Real-Time Semantic Search Engine.

Run the API first (uvicorn backend.app.main:app), then:
    streamlit run frontend/streamlit_app.py
Set API_BASE env var if the backend is not on localhost:8000.
"""
from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

API = os.environ.get("API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="arXiv Semantic Search", page_icon="🔎", layout="wide"
)

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:16]
if "results" not in st.session_state:
    st.session_state.results = []
if "single_chat" not in st.session_state:
    st.session_state.single_chat = {}   # arxiv_id -> [turns]
if "multi_chat" not in st.session_state:
    st.session_state.multi_chat = []


def api_post(path: str, payload: dict, timeout: int = 300) -> dict | None:
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


def api_get(path: str, timeout: int = 120) -> dict | list | None:
    try:
        r = requests.get(f"{API}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


st.title("🔎 Real-Time Semantic Search Engine for Research Papers")
st.caption(
    "Hybrid BM25 + dense retrieval · cross-encoder reranking · FAISS · "
    "on-demand PDF indexing (only when you pick a paper for QA) · "
    "single & multi-document RAG QA"
)

tab_search, tab_single, tab_multi, tab_recs, tab_admin = st.tabs(
    ["Search", "📄 Single-paper QA", "📚 Multi-paper QA", "✨ Recommendations", "⚙️ Admin"]
)

# ------------------------------------------------------------------ Search
with tab_search:
    with st.form("search_form"):
        col1, col2, col3 = st.columns([6, 1, 1])
        query = col1.text_input(
            "Search query",
            placeholder="e.g. hybrid retrieval with reranking for scientific RAG",
            label_visibility="collapsed",
        )
        top_k = col2.number_input("Top-k", 1, 30, 10, label_visibility="collapsed")
        submitted = col3.form_submit_button("Search", use_container_width=True)
    _stats = api_get("/admin/stats", timeout=10)
    if _stats:
        st.caption(
            f"Local corpus: **{_stats['papers']} papers** · "
            f"{_stats['chunk_vectors']} full-text chunks indexed · "
            f"LLM: {_stats['llm_backend']}"
        )
    fetch_live = st.toggle(
        "Fetch fresh papers from arXiv API (real-time)",
        help="When off, search runs over the locally indexed corpus only.",
        value=True,
    )

    if submitted and query.strip():
        with st.spinner("Hybrid retrieval → reranking… (metadata only — no PDF downloads)"):
            data = api_post(
                "/search",
                {
                    "query": query,
                    "top_k": int(top_k),
                    "fetch_arxiv": fetch_live,
                    "session_id": st.session_state.session_id,
                },
            )
        if data:
            st.session_state.results = data["results"]
            if data.get("notice"):
                st.warning(data["notice"])
            if not data["results"]:
                st.info(
                    "No papers matched. Broaden the query, or check the "
                    "Admin tab — an empty corpus means nothing has been "
                    "ingested yet."
                )
            meta = f"{data['latency_ms']:.0f} ms"
            t = data.get("timings_ms") or {}
            if t.get("arxiv_fetch", 0) > 0:
                meta += (f" (arXiv fetch {t['arxiv_fetch']:.0f} · "
                         f"retrieval+rerank {t['retrieve_rerank']:.0f})")
            if data["expansions"]:
                meta += " · expanded with: " + ", ".join(data["expansions"])
            st.caption(meta)

    for r in st.session_state.results:
        with st.container(border=True):
            head, score = st.columns([8, 1])
            head.markdown(
                f"**[{r['title']}](https://arxiv.org/abs/{r['arxiv_id']})**"
            )
            score.metric("score", f"{r.get('score', 0):.3f}", label_visibility="collapsed")
            st.caption(
                f"arXiv:{r['arxiv_id']} · {r.get('published','')} · "
                f"{r.get('categories','')} · full text: "
                f"{'indexed ✅' if r.get('pdf_status') == 'embedded' else 'not downloaded (indexed on QA)'}"
            )
            st.write(
                r["abstract"][:600] + ("…" if len(r["abstract"]) > 600 else "")
            )
            c1, c2 = st.columns(2)
            if c1.button("💬 Ask this paper", key=f"ask_{r['arxiv_id']}"):
                st.session_state.active_paper = r["arxiv_id"]
                st.session_state.active_title = r["title"]
                st.toast("Opened in Single-paper QA tab")
            if c2.button("✨ Similar papers", key=f"rec_{r['arxiv_id']}"):
                st.session_state.rec_paper = r["arxiv_id"]
                st.toast("Opened in Recommendations tab")

# --------------------------------------------------------------- Single QA
with tab_single:
    ids = [r["arxiv_id"] for r in st.session_state.results]
    default = st.session_state.get("active_paper")
    idx = ids.index(default) if default in ids else 0
    paper_id = st.selectbox(
        "Paper (from your last search, or type an arXiv id below)",
        options=ids or ["—"],
        index=idx if ids else 0,
    )
    manual = st.text_input("…or arXiv id", placeholder="2503.23013")
    paper_id = manual.strip() or (paper_id if paper_id != "—" else "")

    if paper_id:
        # On-demand indexing: the PDF is downloaded/parsed/embedded only
        # now that this paper was selected for QA (idempotent server-side).
        if st.session_state.get("indexed_paper") != paper_id:
            with st.status(
                f"Preparing {paper_id}: downloading PDF → parsing → "
                "chunking → embedding…", expanded=False,
            ) as status_box:
                info = api_post(f"/papers/{paper_id}/index", {}, timeout=600)
                if info:
                    st.session_state.indexed_paper = paper_id
                    status_box.update(
                        label=f"Ready — {info.get('n_chunks', '?')} chunks "
                              f"indexed ({info.get('pdf_status', '')})",
                        state="complete",
                    )
        chat = st.session_state.single_chat.setdefault(paper_id, [])
        for turn in chat:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
        q = st.chat_input("Ask about this paper…")
        if q:
            chat.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"), st.spinner("Retrieving relevant chunks → generating grounded answer…"):
                data = api_post(
                    "/qa/single",
                    {
                        "question": q,
                        "arxiv_id": paper_id,
                        "history": chat[:-1],
                        "session_id": st.session_state.session_id,
                    },
                )
                if data:
                    st.markdown(data["answer"])
                    with st.expander("Sources"):
                        for s_ in data["sources"]:
                            st.markdown(
                                f"**[{s_['n']}]** *{s_['section']}* — {s_['snippet']}…"
                            )
                    chat.append({"role": "assistant", "content": data["answer"]})
    else:
        st.info("Run a search first, or paste an arXiv id.")

# ---------------------------------------------------------------- Multi QA
with tab_multi:
    st.markdown(
        "Ask a question **across several papers** — the selected papers are "
        "downloaded & indexed on demand, then the system extracts evidence "
        "per paper (map) and synthesizes one answer (reduce)."
    )
    options = {
        f"{r['title'][:80]} ({r['arxiv_id']})": r["arxiv_id"]
        for r in st.session_state.results
    }
    chosen = st.multiselect(
        "Papers (leave empty to let retrieval pick the best papers)",
        options=list(options),
    )
    for turn in st.session_state.multi_chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
    mq = st.chat_input("e.g. Compare the chunking strategies used across these papers")
    if mq:
        st.session_state.multi_chat.append({"role": "user", "content": mq})
        with st.chat_message("user"):
            st.markdown(mq)
        with st.chat_message("assistant"), st.spinner("Indexing selected papers (first time only) → map-reduce synthesis…"):
            data = api_post(
                "/qa/multi",
                {
                    "question": mq,
                    "paper_ids": [options[c] for c in chosen] or None,
                    "history": st.session_state.multi_chat[:-1],
                    "session_id": st.session_state.session_id,
                },
            )
            if data:
                st.markdown(data["answer"])
                with st.expander("Per-paper evidence"):
                    for s_ in data["sources"]:
                        st.markdown(
                            f"**[{s_['n']}] {s_['title']}** (arXiv:{s_['arxiv_id']})\n\n"
                            f"{s_['snippet']}…"
                        )
                st.session_state.multi_chat.append(
                    {"role": "assistant", "content": data["answer"]}
                )

# ---------------------------------------------------------- Recommendations
with tab_recs:
    rec_default = st.session_state.get("rec_paper", "")
    rid = st.text_input("arXiv id", value=rec_default, placeholder="2503.23013")
    if st.button("Find similar papers") and rid.strip():
        data = api_get(f"/papers/{rid.strip()}/recommendations")
        if data:
            for rec in data["recommendations"]:
                with st.container(border=True):
                    st.markdown(
                        f"**[{rec['title']}](https://arxiv.org/abs/{rec['arxiv_id']})** "
                        f"· similarity {rec['score']:.3f}"
                    )
                    st.caption(f"arXiv:{rec['arxiv_id']} · {rec.get('categories','')}")
                    st.write(rec["abstract"][:400] + "…")

# -------------------------------------------------------------------- Admin
with tab_admin:
    c1, c2 = st.columns(2)
    if c1.button("Refresh stats"):
        st.session_state.stats = api_get("/admin/stats")
    if c2.button("Rebuild indexes", type="secondary"):
        with st.spinner("Rebuilding…"):
            st.session_state.stats = api_post("/admin/rebuild", {})
    stats = st.session_state.get("stats") or api_get("/admin/stats")
    if stats:
        cols = st.columns(4)
        cols[0].metric("Papers", stats["papers"])
        cols[1].metric("Chunks", stats["chunks"])
        cols[2].metric("Paper vectors", stats["paper_vectors"])
        cols[3].metric("Chunk vectors", stats["chunk_vectors"])
        st.caption(
            f"Embedder: {stats['embedding_backend']} (dim {stats['embedding_dim']}) · "
            f"LLM backend: {stats['llm_backend']} · queries logged: {stats['queries']}"
        )
    st.subheader("Recent queries")
    hist = api_get("/admin/history?limit=25")
    if hist:
        st.dataframe(hist, use_container_width=True)
