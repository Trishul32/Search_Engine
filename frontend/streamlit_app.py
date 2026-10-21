"""Streamlit frontend for ScholarLens.

Run the API first (uvicorn backend.app.main:app), then from the project
folder:  streamlit run frontend/streamlit_app.py
Colours come from frontend/theme.py; API_BASE overrides the API address.
"""
from __future__ import annotations

import html
import json
import os
import re
import uuid
from string import Template

import requests
import streamlit as st
import streamlit.components.v1 as components

from graph_view import concept_map_html, graph_html, reading_map, timeline_html
from hero import hero_html
from theme import T, chip_colors, rgba

API = os.environ.get("API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(page_title="ScholarLens", page_icon="🔭", layout="wide")

# ---------------------------------------------------------------- state
_DEFAULTS = {
    "session_id": uuid.uuid4().hex[:16],
    "results": [],
    "single_chat": {},     # arxiv_id -> [turns]
    "multi_chat": {},      # "id,id" -> [turns]
    "reading_list": {},    # arxiv_id -> title
    "compare_ids": [],
    "last_search": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
# A topic button sets this; it must be applied before the search box exists.
_pending_query = st.session_state.pop("pending_query", None)
# Shareable searches: ...?q=graph+algorithms runs that search.
_url_query = st.query_params.get("q")
if not _pending_query and _url_query and st.session_state.get("_url_query_done") != _url_query:
    st.session_state["_url_query_done"] = _url_query
    _pending_query = _url_query
if _pending_query:
    st.session_state.query_input = _pending_query

FIELDS = {
    "Any field": None,
    "Computer Science": "computer science",
    "Theoretical CS": "theoretical computer science",
    "Mathematics": "mathematics",
    "Physics": "physics",
    "Statistics": "statistics",
    "Quantitative Biology": "quantitative biology",
    "Quantitative Finance": "quantitative finance",
    "Economics": "economics",
    "Electrical Engineering": "electrical engineering",
}
FIELD_TOPICS = {
    "computer science": ["graph algorithms", "compilers", "computer networks",
                         "cryptography", "databases", "distributed systems"],
    "theoretical computer science": ["approximation algorithms", "complexity theory",
                                     "automata", "graph theory"],
    "mathematics": ["number theory", "algebraic topology", "probability theory",
                    "differential equations"],
    "physics": ["condensed matter", "quantum optics", "fluid dynamics", "cosmology"],
    "statistics": ["causal inference", "bayesian methods", "time series"],
    "economics": ["market design", "game theory", "econometrics"],
}
DEFAULT_TOPICS = ["survey", "benchmark", "new methods"]
KIND_BADGE = {"Bridge": ("🌉 Bridge", "cyan"), "Foundation": ("🧱 Foundation", "green"),
              "Next step": ("➡️ Next step", "pink")}
# chip label -> the question it asks
SINGLE_SUGGESTIONS = {
    "📝 Summary": "Summarize this paper",
    "💡 Contributions": "What are the key contributions?",
    "⚙️ Method": "How does the method work?",
    "📊 Results": "What are the main results?",
    "⚠️ Limitations": "What are the limitations?",
}
MULTI_SUGGESTIONS = {
    "🔍 Key differences": "What are the key differences between these papers?",
    "⚙️ Methods": "Compare the methods of these papers",
    "📊 Results": "Compare the results of these papers",
    "🤝 In common": "What do these papers have in common?",
    "⚠️ Limitations": "Compare the limitations of these papers",
}

# ----------------------------------------------------------------- style
_R = T["roles"]
_chips = {
    "": chip_colors(T["primary"]), "cyan": chip_colors(_R["background"]),
    "pink": chip_colors(_R["contribution"]), "green": chip_colors(_R["method"]),
    "amber": chip_colors(_R["evaluation"]),
}
_chip_css = "\n".join(
    f".pill{'.' + name if name else ''} {{ background: {bg}; border: 1px solid {border}; color: {text}; }}"
    for name, (bg, border, text) in _chips.items()
)
_CSS = Template("""
<style>
.block-container { padding-top: 3.4rem; max-width: 1240px; }
header[data-testid="stHeader"] { background: transparent; }
.hero { padding: 4px 0 16px; }
.hero h1 { font-size: 2rem; line-height: 1.15; margin: 0 0 .2rem 0; padding: 0; color: $title; }
.hero p { color: $muted; margin: 0; font-size: 1rem; }
.pill { display: inline-block; padding: 3px 12px; margin: 4px 6px 0 0; border-radius: 999px;
  font-size: .8rem; font-weight: 500; }
$chip_css
.paper-title a { font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 600;
  color: $title !important; text-decoration: none; }
.paper-title a:hover { color: $link !important; }
.meta { color: $muted; font-size: .84rem; margin: 2px 0 6px; }
.why { border-left: 3px solid $primary; background: $why_bg; padding: .5rem .8rem;
  border-radius: 8px; margin: .4rem 0 .6rem; font-size: .93rem; color: $text; }
.why .lbl { color: $accent_text; font-weight: 600; margin-right: 4px; }
.understood { margin: 4px 0 10px; }
sup.cite { background: $cite_bg; color: $accent_text; border-radius: 6px; padding: 1px 5px;
  font-size: .68rem; margin-left: 2px; font-weight: 600; }
.thinking { color: $accent_text; font-size: .92rem; animation: pulse 1.4s ease-in-out infinite; }
@keyframes pulse { 0%, 100% { opacity: .5 } 50% { opacity: 1 } }
.paper-head { padding: 12px 16px; border-radius: 12px; border: 1px solid $border;
  border-left: 4px solid $primary; background: $surface; margin-bottom: 10px; }
.paper-head .t { font-family: 'Space Grotesk', sans-serif; font-size: 1.12rem; font-weight: 600; color: $title; }
.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600;
  margin: 6px 0 4px; color: $title; }
div[data-testid="stVerticalBlockBorderWrapper"] { transition: border-color .2s ease; }
div[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: $hover_border !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; padding: 6px 16px; border: 1px solid $border; }
.stTabs [aria-selected="true"] { background: transparent; border-color: $primary !important; }
.stTabs [aria-selected="true"] p { color: $accent_text !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs button[role="tab"] { border-radius: 999px; padding: 6px 16px; }
[data-testid="stChatMessage"] { background: $surface; border: 1px solid $border; border-radius: 14px; }
.stButton > button:hover { border-color: $primary; color: $accent_text; }
.stButton > button[kind="primary"], .stFormSubmitButton > button {
  background: $primary; border: none; color: $on_primary; font-weight: 600; }
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover { color: $on_primary; opacity: .9; }
</style>
""").substitute(
    title=T["title"], muted=T["muted"], link=T["link"], primary=T["primary"], text=T["text"],
    accent_text=T["accent_text"], border=T["border"], surface=T["surface"], on_primary=T["on_primary"],
    why_bg=rgba(T["primary"], 0.07), cite_bg=rgba(T["primary"], 0.18),
    hover_border=rgba(T["primary"], 0.55), chip_css=_chip_css,
)
st.markdown(_CSS, unsafe_allow_html=True)
if T.get("effects") == "glass":
    st.markdown(Template("""<style>
.stApp { background:
  radial-gradient(1100px 700px at 85% -10%, rgba(168,85,247,.30), transparent 60%),
  radial-gradient(900px 600px at -10% 110%, rgba(236,72,153,.16), transparent 60%), $bg;
  background-attachment: fixed; }
[data-testid="stSidebar"] { background: rgba(17,13,39,.72); backdrop-filter: blur(18px);
  border-right: 1px solid rgba(255,255,255,.06); }
div[class*="st-key-card"] { background: rgba(255,255,255,.04); backdrop-filter: blur(14px);
  border-radius: 18px; }
div[class*="st-key-card"] > div { border-color: rgba(255,255,255,.10) !important; }
[data-testid="stChatMessage"] { background: rgba(255,255,255,.045); backdrop-filter: blur(12px);
  border-color: rgba(255,255,255,.09); }
.stTabs [data-baseweb="tab"] { background: rgba(255,255,255,.04); border-color: rgba(255,255,255,.12); }
.stButton > button { background: rgba(255,255,255,.05); border-color: rgba(255,255,255,.14); }
.stButton > button[kind="primary"], .stFormSubmitButton > button {
  background: linear-gradient(90deg, $primary, $pink); color: $on_primary; }
.why { background: rgba(255,255,255,.05); }
.paper-head { background: rgba(255,255,255,.05); backdrop-filter: blur(12px); }
</style>""").substitute(bg=T["bg"], primary=T["primary"], pink=T["roles"]["contribution"],
                        on_primary=T["on_primary"]), unsafe_allow_html=True)


# --------------------------------------------------------------- helpers
def _version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:2])


# Streamlit >= 1.46 sizes elements with width="stretch"; older versions
# only know use_container_width.
STRETCH = (
    {"width": "stretch"} if _version(st.__version__) >= (1, 46)
    else {"use_container_width": True}
)


def show_html(page: str, height: int) -> None:
    """Render generated HTML in the sandboxed components iframe (st.iframe
    only where components.html no longer exists)."""
    if hasattr(components, "html"):
        components.html(page, height=height, scrolling=False)
    else:  # pragma: no cover - future Streamlit
        st.iframe(page, height=height)


def api_post(path: str, payload: dict, timeout: int = 300) -> dict | None:
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


def api_get(path: str, timeout: int = 120, quiet: bool = False) -> dict | list | None:
    try:
        r = requests.get(f"{API}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        if not quiet:
            st.error(f"API error: {exc}")
        return None


@st.cache_data(ttl=20, show_spinner=False)
def corpus_papers() -> list[dict]:
    return api_get("/papers?limit=3000", timeout=20, quiet=True) or []


def esc(text: str) -> str:
    return html.escape(text or "", quote=True)


def md_safe(text: str) -> str:
    """Escape characters Streamlit markdown would interpret ($ = LaTeX)."""
    return (text or "").replace("$", "\\$").replace("*", "\\*")


def answer_html(answer: str) -> str:
    """Render an LLM answer: escape HTML, keep markdown, citations as badges."""
    safe = html.escape(answer or "", quote=False).replace("$", "\\$")
    return re.sub(r"\[(\d{1,2})\]", r"<sup class='cite'>\1</sup>", safe)


def byline(p: dict) -> str:
    authors = [a.strip() for a in (p.get("authors") or "").split(";") if a.strip()]
    who = authors[0] + (" et al." if len(authors) > 1 else "") if authors else ""
    year = (p.get("published") or "")[:4]
    return " · ".join(x for x in (who, year) if x)


def paper_label(p: dict, n: int = 90) -> str:
    title = p["title"] if len(p["title"]) <= n else p["title"][: n - 1] + "…"
    extra = byline(p)
    return title + (f"  —  {extra}" if extra else "")


def paper_catalog() -> dict[str, dict]:
    """id -> paper: last search results first, then the whole corpus."""
    catalog: dict[str, dict] = {}
    for p in st.session_state.results + corpus_papers():
        catalog.setdefault(p["arxiv_id"], p)
    return catalog


def mark_read(arxiv_id: str, title: str) -> None:
    st.session_state.reading_list[arxiv_id] = title
    st.toast("Added to your reading list")


def pills(items: list[tuple[str, str]]) -> str:
    return "".join(f"<span class='pill {cls}'>{text}</span>" for text, cls in items)


def stream_answer(path: str, payload: dict, status_slot, answer_slot) -> dict:
    """Stream a chat answer, then let the caller swap in the checked final text."""
    out: dict = {"sources": [], "final": None, "error": None}

    def tokens():
        with requests.post(f"{API}{path}", json=payload, stream=True, timeout=(10, 900)) as r:
            r.raise_for_status()
            r.encoding = "utf-8"
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                ev = json.loads(line)
                kind = ev.get("type")
                if kind == "status":
                    status_slot.markdown(f"<div class='thinking'>{esc(ev['text'])}</div>",
                                         unsafe_allow_html=True)
                elif kind == "sources":
                    out["sources"] = ev["sources"]
                elif kind == "token":
                    status_slot.empty()
                    yield ev["text"]
                elif kind == "final":
                    out["final"] = ev
                elif kind == "error":
                    out["error"] = ev["text"]

    try:
        with answer_slot.container():
            st.write_stream(tokens())
    except requests.RequestException as exc:
        out["error"] = f"Could not reach the API: {exc}"
    status_slot.empty()
    return out


def render_sources(sources: list[dict], multi: bool) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            where = f"{s['title'][:70]} · {s['section']}" if multi else s["section"].capitalize()
            st.markdown(
                f"<sup class='cite'>{s['n']}</sup> <b>{esc(where)}</b><br>"
                f"<span class='meta'>{esc(s['snippet'])}…</span>",
                unsafe_allow_html=True,
            )


def chat_ui(history: list[dict], endpoint: str, payload_fn, suggestions: dict[str, str],
            key: str, placeholder: str, multi: bool) -> None:
    """Conversation, then suggestion chips and the input box. New turns are
    drawn into the conversation container so they appear above the input."""
    convo = st.container()
    with convo:
        for turn in history:
            with st.chat_message(turn["role"], avatar="🧑‍🎓" if turn["role"] == "user" else "🔭"):
                if turn["role"] == "assistant":
                    st.markdown(answer_html(turn["content"]), unsafe_allow_html=True)
                    render_sources(turn.get("sources", []), multi)
                else:
                    st.markdown(md_safe(turn["content"]))
    clicked = None
    cols = st.columns(len(suggestions))
    for col, (label, question_text) in zip(cols, suggestions.items()):
        if col.button(label, key=f"sug_{key}_{label}", help=question_text, **STRETCH):
            clicked = question_text
    typed = st.chat_input(placeholder, key=f"input_{key}")
    question = typed or clicked
    if not question:
        return
    history.append({"role": "user", "content": question})
    with convo, st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(md_safe(question))
    with convo, st.chat_message("assistant", avatar="🔭"):
        status_slot, answer_slot = st.empty(), st.empty()
        api_history = [{"role": t["role"], "content": t["content"]} for t in history[:-1]]
        result = stream_answer(endpoint, payload_fn(question, api_history), status_slot, answer_slot)
        if result["error"] and not result["final"]:
            answer_slot.error(result["error"])
            history.pop()
            return
        final = result["final"]["answer"]
        answer_slot.markdown(answer_html(final), unsafe_allow_html=True)
        render_sources(result["sources"], multi)
    history.append({"role": "assistant", "content": final, "sources": result["sources"]})


# --------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 📚 Reading list")
    if not st.session_state.reading_list:
        st.caption("No papers yet.")
    for pid, title in list(st.session_state.reading_list.items()):
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"[{md_safe(title[:70])}](https://arxiv.org/abs/{pid})")
        if c2.button("✕", key=f"rm_{pid}", help="Remove"):
            st.session_state.reading_list.pop(pid, None)
            st.rerun()
    with st.expander("Add by arXiv id"):
        new_ids = st.text_input("arXiv ids", placeholder="2004.04906, 2005.11401")
        if st.button("Add") and new_ids.strip():
            for pid in re.split(r"[\s,]+", new_ids.strip()):
                if pid:
                    st.session_state.reading_list.setdefault(pid, pid)
            st.rerun()
    if st.session_state.reading_list:
        st.download_button("Export list", json.dumps(st.session_state.reading_list, indent=2),
                           file_name="reading_list.json", mime="application/json")
    if st.session_state.compare_ids:
        st.markdown("### ⚖️ To compare")
        _cat = paper_catalog()
        for pid in st.session_state.compare_ids:
            st.caption(f"• {_cat.get(pid, {'title': pid})['title'][:70]}")
        if st.button("Clear"):
            st.session_state.compare_ids = []
            st.rerun()

# ------------------------------------------------------------------ header
_stats = api_get("/admin/stats", timeout=10, quiet=True)
st.html(hero_html(compact=True))
if _stats is None:
    st.error(f"Can't reach the API at {API}. Start it with: uvicorn backend.app.main:app --port 8000")

st.html('<div id="search-start"></div>')
(tab_search, tab_map, tab_next, tab_single, tab_multi, tab_recs, tab_admin) = st.tabs([
    "🔎 Search", "🗺️ Research Map", "🧭 Read Next", "💬 Chat",
    "⚖️ Compare", "✨ Similar", "⚙️ Admin",
])

# ------------------------------------------------------------------ Search
with tab_search:
    with st.form("search_form", border=False):
        col1, col2 = st.columns([7, 1])
        query = col1.text_input("Search", key="query_input", placeholder="Search papers…",
                                label_visibility="collapsed")
        submitted = col2.form_submit_button("Search", **STRETCH)
        with st.expander("Filters"):
            f1, f2, f3 = st.columns([3, 3, 1])
            exclude_text = f1.text_input("Exclude topics", placeholder="blockchain, LLMs")
            field_label = f2.selectbox("Field", list(FIELDS))
            top_k = f3.number_input("Results", 1, 30, 10)
            o1, o2 = st.columns(2)
            fetch_live = o1.toggle("Fetch new papers from arXiv", value=True)
            smart = o2.toggle("Rewrite plain-language queries", value=True)

    if (submitted or _pending_query) and query.strip():
        with st.spinner("Searching…"):
            data = api_post("/search", {
                "query": query, "top_k": int(top_k), "fetch_arxiv": fetch_live,
                "session_id": st.session_state.session_id,
                "exclude": [x.strip() for x in exclude_text.split(",") if x.strip()],
                "field": FIELDS[field_label], "smart": smart,
            })
        if data:
            st.session_state.results = data["results"]
            st.session_state.last_search = data
            st.session_state["_url_query_done"] = query
            st.query_params["q"] = query

    data = st.session_state.last_search
    terms: list[str] = []
    if data:
        interp = data.get("interpretation") or {}
        chips: list[tuple[str, str]] = []
        if interp.get("topic"):
            chips.append((esc(interp["topic"]), ""))
        for f in interp.get("fields", []):
            chips.append((esc(f["label"]), "cyan"))
        for e in interp.get("excluded", []):
            chips.append((f"without {esc(e['label'])}", "pink"))
        if interp.get("year"):
            chips.append((str(interp["year"]), "amber"))
        if interp.get("recent") or interp.get("browse"):
            chips.append(("newest first", "green"))
        if chips:
            st.markdown(f"<div class='understood'>{pills(chips)}</div>", unsafe_allow_html=True)
        if data.get("notice"):
            st.warning(data["notice"])
        if interp.get("browse"):
            field_name = interp["fields"][0]["label"] if interp.get("fields") else "this field"
            st.info(f"No topic given, so these are the newest papers in {field_name}. Add a topic:")
            topics = FIELD_TOPICS.get(field_name, DEFAULT_TOPICS)
            excl = " ".join(f"without {e['label']}" for e in interp.get("excluded", []))
            for col, topic in zip(st.columns(len(topics)), topics):
                if col.button(topic, key=f"topic_{topic}"):
                    st.session_state.pending_query = f"{topic} in {field_name} {excl}".strip()
                    st.rerun()
        if not data["results"]:
            st.info("No papers matched. Try a broader query.")
        rw = interp.get("rewrite")
        terms = " ".join(interp.get("concepts", []) + [interp.get("topic") or ""]).split()
        if rw:
            terms += " ".join(rw.get("keywords", [])).split()

    def highlight(sentence: str) -> str:
        out = esc(sentence)
        for t in sorted({t for t in terms if len(t) > 2}, key=len, reverse=True):
            out = re.sub(rf"(?i)(?<![\w>])({re.escape(esc(t))}\w*)", r"<b>\1</b>", out)
        return out

    for r in st.session_state.results:
        with st.container(border=True, key=f"card_r_{r['arxiv_id']}"):
            st.markdown(
                f"<div class='paper-title'><a href='https://arxiv.org/abs/{esc(r['arxiv_id'])}' target='_blank'>"
                f"{esc(r['title'])}</a></div><div class='meta'>{esc(byline(r))}</div>",
                unsafe_allow_html=True,
            )
            if r.get("evidence"):
                st.markdown(f"<div class='why'><span class='lbl'>Why it matched</span>"
                            f"{highlight(r['evidence'])}</div>", unsafe_allow_html=True)
            abstract = r["abstract"]
            st.write(md_safe(abstract[:300]) + ("…" if len(abstract) > 300 else ""))
            if len(abstract) > 300:
                with st.expander("Full abstract"):
                    st.write(md_safe(abstract))
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("💬 Chat", key=f"ask_{r['arxiv_id']}", **STRETCH):
                st.session_state.active_paper = r["arxiv_id"]
                st.toast("Opened in the Chat tab")
            queued = r["arxiv_id"] in st.session_state.compare_ids
            if c2.button("⚖️ Added" if queued else "⚖️ Compare", key=f"cmp_{r['arxiv_id']}",
                         disabled=queued, **STRETCH):
                st.session_state.compare_ids.append(r["arxiv_id"])
                st.toast("Added to Compare")
                st.rerun()
            if c3.button("✨ Similar", key=f"rec_{r['arxiv_id']}", **STRETCH):
                st.session_state.rec_paper = r["arxiv_id"]
                st.toast("Opened in the Similar tab")
            in_list = r["arxiv_id"] in st.session_state.reading_list
            if c4.button("✅ Read" if in_list else "📌 Mark as read", key=f"read_{r['arxiv_id']}",
                         disabled=in_list, **STRETCH):
                mark_read(r["arxiv_id"], r["title"])
                st.rerun()

# ------------------------------------------------------------ Research map
with tab_map:
    view = st.segmented_control(
        "View", ["🧠 Understand a paper", "🗺️ Timeline of papers"],
        default="🧠 Understand a paper", key="map_view", label_visibility="collapsed",
    ) or "🧠 Understand a paper"

    if view == "🧠 Understand a paper":
        _cat = paper_catalog()
        if not _cat:
            st.info("No papers yet. Run a search first.")
        else:
            ids = list(_cat)
            active = st.session_state.get("active_paper")
            c1, c2 = st.columns([3, 1])
            map_paper = c1.selectbox(
                "Paper", ids, index=ids.index(active) if active in ids else 0,
                format_func=lambda i: paper_label(_cat[i]), key="concept_pick",
            )
            explain = c2.toggle("Explain simply", value=False)
            if st.button("Build concept map", type="primary"):
                with st.spinner("Building…"):
                    st.session_state.concept_map = api_post(
                        f"/papers/{map_paper}/concept-map?explain={'true' if explain else 'false'}", {},
                        timeout=900)
            cm = st.session_state.get("concept_map")
            if cm:
                if cm.get("notice"):
                    st.info(cm["notice"])
                if not cm.get("topics"):
                    st.warning("Not enough topics found in this paper.")
                else:
                    show_html(concept_map_html(cm), height=740)
                    left, right = st.columns([2, 3])
                    with left, st.container(border=True, key="card_learn"):
                        st.markdown("<div class='section-title'>Learning order</div>", unsafe_allow_html=True)
                        st.markdown("\n".join(
                            f"{n}. **{md_safe(t['name'])}** · {t['role_label']}"
                            for n, t in enumerate(cm["learning_path"], start=1)))
                    with right, st.container(border=True, key="card_outline"):
                        st.markdown("<div class='section-title'>Outline</div>", unsafe_allow_html=True)
                        for t in cm["topics"]:
                            with st.expander(f"{t['name']}  ·  {t['role_label']}"):
                                if t.get("explanation"):
                                    st.markdown(md_safe(t["explanation"]))
                                if t.get("evidence"):
                                    st.markdown(f"> {md_safe(t['evidence'])}")
                                for s in t["subtopics"]:
                                    line = f"- **{md_safe(s['name'])}**"
                                    if s.get("explanation"):
                                        line += f": {md_safe(s['explanation'])}"
                                    st.markdown(line)
    else:
        g1, g2 = st.columns([3, 1])
        source = g1.radio("Papers", ["Last search results", "Reading list", "A query", "Pick papers"],
                          horizontal=True)
        top_n = g2.slider("How many", 3, 25, 12)
        graph_query, graph_ids = "", []
        if source == "A query":
            graph_query = st.text_input("Query", placeholder="retrieval augmented generation")
        elif source == "Pick papers":
            _cat = paper_catalog()
            graph_ids = st.multiselect("Papers", list(_cat), format_func=lambda i: paper_label(_cat[i], 80))
        if st.button("Build timeline", type="primary"):
            payload: dict = {"top_n": int(top_n)}
            if source == "Last search results":
                payload["paper_ids"] = [r["arxiv_id"] for r in st.session_state.results][:top_n]
            elif source == "Reading list":
                payload["paper_ids"] = list(st.session_state.reading_list)[:top_n]
            elif source == "Pick papers":
                payload["paper_ids"] = graph_ids
            else:
                payload["query"] = graph_query
            if not payload.get("paper_ids") and not payload.get("query"):
                st.warning("Nothing to map yet. Run a search or pick papers.")
            else:
                with st.spinner("Building…"):
                    st.session_state.cmap = api_post("/graph/citations", payload, timeout=300)
        cmap = st.session_state.get("cmap")
        if cmap:
            if cmap.get("notice"):
                st.warning(cmap["notice"])
            papers = [n for n in cmap["nodes"] if n["kind"] == "paper"]
            if not papers:
                st.info("No papers to map.")
            else:
                show_f = st.toggle("Show shared foundations", value=True)
                show_html(timeline_html(cmap, show_foundations=show_f), height=680)
                with st.expander("List view"):
                    st.dataframe(
                        [{"Paper": n["title"], "Year": n.get("year"), "Citations": n.get("citations"),
                          "TL;DR": n.get("tldr", "")}
                         for n in sorted(papers, key=lambda n: -(n.get("citations") or 0))],
                        **STRETCH, hide_index=True,
                    )

# --------------------------------------------------------------- Read next
with tab_next:
    reading = st.session_state.reading_list
    if not reading:
        st.info("Mark papers as read in Search to get suggestions.")
    else:
        n1, n2, n3 = st.columns(3)
        k_next = n1.slider("Suggestions", 1, 10, 5)
        live_next = n2.toggle("Search arXiv", value=True)
        llm_next = n3.toggle("Explain with AI", value=False)
        if st.button("🧭 What should I read next?", type="primary"):
            with st.spinner("Finding papers…"):
                st.session_state.read_next = api_post(
                    "/read-next", {"read_ids": list(reading), "top_k": int(k_next),
                                   "fetch_arxiv": live_next, "use_llm": llm_next}, timeout=900)
        res = st.session_state.get("read_next")
        if res:
            if res.get("notice"):
                st.warning(res["notice"])
            for k in res.get("you_know", []):
                if st.session_state.reading_list.get(k["arxiv_id"]) == k["arxiv_id"]:
                    st.session_state.reading_list[k["arxiv_id"]] = k["title"]
            left, right = st.columns(2)
            with left, st.container(border=True, key="card_know"):
                st.markdown("<div class='section-title'>You know</div>", unsafe_allow_html=True)
                known = sorted({c for k in res.get("you_know", []) for c in k["concepts"]})
                st.markdown(pills([(esc(c), "green") for c in known]) or "—", unsafe_allow_html=True)
            with right, st.container(border=True, key="card_miss"):
                st.markdown("<div class='section-title'>You're missing</div>", unsafe_allow_html=True)
                st.markdown(pills([(esc(m["concept"]), "pink") for m in res.get("you_are_missing", [])]) or "—",
                            unsafe_allow_html=True)
            for r in res.get("recommendations", []):
                with st.container(border=True, key=f"card_n_{r['arxiv_id']}"):
                    badge, cls = KIND_BADGE.get(r["kind"], (r["kind"], ""))
                    st.markdown(
                        f"<span class='pill {cls}'>{badge}</span>"
                        f"<div class='paper-title'><a href='https://arxiv.org/abs/{esc(r['arxiv_id'])}' "
                        f"target='_blank'>{esc(r['title'])}</a></div>"
                        f"<div class='why'>{esc(r.get('llm_headline') or r['headline'])}</div>",
                        unsafe_allow_html=True,
                    )
                    with st.expander("Abstract"):
                        st.write(md_safe(r["abstract"]))
                    b1, b2 = st.columns(2)
                    if b1.button("📌 Mark as read", key=f"nx_read_{r['arxiv_id']}",
                                 disabled=r["arxiv_id"] in reading, **STRETCH):
                        mark_read(r["arxiv_id"], r["title"])
                        st.rerun()
                    if b2.button("💬 Chat", key=f"nx_ask_{r['arxiv_id']}", **STRETCH):
                        st.session_state.active_paper = r["arxiv_id"]
                        st.toast("Opened in the Chat tab")
            rmap = reading_map(res)
            if rmap["edges"]:
                titles = {k["arxiv_id"]: k["title"] for k in res.get("you_know", [])}
                titles.update({r["arxiv_id"]: r["title"] for r in res.get("recommendations", [])})
                show_html(graph_html(rmap, titles, height=480), height=520)

# -------------------------------------------------------------------- Chat
with tab_single:
    catalog = paper_catalog()
    active = st.session_state.get("active_paper")
    if active and active not in catalog:
        info = api_get(f"/papers/{active}", quiet=True)
        if info:
            catalog = {active: info, **catalog}
    if not catalog:
        st.info("No papers yet. Run a search first.")
    else:
        ids = list(catalog)
        if st.session_state.get("single_pick") not in ids or (active and st.session_state.get("single_pick") != active):
            st.session_state.single_pick = active if active in ids else ids[0]
        paper_id = st.selectbox("Paper", ids, format_func=lambda i: paper_label(catalog[i]),
                                key="single_pick")
        st.session_state.active_paper = paper_id
        p = catalog[paper_id]
        # index the PDF as soon as a paper is picked, so the first answer is faster
        if st.session_state.get("indexed_paper") != paper_id:
            with st.status("Preparing paper…", expanded=False) as box:
                info = api_post(f"/papers/{paper_id}/index", {}, timeout=600)
                if info:
                    st.session_state.indexed_paper = paper_id
                    box.update(label="Ready", state="complete")
        st.markdown(
            f"<div class='paper-head'><div class='t'>{esc(p['title'])}</div>"
            f"<div class='meta'>{esc(byline(p))} · <a href='https://arxiv.org/abs/{esc(paper_id)}' "
            f"target='_blank'>arXiv ↗</a></div></div>",
            unsafe_allow_html=True,
        )
        history = st.session_state.single_chat.setdefault(paper_id, [])
        if history and st.button("New chat", key="clear_single"):
            st.session_state.single_chat[paper_id] = []
            st.rerun()
        chat_ui(
            history, "/qa/single/stream",
            lambda q, h: {"question": q, "arxiv_id": paper_id, "history": h,
                          "session_id": st.session_state.session_id},
            SINGLE_SUGGESTIONS, key=f"single_{paper_id}",
            placeholder="Ask about this paper…", multi=False,
        )

# ----------------------------------------------------------------- Compare
with tab_multi:
    catalog = paper_catalog()
    if "multi_pick" not in st.session_state or set(st.session_state.multi_pick) != set(st.session_state.compare_ids):
        st.session_state.multi_pick = [i for i in st.session_state.compare_ids if i in catalog][:6]
    chosen = st.multiselect(
        "Papers (up to 6)", list(catalog), max_selections=6,
        format_func=lambda i: paper_label(catalog[i], 80), key="multi_pick",
    )
    st.session_state.compare_ids = list(chosen)
    chat_key = ",".join(sorted(chosen)) or "auto"
    history = st.session_state.multi_chat.setdefault(chat_key, [])
    if history and st.button("New chat", key="clear_multi"):
        st.session_state.multi_chat[chat_key] = []
        st.rerun()
    chat_ui(
        history, "/qa/multi/stream",
        lambda q, h: {"question": q, "paper_ids": list(chosen) or None, "history": h,
                      "session_id": st.session_state.session_id},
        MULTI_SUGGESTIONS, key=f"multi_{chat_key}",
        placeholder="Ask about these papers…", multi=True,
    )

# ----------------------------------------------------------------- Similar
with tab_recs:
    catalog = paper_catalog()
    if catalog:
        ids = list(catalog)
        rec_default = st.session_state.get("rec_paper")
        rid = st.selectbox("Paper", ids, index=ids.index(rec_default) if rec_default in ids else 0,
                           format_func=lambda i: paper_label(catalog[i]), key="rec_pick")
        if st.button("Find similar", type="primary"):
            st.session_state.rec_results = api_get(f"/papers/{rid}/recommendations?top_k=8")
        recs = st.session_state.get("rec_results")
        if recs:
            for rec in recs["recommendations"]:
                closeness = (("Very similar", "") if rec["score"] >= 0.75 else
                             ("Closely related", "cyan") if rec["score"] >= 0.55 else ("Related", "green"))
                with st.container(border=True, key=f"card_s_{rec['arxiv_id']}"):
                    st.markdown(
                        f"<span class='pill {closeness[1]}'>{closeness[0]}</span>"
                        f"<div class='paper-title'><a href='https://arxiv.org/abs/{esc(rec['arxiv_id'])}' "
                        f"target='_blank'>{esc(rec['title'])}</a></div>",
                        unsafe_allow_html=True,
                    )
                    st.write(md_safe(rec["abstract"][:300]) + "…")

# ------------------------------------------------------------------- Admin
with tab_admin:
    c1, c2 = st.columns(2)
    if c1.button("Refresh stats"):
        st.session_state.stats = api_get("/admin/stats")
    if c2.button("Rebuild indexes", type="secondary"):
        with st.spinner("Rebuilding…"):
            st.session_state.stats = api_post("/admin/rebuild", {})
    stats = st.session_state.get("stats") or _stats
    if stats:
        cols = st.columns(4)
        cols[0].metric("Papers", stats["papers"])
        cols[1].metric("Passages", stats["chunks"])
        cols[2].metric("Queries", stats["queries"])
        cols[3].metric("LLM", stats.get("llm_model", "") or stats["llm_backend"])
    st.subheader("Recent queries")
    hist = api_get("/admin/history?limit=25")
    if hist:
        st.dataframe(hist, **STRETCH)

# ------------------------------------------------------- how to use (end)
st.html(hero_html())
