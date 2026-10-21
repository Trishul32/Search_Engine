"""Interactive graph views (vis-network inside a sandboxed iframe).

* ``timeline_html``  — research map in the style of ResearchRabbit: each
  paper is a circle placed by publication date (x) and citation count (y),
  arrows show who cites whom, hollow circles are shared foundations.
* ``graph_html``     — concept graph: papers, methods, datasets, metrics
  and tasks with their relations.
* ``reading_map``    — small graph for Read Next.

No extra Python dependency: graph JSON is embedded in a page that loads
vis-network from a pinned CDN version. All text from papers/LLMs is
escaped before it reaches HTML; node labels are drawn on canvas.
"""
from __future__ import annotations

import json
import math

VIS_JS = "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"

from theme import T, rgba

# every colour comes from the active theme (frontend/theme.py)
FONT = "Segoe UI, system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif"
BG, PANEL, PANEL2 = T["bg"], T["surface"], T["surface2"]
TEXT, MUTED, BORDER = T["text"], T["muted"], T["border"]
PRIMARY, ACCENT, LINK = T["primary"], T["accent_text"], T["link"]
ON_PRIMARY, ON_ROLE = T["on_primary"], T["on_role"]
_R = T["roles"]
HL = _R["background"]                      # colour of a selected / hovered node

TYPE_STYLE = {
    "paper":   {"color": PRIMARY, "label": "Paper"},
    "method":  {"color": _R["method"], "label": "Method"},
    "dataset": {"color": _R["evaluation"], "label": "Dataset"},
    "metric":  {"color": _R["contribution"], "label": "Metric"},
    "task":    {"color": _R["background"], "label": "Task"},
    "read":    {"color": _R["method"], "label": "You read"},
    "suggest": {"color": _R["contribution"], "label": "Suggested"},
}
EDGE_COLOR = {
    "introduces": _R["method"], "extends": _R["contribution"], "uses": MUTED,
    "evaluated on": _R["evaluation"], "improves": _R["contribution"], "measured by": _R["contribution"],
    "addresses": _R["background"], "bridges": _R["contribution"], "builds on": _R["method"],
    "knows": MUTED, "cites": MUTED,
}

_BASE_CSS = f"""
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: "Segoe UI", Inter, system-ui, -apple-system, Roboto, sans-serif;
         background: {BG}; color:{TEXT}; }}
  #wrap {{ display:flex; gap:12px; }}
  #stage {{ position:relative; flex:1; border-radius:16px; overflow:hidden;
           border:1px solid {BORDER}; background: {BG}; }}
  /* legend + toolbar sit in a strip above the canvas, never over nodes */
  #stage {{ display:grid; grid-template-columns: minmax(0, 1fr) auto; grid-template-rows: auto minmax(0, 1fr); }}
  #net {{ grid-column: 1 / 3; grid-row: 2; width:100%; height:100%; min-height:0; }}
  #side {{ width:300px; padding:16px; border-radius:16px; border:1px solid {BORDER};
          background: {PANEL}; overflow:auto; font-size:13px; line-height:1.5; }}
  #side h4 {{ margin:6px 0 8px 0; font-size:15px; line-height:1.35; }}
  #side .muted {{ color:{MUTED}; }}
  .tag {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:11px; font-weight:600; color:{ON_ROLE}; }}
  .stat {{ display:inline-block; margin:2px 6px 2px 0; padding:2px 8px; border-radius:8px; background:{PANEL2}; font-size:12px; }}
  .tldr {{ margin:10px 0; padding:10px; border-radius:10px; background:{rgba(PRIMARY, .08)}; border-left:3px solid {PRIMARY}; }}
  a {{ color:{LINK}; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
  #toolbar {{ grid-column: 2; grid-row: 1; padding:12px 12px 6px; display:flex; gap:6px; align-self:start; }}
  #toolbar button, #toolbar input {{ background:{PANEL}; color:{TEXT}; border:1px solid {BORDER};
           border-radius:999px; padding:6px 12px; font-size:12px; cursor:pointer; backdrop-filter: blur(6px); }}
  #toolbar input {{ width:150px; cursor:text; }}
  #toolbar button:hover {{ border-color:{PRIMARY}; }}
  #legend {{ grid-column: 1; grid-row: 1; padding:12px 12px 6px; display:flex; flex-wrap:wrap; gap:6px; }}
  .lg {{ font-size:11px; padding:3px 10px; border-radius:999px; background:{PANEL}; border:1px solid {BORDER}; }}
  .lg i {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; vertical-align:middle; }}
  #fallback {{ display:none; padding:20px; color:{T["danger"]}; }}
  ul {{ padding-left:18px; margin:6px 0; }}
  @media (max-width: 760px) {{ #wrap {{ flex-direction:column; }} #side {{ width:auto; }} }}
"""

_COMMON_JS = """
function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]); }
function fmt(n) { if (n === null || n === undefined) return "n/a"; return n >= 1000 ? (n/1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n); }
function libMissing() {
  if (typeof vis !== "undefined") return false;
  document.getElementById("wrap").style.display = "none";
  document.getElementById("fallback").style.display = "block";
  return true;
}
function focusNeighbourhood(net, nodes, edges, id) {
  const keep = new Set([id, ...net.getConnectedNodes(id)]);
  nodes.update(nodes.get().map(n => ({id: n.id, opacity: keep.has(n.id) ? 1 : 0.15})));
  edges.update(edges.get().map(e => ({id: e.id, hidden: !(keep.has(e.from) && keep.has(e.to))})));
}
function resetFocus(nodes, edges) {
  nodes.update(nodes.get().map(n => ({id: n.id, opacity: 1})));
  edges.update(edges.get().map(e => ({id: e.id, hidden: false})));
}
"""


def _page(body: str, script: str, height: int) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<script src="{VIS_JS}"></script>
<style>{_BASE_CSS} #stage {{ height:{height}px; }} #side {{ max-height:{height}px; }}</style>
</head><body>{body}
<div id="fallback">Could not load the graph library from the CDN (no internet?). The table
below the graph lists the same information.</div>
<script>{_COMMON_JS}{script}</script></body></html>"""


def _data(obj) -> str:
    return json.dumps(obj).replace("</", "<\\/")


# --------------------------------------------------------------------------
# Timeline research map (ResearchRabbit-style)
# --------------------------------------------------------------------------
def _year_frac(node: dict) -> float | None:
    pub = node.get("published") or ""
    if len(pub) >= 7 and pub[:4].isdigit() and pub[5:7].isdigit():
        return int(pub[:4]) + (int(pub[5:7]) - 1) / 12
    return float(node["year"]) if node.get("year") else None


def timeline_html(cmap: dict, height: int = 640, show_foundations: bool = True) -> str:
    """Papers placed by date (x) and citations (y); arrows = 'cites'."""
    nodes_in = [n for n in cmap["nodes"] if show_foundations or n["kind"] == "paper"]
    keep = {n["id"] for n in nodes_in}
    xs = [_year_frac(n) for n in nodes_in if _year_frac(n) is not None]
    lo, hi = (min(xs), max(xs)) if xs else (2000, 2026)
    if hi - lo < 1:
        lo, hi = lo - 0.5, hi + 0.5
    cites = [n["citations"] for n in nodes_in if n.get("citations") is not None]
    cmax = math.log10(max(cites) + 1) if cites else 1.0
    cmed = sorted(cites)[len(cites) // 2] if cites else 0
    width_px, height_px = 1100.0, 560.0

    def node_size(n: dict) -> float:
        c = n.get("citations") or 0
        ratio = math.log10(c + 1) / (cmax or 1)
        # foundations are context, not the subject: keep them smaller
        return (8 + 12 * ratio) if n["kind"] == "foundation" else (10 + 20 * ratio)

    placed = []
    for n in nodes_in:
        yf = _year_frac(n) or lo
        c = n["citations"] if n.get("citations") is not None else cmed
        x = (yf - lo) / (hi - lo) * width_px
        y = height_px - (math.log10(c + 1) / (cmax or 1)) * height_px
        placed.append([x, y, n, node_size(n)])
    # relax overlaps: circles (plus their label below) must not collide.
    # Mostly vertical moves keep the time axis honest.
    for _ in range(150):
        moved = False
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                xi, yi, _, ri = placed[i]
                xj, yj, _, rj = placed[j]
                need_x, need_y = ri + rj + 70, ri + rj + 26
                dx, dy = xj - xi, yj - yi
                if abs(dx) < need_x and abs(dy) < need_y:
                    push = (need_y - abs(dy)) / 2 + 0.5
                    sign = 1 if dy >= 0 else -1
                    placed[i][1] -= sign * push
                    placed[j][1] += sign * push
                    nudge = 3 if dx >= 0 else -3
                    placed[i][0] -= nudge
                    placed[j][0] += nudge
                    moved = True
        if not moved:
            break

    years = list(range(int(math.floor(lo)), int(math.ceil(hi)) + 1))
    step = max(1, len(years) // 10)
    ticks = [{"x": (y - lo) / (hi - lo) * width_px, "label": str(y)} for y in years[::step]]
    vis_nodes = []
    for x, y, n, size in placed:
        foundation = n["kind"] == "foundation"
        c = n.get("citations") or 0
        t = ((_year_frac(n) or lo) - lo) / (hi - lo)
        # cool (older) -> warm (newer) for papers of the set
        old_c, mid_c, new_c = T["timeline"]
        color = BG if foundation else (old_c if t < 0.33 else mid_c if t < 0.66 else new_c)
        vis_nodes.append({
            "id": n["id"], "x": x, "y": y, "label": n["label"], "shape": "dot", "size": size,
            "color": {"background": color, "border": MUTED if foundation else ACCENT,
                      "highlight": {"background": HL, "border": TEXT},
                      "hover": {"background": HL, "border": TEXT}},
            "borderWidth": 2, "shapeProperties": {"borderDashes": [4, 3] if foundation else False},
            "font": {"color": TEXT, "size": 13, "face": FONT, "strokeWidth": 4, "strokeColor": BG},
            "shadow": {"enabled": not foundation, "color": rgba(PRIMARY, .4), "size": 16, "x": 0, "y": 0},
            "title": f"{n['title']} — {c if n.get('citations') is not None else 'n/a'} citations",
            "meta": n,
        })
    vis_edges = [
        {"from": e["source"], "to": e["target"], "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
         "color": {"color": rgba(MUTED, .22) if e["target"].startswith("ref:") else rgba(PRIMARY, .5),
                   "highlight": HL, "hover": HL},
         "width": 1 if e["target"].startswith("ref:") else 1.6,
         "dashes": e["target"].startswith("ref:"), "smooth": {"type": "curvedCW", "roundness": 0.08}}
        for e in cmap["edges"] if e["source"] in keep and e["target"] in keep
    ]
    old_c, mid_c, new_c = T["timeline"]
    legend = (
        f'<span class="lg"><i style="background:{old_c}"></i>older</span>'
        f'<span class="lg"><i style="background:{mid_c}"></i>middle</span>'
        f'<span class="lg"><i style="background:{new_c}"></i>newer</span>'
        f'<span class="lg"><i style="background:transparent;border:1px dashed {MUTED}"></i>shared foundation</span>'
        '<span class="lg">size = citations</span>'
    )
    body = f"""<div id="wrap"><div id="stage"><div id="legend">{legend}</div>
<div id="toolbar"><input id="q" placeholder="Find paper…"><button id="fit">Fit</button></div>
<div id="net"></div></div>
<div id="side"><span class="tag" style="background:{PRIMARY}">Timeline</span>
<p class="muted">Click a paper to see its details.</p></div></div>"""
    script = f"""
const DATA = {_data({"nodes": vis_nodes, "edges": vis_edges, "ticks": ticks, "h": height_px, "w": width_px})};
if (!libMissing()) {{
  const nodes = new vis.DataSet(DATA.nodes), edges = new vis.DataSet(DATA.edges);
  const net = new vis.Network(document.getElementById("net"), {{nodes, edges}}, {{
    physics: false, interaction: {{ hover: true, tooltipDelay: 80, dragNodes: true }},
    edges: {{ selectionWidth: 2 }},
  }});
  net.on("afterDrawing", (ctx) => {{
    const y0 = DATA.h + 70;
    ctx.save(); ctx.strokeStyle = "{rgba(MUTED, .35)}"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(-40, y0); ctx.lineTo(DATA.w + 40, y0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-40, y0); ctx.lineTo(-40, -60); ctx.stroke();
    ctx.fillStyle = "{MUTED}"; ctx.font = "13px Segoe UI, sans-serif"; ctx.textAlign = "center";
    for (const t of DATA.ticks) {{ ctx.fillText(t.label, t.x, y0 + 22);
      ctx.beginPath(); ctx.moveTo(t.x, y0 - 4); ctx.lineTo(t.x, y0 + 4); ctx.stroke(); }}
    ctx.fillStyle = "{ACCENT}"; ctx.font = "600 13px Segoe UI, sans-serif";
    ctx.fillText("More recently published  →", DATA.w / 2, y0 + 48);
    ctx.translate(-70, DATA.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("More citations  →", 0, 0);
    ctx.restore();
  }});
  net.once("afterDrawing", () => {{ net.fit({{ animation: false }});
    net.moveTo({{ scale: net.getScale() * 0.9 }}); }});
  document.getElementById("fit").onclick = () => {{ resetFocus(nodes, edges); net.fit({{ animation: true }}); }};
  document.getElementById("q").oninput = (ev) => {{
    const q = ev.target.value.toLowerCase(); if (!q) return;
    const hit = nodes.get().find(n => (n.meta.title + " " + n.label).toLowerCase().includes(q));
    if (hit) {{ net.selectNodes([hit.id]); net.focus(hit.id, {{ scale: 1.2, animation: true }}); }}
  }};
  net.on("click", (p) => {{
    if (!p.nodes.length) {{ resetFocus(nodes, edges); return; }}
    const n = nodes.get(p.nodes[0]), m = n.meta;
    focusNeighbourhood(net, nodes, edges, n.id);
    const citesOut = net.getConnectedEdges(n.id).map(id => edges.get(id)).filter(e => e.from === n.id).length;
    const citedBy = net.getConnectedEdges(n.id).map(id => edges.get(id)).filter(e => e.to === n.id).length;
    let h = `<span class="tag" style="background:${{m.kind === "paper" ? "{PRIMARY}" : "{MUTED}"}}">${{m.kind === "paper" ? "Paper" : "Shared foundation"}}</span>`;
    h += `<h4>${{esc(m.title)}}</h4>`;
    if (m.authors) h += `<div class="muted">${{esc(m.authors.split(";").slice(0, 4).join(", "))}}${{m.authors.split(";").length > 4 ? " et al." : ""}}</div>`;
    h += `<div style="margin:8px 0"><span class="stat">📅 ${{esc(m.year || "?")}}</span><span class="stat">⭐ ${{fmt(m.citations)}} citations</span>`;
    h += `<span class="stat">↗ cites ${{citesOut}} here</span><span class="stat">↙ cited by ${{citedBy}} here</span></div>`;
    if (m.tldr) h += `<div class="tldr"><b>TL;DR</b> ${{esc(m.tldr)}}</div>`;
    if (m.abstract) h += `<p class="muted">${{esc(m.abstract)}}…</p>`;
    if (m.kind === "foundation" && m.cited_by) h += `<p>Cited by ${{m.cited_by.length}} papers in this map.</p>`;
    if (m.url) h += `<p><a href="${{esc(m.url)}}" target="_blank">Open paper ↗</a></p>`;
    document.getElementById("side").innerHTML = h;
  }});
}}"""
    return _page(body, script, height)


# --------------------------------------------------------------------------
# Concept graph
# --------------------------------------------------------------------------
def graph_html(graph: dict, titles: dict[str, str] | None = None, height: int = 640) -> str:
    """Concept graph: papers and the methods/datasets/metrics/tasks they share."""
    titles = titles or {}
    nodes = []
    for n in graph["nodes"]:
        style = TYPE_STYLE.get(n["type"], TYPE_STYLE["method"])
        is_paper = n["type"] in ("paper", "read", "suggest")
        size = (22 if is_paper else 10) + min(22, 4 * len(n.get("papers", [])))
        nodes.append({
            "id": n["id"], "label": n["label"], "shape": "dot", "size": size,
            "color": {"background": style["color"], "border": TEXT if n.get("introduced_by") else style["color"],
                      "highlight": {"background": HL, "border": TEXT},
                      "hover": {"background": HL, "border": TEXT}},
            "borderWidth": 3 if n.get("introduced_by") else 1,
            "font": {"color": TEXT, "size": 13 if is_paper else 12, "face": FONT,
                     "strokeWidth": 4, "strokeColor": BG},
            "shadow": {"enabled": True, "color": style["color"] + "88", "size": 16, "x": 0, "y": 0},
            "title": n.get("title", n["label"]),
            "meta": {"type": n["type"], "title": n.get("title", n["label"]),
                     "papers": n.get("papers", []), "introduced_by": n.get("introduced_by"),
                     "url": n.get("url")},
        })
    edges = [{
        "from": e["source"], "to": e["target"], "title": e["relation"], "relation": e["relation"],
        "arrows": {"to": {"enabled": True, "scaleFactor": 0.45}},
        "color": {"color": EDGE_COLOR.get(e["relation"], MUTED), "opacity": 0.55,
                  "highlight": HL, "hover": HL},
        "width": 1 + min(3, max(0, len(e.get("papers", [])) - 1)),
        "dashes": e["relation"] in ("uses", "measured by", "knows"),
        "smooth": {"type": "continuous"},
    } for e in graph["edges"]]
    legend = "".join(
        f'<span class="lg"><i style="background:{s["color"]}"></i>{s["label"]}</span>'
        for t, s in TYPE_STYLE.items() if any(n["type"] == t for n in graph["nodes"])
    )
    body = f"""<div id="wrap"><div id="stage"><div id="legend">{legend}</div>
<div id="toolbar"><input id="q" placeholder="Find node…"><button id="phys">Freeze</button><button id="fit">Fit</button></div>
<div id="net"></div></div>
<div id="side"><span class="tag" style="background:{PRIMARY}">Map</span>
<p class="muted">Click a node to see its connections.</p></div></div>"""
    script = f"""
const DATA = {_data({"nodes": nodes, "edges": edges, "titles": titles})};
const COLORS = {_data({k: v["color"] for k, v in TYPE_STYLE.items()})};
if (!libMissing()) {{
  const nodes = new vis.DataSet(DATA.nodes), edges = new vis.DataSet(DATA.edges);
  const net = new vis.Network(document.getElementById("net"), {{nodes, edges}}, {{
    physics: {{ solver: "forceAtlas2Based", forceAtlas2Based: {{ gravitationalConstant: -95, springLength: 165, avoidOverlap: 0.8 }},
               stabilization: {{ iterations: 300 }} }},
    interaction: {{ hover: true, tooltipDelay: 80 }},
  }});
  let frozen = false;
  net.once("stabilizationIterationsDone", () => {{ net.setOptions({{ physics: false }}); frozen = true;
    document.getElementById("phys").textContent = "Animate"; }});
  document.getElementById("phys").onclick = (ev) => {{ frozen = !frozen; net.setOptions({{ physics: !frozen }});
    ev.target.textContent = frozen ? "Animate" : "Freeze"; }};
  document.getElementById("fit").onclick = () => {{ resetFocus(nodes, edges); net.fit({{ animation: true }}); }};
  document.getElementById("q").oninput = (ev) => {{
    const q = ev.target.value.toLowerCase(); if (!q) return;
    const hit = nodes.get().find(n => (n.meta.title + " " + n.label).toLowerCase().includes(q));
    if (hit) {{ net.selectNodes([hit.id]); focusNeighbourhood(net, nodes, edges, hit.id); net.focus(hit.id, {{ scale: 1.1, animation: true }}); }}
  }};
  net.on("click", (p) => {{
    if (!p.nodes.length) {{ resetFocus(nodes, edges); return; }}
    const n = nodes.get(p.nodes[0]), m = n.meta;
    focusNeighbourhood(net, nodes, edges, n.id);
    const conn = net.getConnectedEdges(n.id).map(id => edges.get(id));
    let h = `<span class="tag" style="background:${{COLORS[m.type] || "{MUTED}"}}">${{esc(m.type)}}</span><h4>${{esc(m.title)}}</h4>`;
    if (m.url) h += `<p><a href="${{esc(m.url)}}" target="_blank">Open on arXiv ↗</a></p>`;
    if (m.introduced_by) h += `<p>Introduced by <a href="https://arxiv.org/abs/${{esc(m.introduced_by)}}" target="_blank">${{esc(DATA.titles[m.introduced_by] || m.introduced_by)}}</a></p>`;
    if (m.type !== "paper" && m.papers.length) h += `<p><b>${{m.papers.length}} paper(s)</b></p><ul>` + m.papers.map(pid =>
      `<li><a href="https://arxiv.org/abs/${{esc(pid)}}" target="_blank">${{esc(DATA.titles[pid] || pid)}}</a></li>`).join("") + "</ul>";
    if (conn.length) h += "<p><b>Relations</b></p><ul>" + conn.slice(0, 25).map(e => {{
      const a = nodes.get(e.from), b = nodes.get(e.to);
      return `<li>${{esc(a ? a.label : e.from)}} <i class="muted">${{esc(e.relation)}}</i> ${{esc(b ? b.label : e.to)}}</li>`; }}).join("") + "</ul>";
    document.getElementById("side").innerHTML = h;
  }});
}}"""
    return _page(body, script, height)


def reading_map(result: dict) -> dict:
    """Small graph: read papers -> concepts -> suggested papers."""
    nodes, edges = [], []
    seen: set[str] = set()

    def add(nid: str, label: str, kind: str, **extra) -> None:
        if nid not in seen:
            seen.add(nid)
            nodes.append({"id": nid, "label": label, "type": kind, "papers": [], **extra})

    def short(t: str) -> str:
        return t if len(t) <= 40 else t[:39] + "…"

    for k in result.get("you_know", []):
        add(f"read:{k['arxiv_id']}", short(k["title"]), "read", title=k["title"],
            url=f"https://arxiv.org/abs/{k['arxiv_id']}")
    for r in result.get("recommendations", []):
        rid = f"suggest:{r['arxiv_id']}"
        add(rid, short(r["title"]), "suggest", title=r["title"],
            url=f"https://arxiv.org/abs/{r['arxiv_id']}")
        for b in r.get("bridges", []):
            cid = "concept:" + b["concept"].lower()
            add(cid, b["concept"], "task")
            edges.append({"source": f"read:{b['from_id']}", "target": cid, "relation": "knows", "papers": []})
            edges.append({"source": rid, "target": cid, "relation": "bridges", "papers": []})
        for f in r.get("foundations", []):
            for u in f.get("used_by", []):
                edges.append({"source": f"read:{u}", "target": rid, "relation": "builds on", "papers": []})
    return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------
# Single-paper concept map (radial mind map)
# --------------------------------------------------------------------------
ROLE_COLOR = dict(_R)        # background, concept, method, contribution, evaluation
ROLE_NAME = {
    "background": "Background you need", "concept": "Core idea", "method": "How it works",
    "contribution": "What's new", "evaluation": "How it's tested",
}


def _wrap(text: str, width: int) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _box(node: dict) -> tuple[float, float]:
    """Approximate rendered size of a vis box node (label lines x font)."""
    lines = node["label"].splitlines() or [""]
    size = node["font"]["size"]
    w = max(len(line) for line in lines) * size * 0.56 + 2 * node.get("margin", 8)
    h = len(lines) * size * 1.25 + 2 * node.get("margin", 8)
    return w, h


def _separate_labels(nodes: list[dict], pad: float = 10, rounds: int = 300) -> None:
    """Nudge subtopic boxes apart until none overlap (paper and major
    topics stay put, so the structure keeps its shape)."""
    sizes = {n["id"]: _box(n) for n in nodes}
    movable = [n for n in nodes if n["id"].startswith("s:")]
    for _ in range(rounds):
        moved = False
        for a in movable:
            for b in nodes:
                if a is b:
                    continue
                wa, ha = sizes[a["id"]]
                wb, hb = sizes[b["id"]]
                dx, dy = a["x"] - b["x"], a["y"] - b["y"]
                ox = (wa + wb) / 2 + pad - abs(dx)
                oy = (ha + hb) / 2 + pad - abs(dy)
                if ox > 0 and oy > 0:
                    # move along the axis that needs the smaller push
                    share = 0.5 if b["id"].startswith("s:") else 1.0
                    if ox < oy:
                        step = ox * share * (1 if dx >= 0 else -1)
                        a["x"] += step
                        if share < 1:
                            b["x"] -= step
                    else:
                        step = oy * share * (1 if dy >= 0 else -1)
                        a["y"] += step
                        if share < 1:
                            b["y"] -= step
                    moved = True
        if not moved:
            break


def concept_map_html(cmap: dict, height: int = 720) -> str:
    """The paper in the centre; major topics on an inner ring (more
    important = closer, bigger, thicker link); subtopics branch outward."""
    paper, topics = cmap["paper"], cmap["topics"]
    n = max(1, len(topics))
    r1, r2 = 215.0, 395.0
    nodes = [{
        "id": "paper", "label": _wrap(paper["title"], 30), "x": 0, "y": 0, "shape": "box",
        "margin": 16, "widthConstraint": {"maximum": 260},
        "color": {"background": PRIMARY, "border": ACCENT,
                  "highlight": {"background": PRIMARY, "border": TEXT},
                  "hover": {"background": PRIMARY, "border": TEXT}},
        "font": {"color": ON_PRIMARY, "size": 18, "face": FONT},
        "borderWidth": 2, "shapeProperties": {"borderRadius": 14},
        "shadow": {"enabled": True, "color": rgba(PRIMARY, .4), "size": 22, "x": 0, "y": 0},
        "meta": {"kind": "paper", **paper, "path": cmap.get("learning_path", [])},
    }]
    edges = []
    top = max((t["importance"] for t in topics), default=1) or 1
    for i, t in enumerate(topics):
        angle = -math.pi / 2 + 2 * math.pi * i / n
        rel = t["importance"] / top
        radius = r1 - 45 * rel                      # more important = closer
        color = ROLE_COLOR.get(t["role"], PRIMARY)
        tid = "t:" + t["id"]
        nodes.append({
            "id": tid, "label": _wrap(t["name"], 18),
            "x": radius * math.cos(angle), "y": radius * math.sin(angle),
            "shape": "box", "margin": 11, "widthConstraint": {"maximum": 170},
            "color": {"background": color, "border": color,
                      "highlight": {"background": color, "border": TEXT},
                      "hover": {"background": color, "border": TEXT}},
            "font": {"color": ON_ROLE, "size": round(15 + 4 * rel), "face": FONT},
            "borderWidth": 2, "shapeProperties": {"borderRadius": 12},
            "shadow": {"enabled": True, "color": color + "77", "size": 18, "x": 0, "y": 0},
            "meta": {"kind": "topic", **{k: v for k, v in t.items() if k != "subtopics"},
                     "subs": [s["name"] for s in t["subtopics"]]},
        })
        edges.append({"from": "paper", "to": tid, "width": 1.5 + 5 * rel,
                      "color": {"color": color, "opacity": 0.85},
                      "smooth": {"type": "curvedCW", "roundness": 0.12}})
        subs = t["subtopics"]
        sector = 2 * math.pi / n
        span = min(sector * 0.88, 0.36 * max(1, len(subs)))
        for j, s in enumerate(subs):
            offset = 0 if len(subs) == 1 else -span / 2 + span * j / (len(subs) - 1)
            a2 = angle + offset
            rr = r2 + (48 if j % 2 else 0)            # stagger to keep labels apart
            srel = s["importance"] / top
            sid = "s:" + t["id"] + ":" + s["id"]
            nodes.append({
                "id": sid, "label": _wrap(s["name"], 20),
                "x": rr * math.cos(a2), "y": rr * math.sin(a2),
                "shape": "box", "margin": 7, "widthConstraint": {"maximum": 150},
                "color": {"background": PANEL, "border": color,
                          "highlight": {"background": PANEL2, "border": TEXT},
                          "hover": {"background": PANEL2, "border": TEXT}},
                "font": {"color": TEXT, "size": round(13 + 2 * srel), "face": FONT},
                "borderWidth": 1.5, "shapeProperties": {"borderRadius": 9},
                "meta": {"kind": "subtopic", "parent": t["name"], **s},
            })
            edges.append({"from": tid, "to": sid, "width": 0.8 + 2 * srel,
                          "color": {"color": color, "opacity": 0.55},
                          "smooth": {"type": "curvedCW", "roundness": 0.1}})
    _separate_labels(nodes)
    legend = "".join(
        f'<span class="lg"><i style="background:{c}"></i>{ROLE_NAME[r]}</span>'
        for r, c in ROLE_COLOR.items()
        if any(t["role"] == r or any(s["role"] == r for s in t["subtopics"]) for t in topics)
    ) + '<span class="lg">bigger = more important</span>'
    body = f"""<div id="wrap"><div id="stage"><div id="legend">{legend}</div>
<div id="toolbar"><input id="q" placeholder="Find topic…"><button id="fit">Fit</button></div>
<div id="net"></div></div>
<div id="side"><span class="tag" style="background:{PRIMARY}">Concept map</span>
<p class="muted">Click a topic to see how the paper explains it.</p></div></div>"""
    script = f"""
const DATA = {_data({"nodes": nodes, "edges": edges})};
const ROLE = {_data({r: {"color": c, "name": ROLE_NAME[r]} for r, c in ROLE_COLOR.items()})};
if (!libMissing()) {{
  const nodes = new vis.DataSet(DATA.nodes), edges = new vis.DataSet(DATA.edges);
  const net = new vis.Network(document.getElementById("net"), {{nodes, edges}}, {{
    physics: false, interaction: {{ hover: true, tooltipDelay: 80, dragNodes: true }},
  }});
  net.once("afterDrawing", () => {{ net.fit({{ animation: false }}); net.moveTo({{ scale: net.getScale() * 0.95 }}); }});
  document.getElementById("fit").onclick = () => {{ resetFocus(nodes, edges); net.fit({{ animation: true }}); }};
  document.getElementById("q").oninput = (ev) => {{
    const q = ev.target.value.toLowerCase(); if (!q) return;
    const hit = nodes.get().find(n => n.meta.kind !== "paper" && n.meta.name.toLowerCase().includes(q));
    if (hit) {{ net.selectNodes([hit.id]); focusNeighbourhood(net, nodes, edges, hit.id);
      net.focus(hit.id, {{ scale: 1.1, animation: true }}); }}
  }};
  const pill = (text, color) => `<span class="tag" style="background:${{color}}">${{esc(text)}}</span> `;
  net.on("click", (p) => {{
    if (!p.nodes.length) {{ resetFocus(nodes, edges); return; }}
    const n = nodes.get(p.nodes[0]), m = n.meta;
    focusNeighbourhood(net, nodes, edges, n.id);
    let h = "";
    if (m.kind === "paper") {{
      h = pill("The paper", "{PRIMARY}") + `<h4>${{esc(m.title)}}</h4>`;
      if (m.authors) h += `<div class="muted">${{esc(m.authors.split(";").slice(0, 3).join(", "))}}` +
        `${{m.authors.split(";").length > 3 ? " et al." : ""}} · ${{esc((m.published || "").slice(0, 4))}}</div>`;
      if (m.summary) h += `<div class="tldr"><b>In one line</b> ${{esc(m.summary)}}</div>`;
      if (m.path && m.path.length) h += "<p><b>Suggested order to learn</b></p><ol>" + m.path.map(t =>
        `<li>${{esc(t.name)}} <span class="muted">· ${{esc(t.role_label)}}</span></li>`).join("") + "</ol>";
      h += `<p><a href="https://arxiv.org/abs/${{esc(m.arxiv_id)}}" target="_blank">Open on arXiv ↗</a></p>`;
    }} else {{
      const role = ROLE[m.role] || ROLE.concept;
      h = pill(role.name, role.color) + `<h4>${{esc(m.name)}}</h4>`;
      if (m.kind === "subtopic") h += `<div class="muted">Part of: ${{esc(m.parent)}}</div>`;
      if (m.explanation) h += `<div class="tldr"><b>In plain words</b> ${{esc(m.explanation)}}</div>`;
      if (m.evidence) h += `<p><b>How the paper puts it</b><br><span class="muted">“${{esc(m.evidence)}}”</span></p>`;
      h += `<div><span class="stat">💬 ${{m.mentions}} mentions</span>`;
      const secs = (m.sections || []).filter(s => s !== "body");
      if (secs.length) h += `<span class="stat">📍 ${{esc(secs.join(", "))}}</span>`;
      h += "</div>";
      if (m.subs && m.subs.length) h += "<p><b>Subtopics</b></p><ul>" +
        m.subs.map(s => `<li>${{esc(s)}}</li>`).join("") + "</ul>";
    }}
    document.getElementById("side").innerHTML = h;
  }});
}}"""
    return _page(body, script, height)
