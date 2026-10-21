"""ScholarLens hero: a glass lens bringing a stream of papers into focus.

Pure HTML + CSS (no JavaScript): rendered with st.html so it lives in the
page itself and its button can jump to the search box. Paper cards drift
behind a frosted glass lens; under the lens the same cards are shown
magnified and sharp, outside it they stay soft, which is the product idea
in one picture. Motion stops for viewers who prefer reduced motion.
"""
from __future__ import annotations

import html

from theme import T

# real papers from the corpus, shown on the drifting cards
_PAPERS = [
    ("2005.11401", "2020", "Retrieval-Augmented Generation for Knowledge-Intensive NLP", "RAG"),
    ("2004.04906", "2020", "Dense Passage Retrieval for Open-Domain Question Answering", "Retrieval"),
    ("1703.10023", "2017", "Engineering DFS-Based Graph Algorithms", "Graphs"),
    ("2503.16692", "2025", "Limits of Trust in Medical AI", "Trust"),
    ("2602.10527", "2026", "AI-PACE: Integrating AI into Medical Education", "Education"),
    ("2310.11511", "2023", "Self-RAG: Learning to Retrieve, Generate and Critique", "LLMs"),
    ("2002.08909", "2020", "REALM: Retrieval-Augmented Language Model Pre-Training", "Pre-training"),
    ("2312.10997", "2023", "Retrieval-Augmented Generation for LLMs: A Survey", "Survey"),
    ("2106.08433", "2021", "Combining Lexical and Dense Retrieval for Multi-hop QA", "Hybrid search"),
]

# scene geometry (px): lens centre and radius, shared by every layer
_CX, _CY, _R = 230, 215, 122


def _card(pid: str, year: str, title: str, tag: str) -> str:
    return (f'<div class="pc"><span class="pc-id">arXiv {html.escape(pid)} · {year}</span>'
            f'<b>{html.escape(title)}</b><i></i><i class="s"></i><em>{html.escape(tag)}</em></div>')


def _stream() -> str:
    cols = [_PAPERS[0::3], _PAPERS[1::3], _PAPERS[2::3]]
    out = []
    for n, col in enumerate(cols):
        cards = "".join(_card(*p) for p in col)
        out.append(f'<div class="col c{n}"><div class="track">{cards}{cards}</div></div>')
    return f'<div class="stream">{"".join(out)}</div>'


def _spark(cls: str) -> str:
    # a four-point star drawn with clip-path (page HTML may not keep <svg>)
    return f'<span class="star {cls}" aria-hidden="true"></span>'


def hero_html(compact: bool = False) -> str:
    spark = T.get("spark", T["roles"]["method"])
    title_c, muted, primary = T["title"], T["muted"], T["primary"]
    pink = T["roles"]["contribution"]
    fonts = ('@import url("https://fonts.googleapis.com/css2?family=Syne:wght@700;800'
             '&family=Space+Grotesk:wght@500;700&display=swap");')
    if compact:
        return f"""<style>{fonts}
.sl-mini {{ display:flex; align-items:center; gap:12px; padding:2px 0 14px; }}
.star {{ display:inline-block; background:{spark};
  clip-path:polygon(50% 0%, 60% 40%, 100% 50%, 60% 60%, 50% 100%, 40% 60%, 0% 50%, 40% 40%); }}
.sl-mini .star {{ width:22px; height:22px; }}
.sl-mini b {{ font-family:"Syne",sans-serif; font-weight:800; font-size:26px; letter-spacing:.02em; color:{title_c}; }}
.sl-mini span {{ color:{muted}; font-size:14px; }}
</style><div class="sl-mini">{_spark("s")}<b>SCHOLARLENS</b><span>Search, map and chat with research papers.</span></div>"""

    clip = f"circle({_R}px at {_CX}px {_CY}px)"
    rim = _R + 12
    hx = _CX + rim * 0.7071 - 17
    hy = _CY + rim * 0.7071 - 6
    stream = _stream()
    return f"""<style>{fonts}
.sl-hero {{ position:relative; display:grid; grid-template-columns:minmax(0,1.05fr) minmax(0,.95fr);
  align-items:center; gap:12px; padding:30px 36px 26px; margin:34px 0 18px; border-radius:28px; overflow:hidden;
  background: linear-gradient(135deg, rgba(255,255,255,.06), rgba(255,255,255,.015));
  border:1px solid rgba(255,255,255,.09); box-shadow: 0 30px 80px rgba(10,5,30,.45); }}
.sl-brand {{ grid-column:1 / -1; display:flex; align-items:center; gap:10px; }}
.star {{ display:inline-block; background:{spark};
  clip-path:polygon(50% 0%, 60% 40%, 100% 50%, 60% 60%, 50% 100%, 40% 60%, 0% 50%, 40% 40%); }}
.sl-brand .star {{ width:20px; height:20px; }}
.sl-brand b {{ font-family:"Syne",sans-serif; font-weight:800; font-size:22px; letter-spacing:.04em; color:{title_c}; }}
.sl-brand i {{ flex:1; height:1px; background:linear-gradient(90deg, rgba(255,255,255,.18), transparent); }}
.sl-copy {{ position:relative; z-index:2; padding:6px 0 10px; container-type:inline-size; }}
.sl-h {{ font-family:"Syne","Space Grotesk",sans-serif; font-weight:800; text-transform:uppercase; color:{title_c};
  font-size:clamp(28px, 8cqi, 52px); line-height:.95; letter-spacing:-.01em; margin:0 0 18px; }}
.sl-h .focus {{ background:linear-gradient(90deg, {primary}, {pink}); -webkit-background-clip:text;
  background-clip:text; color:transparent; }}
.sl-h .star {{ width:.62em; height:.62em; vertical-align:.04em; margin-right:.14em;
  animation: sl-twinkle 3.2s ease-in-out infinite; transform-origin:center; }}
.sl-sub {{ color:{muted}; font-size:17px; line-height:1.55; max-width:30em; margin:0 0 28px; }}
.sl-eyebrow {{ display:flex; align-items:center; gap:8px; margin:0 0 12px; color:{spark};
  font:700 12px "Space Grotesk",sans-serif; letter-spacing:.18em; }}
.sl-eyebrow .star {{ width:14px; height:14px; }}
.sl-steps {{ list-style:none; margin:0 0 26px !important; padding:0 !important; display:grid; gap:12px; counter-reset:step; }}
.sl-steps li {{ margin:0 !important; padding:0 !important; }}
.sl-steps li {{ counter-increment:step; display:grid; grid-template-columns:32px minmax(0,1fr); gap:12px;
  align-items:start; color:{muted}; font-size:15px; line-height:1.5; }}
.sl-steps li::before {{ content:counter(step); display:grid; place-items:center; width:30px; height:30px;
  border-radius:50%; border:1px solid rgba(255,255,255,.25); background:rgba(255,255,255,.06);
  color:#ffffff; font:700 13px "Space Grotesk",sans-serif; }}
.sl-steps b {{ color:{title_c}; font-weight:600; }}
.sl-cta {{ display:inline-flex; align-items:stretch; text-decoration:none !important;
  border:1.5px solid rgba(255,255,255,.75); transition: transform .2s ease, box-shadow .2s ease; }}
.sl-cta span {{ padding:16px 26px; font-family:"Space Grotesk",sans-serif; font-weight:700; font-size:14px;
  letter-spacing:.08em; color:{pink}; }}
.sl-cta b {{ display:grid; place-items:center; width:62px; background:#ece8f7; color:#130b26; font-size:22px; }}
.sl-cta:hover {{ transform:translateY(-2px); box-shadow:0 12px 30px rgba(244,114,182,.25); }}

.sl-scene {{ position:relative; width:460px; height:460px; justify-self:center; }}
.blob {{ position:absolute; border-radius:50%; filter:blur(38px); opacity:.75; }}
.blob.b1 {{ width:280px; height:280px; left:150px; top:40px;
  background:radial-gradient(circle, {primary} 0%, rgba(124,58,237,.6) 45%, transparent 70%);
  animation: sl-float 11s ease-in-out infinite alternate; }}
.blob.b2 {{ width:220px; height:220px; left:40px; top:230px;
  background:radial-gradient(circle, {pink} 0%, rgba(236,72,153,.45) 50%, transparent 72%);
  animation: sl-float 14s ease-in-out infinite alternate-reverse; }}
.stream {{ position:absolute; inset:0; overflow:hidden; display:flex; gap:12px; padding:0 14px;
  -webkit-mask-image:linear-gradient(180deg, transparent 0%, #000 14%, #000 86%, transparent 100%);
          mask-image:linear-gradient(180deg, transparent 0%, #000 14%, #000 86%, transparent 100%); }}
.col {{ flex:1; min-width:0; }}
.track {{ display:flex; flex-direction:column; gap:12px; animation: sl-up 30s linear infinite; }}
.c1 .track {{ animation: sl-down 36s linear infinite; }}
.c2 .track {{ animation-duration: 25s; }}
.pc {{ display:flex; flex-direction:column; gap:5px; padding:11px 12px 12px; border-radius:12px;
  background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12); color:#efeafd; }}
.pc-id {{ font:600 8.5px "Space Grotesk",sans-serif; letter-spacing:.06em; color:{muted}; text-transform:uppercase; }}
.pc b {{ font:700 11.5px/1.25 "Space Grotesk",sans-serif; color:#ffffff; }}
.pc i {{ display:block; height:5px; border-radius:3px; background:rgba(255,255,255,.16); }}
.pc i.s {{ width:64%; }}
.pc em {{ align-self:flex-start; font:600 9px "Space Grotesk",sans-serif; font-style:normal; padding:2px 8px;
  border-radius:999px; background:rgba(192,132,252,.22); color:#f3e8ff; }}
.dim {{ position:absolute; inset:0; filter:blur(1.8px) saturate(.85); opacity:.55; }}
.lens-view {{ position:absolute; inset:0; clip-path:{clip}; -webkit-clip-path:{clip}; }}
.lens-view .stream {{ transform:scale(1.42); transform-origin:{_CX}px {_CY}px; }}
.lens-view .pc {{ background:rgba(255,255,255,.12); border-color:rgba(255,255,255,.28); }}
.lens-view .pc em {{ background:{pink}; color:#2a0a1f; }}
.lens-tint {{ position:absolute; inset:0; clip-path:{clip}; -webkit-clip-path:{clip};
  background:radial-gradient(circle at {_CX - 40}px {_CY - 50}px, rgba(255,255,255,.18), transparent 45%),
             radial-gradient(circle at {_CX}px {_CY}px, transparent 60%, rgba(192,132,252,.22) 100%); }}
.rim {{ position:absolute; left:{_CX - rim}px; top:{_CY - rim}px; width:{2 * rim}px; height:{2 * rim}px;
  border-radius:50%; border:12px solid rgba(255,255,255,.10);
  box-shadow: inset 0 0 0 2px rgba(255,255,255,.6), 0 0 0 1px rgba(255,255,255,.28),
              0 28px 70px rgba(124,58,237,.5), inset 0 0 46px rgba(192,132,252,.3); }}
.rim::after {{ content:""; position:absolute; inset:-12px; border-radius:50%;
  background:linear-gradient(115deg, transparent 38%, rgba(255,255,255,.28) 49%, transparent 60%);
  background-size:260% 260%; animation: sl-sweep 6.5s ease-in-out infinite; }}
.handle {{ position:absolute; left:{hx:.0f}px; top:{hy:.0f}px; width:34px; height:150px; border-radius:18px;
  transform:rotate(-45deg); transform-origin:top center;
  background:linear-gradient(90deg, rgba(255,255,255,.42), rgba(192,132,252,.38) 45%, rgba(109,40,217,.65));
  border:1px solid rgba(255,255,255,.4); box-shadow:0 18px 40px rgba(10,5,30,.45); }}
.sp1 {{ position:absolute; left:34px; top:40px; width:46px; height:46px;
  animation: sl-twinkle 2.8s ease-in-out infinite; }}
.sp2 {{ position:absolute; right:30px; top:54px; width:22px; height:22px; background:#ffffff;
  animation: sl-twinkle 3.6s ease-in-out .8s infinite; }}
.match {{ position:absolute; left:6px; bottom:44px; display:flex; align-items:center; gap:8px; padding:9px 13px;
  border-radius:14px; background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.2);
  font:600 12px "Space Grotesk",sans-serif; color:#ffffff; box-shadow:0 14px 34px rgba(10,5,30,.4);
  animation: sl-pop 7s ease-in-out infinite; }}
.match i {{ width:8px; height:8px; border-radius:50%; background:{spark}; box-shadow:0 0 10px {spark}; }}

@keyframes sl-up {{ from {{ transform:translateY(0) }} to {{ transform:translateY(-50%) }} }}
@keyframes sl-down {{ from {{ transform:translateY(-50%) }} to {{ transform:translateY(0) }} }}
@keyframes sl-float {{ from {{ transform:translate(0,0) scale(1) }} to {{ transform:translate(-26px,22px) scale(1.12) }} }}
@keyframes sl-sweep {{ 0%, 100% {{ background-position:100% 100% }} 50% {{ background-position:0% 0% }} }}
@keyframes sl-twinkle {{ 0%, 100% {{ transform:scale(1) rotate(0); opacity:1 }} 50% {{ transform:scale(.75) rotate(18deg); opacity:.75 }} }}
@keyframes sl-pop {{ 0%, 12% {{ opacity:0; transform:translateY(10px) }} 22%, 78% {{ opacity:1; transform:translateY(0) }}
  90%, 100% {{ opacity:0; transform:translateY(-6px) }} }}

@media (max-width: 1100px) {{ .sl-scene {{ transform:scale(.82); margin:-40px 0; }} }}
@media (max-width: 900px) {{ .sl-hero {{ grid-template-columns:1fr; padding:24px 20px; }}
  .sl-scene {{ transform:scale(.7); margin:-70px auto; }} }}
@media (prefers-reduced-motion: reduce) {{ .sl-hero *, .sl-hero *::after {{ animation:none !important; }} }}
</style>
<section class="sl-hero">
  <div class="sl-copy">
    <p class="sl-eyebrow">{_spark("e")}HOW TO USE</p>
    <h2 class="sl-h">Research in sharp <span class="focus">focus</span></h2>
    <ol class="sl-steps">
      <li><span><b>Search.</b> Type a topic in plain words. Add “without …” to leave something out.</span></li>
      <li><span><b>Chat.</b> Open any paper and ask questions. Every answer shows its sources.</span></li>
      <li><span><b>Compare.</b> Pick up to six papers and ask how they differ.</span></li>
      <li><span><b>Research Map.</b> See a paper's key topics, or a timeline of papers and citations.</span></li>
      <li><span><b>Read Next.</b> Mark papers as read to get suggestions on what to read next.</span></li>
    </ol>
    <a class="sl-cta" href="#search-start"><span>START SEARCHING</span><b>&uarr;</b></a>
  </div>
  <div class="sl-scene" aria-hidden="true">
    <div class="blob b1"></div><div class="blob b2"></div>
    <div class="dim">{stream}</div>
    <div class="lens-view">{stream}</div>
    <div class="lens-tint"></div>
    <div class="rim"></div>
    <div class="handle"></div>
    {_spark("sp1")}{_spark("sp2")}
    <div class="match"><i></i>In focus: dense retrieval</div>
  </div>
</section>"""
