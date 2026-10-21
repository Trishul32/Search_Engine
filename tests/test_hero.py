"""Landing hero: renders without SVG (Streamlit strips it) and links to search."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))


def test_hero_full_and_compact():
    from hero import hero_html

    full = hero_html()
    assert "<svg" not in full                       # page HTML drops svg; stars are CSS
    assert 'href="#search-start"' in full and "START SEARCHING" in full
    assert full.count('class="stream"') == 2        # soft layer + magnified lens layer
    assert "prefers-reduced-motion" in full
    assert "HOW TO USE" in full and full.count("<li>") == 5
    compact = hero_html(compact=True)
    assert "SCHOLARLENS" in compact and "sl-scene" not in compact


def test_theme_switch_writes_streamlit_config(tmp_path, monkeypatch):
    import theme

    cfg = theme.config_toml("paper")
    assert 'base = "light"' in cfg and theme.THEMES["paper"]["primary"] in cfg
    assert set(theme.THEMES) >= {"paper", "graphite", "lagoon", "aurora", "glass"}
