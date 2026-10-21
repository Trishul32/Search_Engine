"""Colour themes for ScholarLens: every colour the app uses lives here.

Switch theme (then restart Streamlit):

    python frontend/theme.py paper        # or: graphite, lagoon, aurora

That writes .streamlit/config.toml (Streamlit's own widgets) and saves the
choice in .streamlit/scholarlens-theme, which the custom styles and the
graphs read. SCHOLARLENS_THEME=<name> overrides the saved choice.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHOICE_FILE = ROOT / ".streamlit" / "scholarlens-theme"
DEFAULT = "aurora"

THEMES: dict[str, dict] = {
    # clean light theme with one blue accent
    "paper": {
        "base": "light", "bg": "#f6f7f9", "sidebar": "#eef1f5", "surface": "#ffffff",
        "surface2": "#eef1f5", "border": "#dde2ea", "text": "#1b2230", "title": "#111827",
        "muted": "#5d6879", "primary": "#2453c7", "on_primary": "#ffffff",
        "accent_text": "#1e429f", "link": "#2453c7", "danger": "#b91c1c",
        "roles": {"background": "#0e7490", "concept": "#2453c7", "method": "#15803d",
                  "contribution": "#be185d", "evaluation": "#b45309"},
        "on_role": "#ffffff",
        "timeline": ["#93a8e6", "#2453c7", "#be185d"],      # older, middle, newer
    },
    # calm neutral dark with one blue accent
    "graphite": {
        "base": "dark", "bg": "#14161a", "sidebar": "#181b20", "surface": "#1c1f25",
        "surface2": "#23272e", "border": "#2e333c", "text": "#e7e9ed", "title": "#f4f5f7",
        "muted": "#9aa1ac", "primary": "#3d8bfd", "on_primary": "#ffffff",
        "accent_text": "#8fbaff", "link": "#8fbaff", "danger": "#fca5a5",
        "roles": {"background": "#2dd4bf", "concept": "#3d8bfd", "method": "#84cc16",
                  "contribution": "#f472b6", "evaluation": "#f5b83d"},
        "on_role": "#101215",
        "timeline": ["#5b7db8", "#3d8bfd", "#f472b6"],
    },
    # deep teal with warm highlights
    "lagoon": {
        "base": "dark", "bg": "#0a1517", "sidebar": "#0d1b1d", "surface": "#102123",
        "surface2": "#16292b", "border": "#213a3c", "text": "#e4efed", "title": "#f0fbf9",
        "muted": "#8aa5a2", "primary": "#14b8a6", "on_primary": "#062a26",
        "accent_text": "#5eead4", "link": "#5eead4", "danger": "#fda4af",
        "roles": {"background": "#38bdf8", "concept": "#2dd4bf", "method": "#a3e635",
                  "contribution": "#fb7185", "evaluation": "#fbbf24"},
        "on_role": "#06201d",
        "timeline": ["#38bdf8", "#14b8a6", "#fb7185"],
    },
    # deep indigo with frosted-glass cards, violet / pink glows, lime sparkle
    "glass": {
        "base": "dark", "bg": "#0d0a1f", "sidebar": "#110d27", "surface": "#171230",
        "surface2": "#1d1738", "border": "#2f2752", "text": "#efeafd", "title": "#ffffff",
        "muted": "#a79fc4", "primary": "#c084fc", "on_primary": "#1a0b2e",
        "accent_text": "#f0abfc", "link": "#e9d5ff", "danger": "#fda4af",
        "roles": {"background": "#67e8f9", "concept": "#c084fc", "method": "#a3e635",
                  "contribution": "#f472b6", "evaluation": "#fbbf24"},
        "on_role": "#140a26",
        "timeline": ["#818cf8", "#c084fc", "#f472b6"],
        "effects": "glass",
        "spark": "#e4f75b",
    },
    # navy with violet, cyan and pink (the original look)
    "aurora": {
        "base": "dark", "bg": "#0b1020", "sidebar": "#0f1529", "surface": "#131a30",
        "surface2": "#182039", "border": "#26304d", "text": "#e6e9f5", "title": "#f0edff",
        "muted": "#8f9bbd", "primary": "#8b5cf6", "on_primary": "#ffffff",
        "accent_text": "#c4b5fd", "link": "#a78bfa", "danger": "#fca5a5",
        "roles": {"background": "#22d3ee", "concept": "#8b5cf6", "method": "#22c55e",
                  "contribution": "#f472b6", "evaluation": "#f59e0b"},
        "on_role": "#0b1020",
        "timeline": ["#6366f1", "#8b5cf6", "#ec4899"],
    },
}


def active_name() -> str:
    name = os.environ.get("SCHOLARLENS_THEME", "").strip().lower()
    if not name and CHOICE_FILE.exists():
        name = CHOICE_FILE.read_text(encoding="utf-8").strip().lower()
    return name if name in THEMES else DEFAULT


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def mix(hex_a: str, hex_b: str, t: float) -> str:
    """Blend two colours: t=0 gives a, t=1 gives b."""
    a = [int(hex_a.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(hex_b.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


T = THEMES[active_name()]


def chip_colors(color: str) -> tuple[str, str, str]:
    """(background, border, text) for a soft tinted chip in this theme."""
    text = mix(color, "#000000", 0.25) if T["base"] == "light" else mix(color, "#ffffff", 0.55)
    return rgba(color, 0.12), rgba(color, 0.35), text


def config_toml(name: str) -> str:
    t = THEMES[name]
    return f'''# Generated by frontend/theme.py ({name}). Run `python frontend/theme.py <name>` to change.
[theme]
base = "{t["base"]}"
primaryColor = "{t["primary"]}"
backgroundColor = "{t["bg"]}"
secondaryBackgroundColor = "{t["surface2"]}"
textColor = "{t["text"]}"
linkColor = "{t["link"]}"
borderColor = "{t["border"]}"
baseRadius = "0.7rem"
buttonRadius = "full"
font = "Inter:https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
headingFont = "Space Grotesk:https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap"
codeFont = "JetBrains Mono:https://fonts.googleapis.com/css2?family=JetBrains+Mono&display=swap"
showWidgetBorder = true
chartCategoricalColors = {list(t["roles"].values())!r}

[theme.sidebar]
backgroundColor = "{t["sidebar"]}"
secondaryBackgroundColor = "{t["surface2"]}"

[server]
headless = true

[browser]
gatherUsageStats = false
'''.replace("'", '"')


if __name__ == "__main__":
    choice = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    if choice not in THEMES:
        print(f"usage: python frontend/theme.py [{' | '.join(THEMES)}]")
        sys.exit(1)
    (ROOT / ".streamlit").mkdir(exist_ok=True)
    (ROOT / ".streamlit" / "config.toml").write_text(config_toml(choice), encoding="utf-8")
    CHOICE_FILE.write_text(choice, encoding="utf-8")
    print(f"Theme set to '{choice}'. Restart Streamlit to see it.")
