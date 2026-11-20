"""iOS-style look: SF system font, grouped-background grays, rounded cards, segmented tabs, pills."""

import html

LLM_COLOR = "#007AFF"   # iOS blue
LORA_COLOR = LLM_COLOR
QLORA_COLOR = "#AF52DE"  # iOS purple
XGB_COLOR = "#FF9500"   # iOS orange
GREEN, ORANGE, RED, GRAY = "#34C759", "#FF9500", "#FF3B30", "#8E8E93"

CSS = """
<style>
html, body, [class*="css"], .stApp, button, input, textarea, select {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Inter",
               "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}
.stApp { background: #F2F2F7; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: rgba(242,242,247,0.7); backdrop-filter: saturate(180%) blur(20px); }
section[data-testid="stSidebar"] > div { background: rgba(255,255,255,0.88); backdrop-filter: blur(20px); }
section[data-testid="stSidebar"] { border-right: 1px solid #E5E5EA; }

h1, h2, h3 { letter-spacing: -0.02em; color: #1C1C1E; }
.large-title { font-size: 34px; font-weight: 700; letter-spacing: -0.03em; margin: 0; color: #1C1C1E; }
.subtitle { color: #8E8E93; font-size: 15px; margin: 2px 0 18px 0; }
.section { font-size: 13px; font-weight: 600; color: #8E8E93; text-transform: uppercase;
           letter-spacing: .04em; margin: 18px 4px 8px; }

.card { background: #FFFFFF; border-radius: 18px; padding: 16px 18px; margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 6px 20px rgba(0,0,0,0.04); }
.card h4 { margin: 0 0 6px 0; font-size: 17px; font-weight: 600; }
.muted { color: #8E8E93; font-size: 13px; }
.kpi-label { color: #8E8E93; font-size: 13px; font-weight: 500; }
.kpi-value { font-size: 28px; font-weight: 700; letter-spacing: -0.02em; color: #1C1C1E; line-height: 1.15; }
.kpi-sub { font-size: 12px; color: #8E8E93; }

.pill { display: inline-block; border-radius: 999px; padding: 3px 11px; font-weight: 600; font-size: 13px; margin: 2px 4px 2px 0; }
.p-allow { background: #E7F8EC; color: #1F9D48; }
.p-review { background: #FFF3E0; color: #C46A00; }
.p-block { background: #FFE9E8; color: #D70015; }
.p-gray { background: #EFEFF4; color: #3A3A3C; }
.p-blue { background: #E6F0FF; color: #0062CC; }
.verdict { border-radius: 18px; padding: 18px 22px; margin-bottom: 12px; color: #fff; }
.verdict h2 { color: #fff; margin: 0; font-size: 30px; }
.v-go { background: #34C759; } .v-nogo { background: #FF3B30; } .v-wait { background: #8E8E93; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }

.bar { background: #E5E5EA; border-radius: 999px; height: 10px; overflow: hidden; margin: 6px 0 4px; }
.bar > div { height: 10px; border-radius: 999px; }

.stTabs [data-baseweb="tab-list"] { background: #E3E3E8; border-radius: 11px; padding: 3px; gap: 2px; width: fit-content; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 6px 18px; height: auto; background: transparent; }
.stTabs [data-baseweb="tab"] p { font-size: 14px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #FFFFFF; box-shadow: 0 1px 3px rgba(0,0,0,0.14); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 12px; border: none; background: #007AFF; color: #fff; font-weight: 600; padding: 0.5rem 1.1rem; }
.stButton > button:hover, .stFormSubmitButton > button:hover { background: #0062CC; color: #fff; }
.stButton > button:focus { box-shadow: 0 0 0 3px rgba(0,122,255,.25); color: #fff; }
div[data-baseweb="input"], div[data-baseweb="select"] > div, .stTextArea textarea { border-radius: 10px !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] { background: #fff; border-radius: 14px; }
[data-testid="stExpander"] { background: #fff; border-radius: 14px; border: none; }
</style>
"""

DECISION_CLASS = {"ALLOW": "p-allow", "REVIEW": "p-review", "BLOCK": "p-block"}
DECISION_COLOR = {"ALLOW": GREEN, "REVIEW": ORANGE, "BLOCK": RED}


def esc(x) -> str:
    return html.escape(str(x))


def pill(text, cls="p-gray") -> str:
    return f'<span class="pill {cls}">{esc(text)}</span>'


def decision_pill(d: str) -> str:
    return pill(d, DECISION_CLASS.get(d, "p-gray"))


def dot(color: str) -> str:
    return f'<span class="dot" style="background:{color}"></span>'


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="card"><div class="kpi-label">{esc(label)}</div>'
            f'<div class="kpi-value">{esc(value)}</div><div class="kpi-sub">{esc(sub)}</div></div>')


def prob_bar(p: float, color: str) -> str:
    w = max(0.0, min(1.0, float(p))) * 100
    return f'<div class="bar"><div style="width:{w:.1f}%;background:{color}"></div></div>'


def plotly_layout(fig, height=340, **kw):
    fig.update_layout(template="simple_white", height=height, margin=dict(l=10, r=10, t=36, b=10),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=13),
                      legend=dict(orientation="h", y=1.12, x=0), **kw)
    return fig
