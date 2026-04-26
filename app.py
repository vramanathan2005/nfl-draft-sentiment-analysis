"""
2026 NFL Draft Intelligence — Streamlit app
"""


from pydoc import text
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import html
import re
import requests
from pathlib import Path
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from urllib.parse import quote

st.set_page_config(
    page_title="2026 NFL Draft Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Colors ─────────────────────────────────────────────────────────────────────

NFL_RED = "#D50A0A"
BG      = "#080d18"
CARD    = "#0e1520"
BORDER  = "#1c2840"

DRAFT_COLORS = {
    "riser":     "#22c55e",
    "consensus": "#3b82f6",
    "slide":     "#D50A0A",
}
DRAFT_LABELS = {
    "riser":     "Riser",
    "consensus": "Consensus",
    "slide":     "Slide",
}

# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 0 48px !important;
    max-width: 1480px !important;
    margin: 0 auto !important;
}
.stApp { background-color: #080d18; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a1020;
    border-bottom: 1px solid #1c2840;
    padding: 0 28px 0 270px;
    gap: 0;
    display: flex;
    align-items: center;
}
.stTabs [data-baseweb="tab"] {
    color: #4a6179;
    font-weight: 600;
    font-size: 13px;
    padding: 16px 20px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    background: transparent;
}
.stTabs [data-baseweb="tab"]:hover { color: #94a3b8; }
.stTabs [aria-selected="true"]     { color: #ffffff; border-bottom: 2px solid #D50A0A; }
.stTabs [data-baseweb="tab-panel"] { padding: 28px 28px 40px; }
.brand-row {
    display: flex;
    align-items: flex-end;
    gap: 22px;
    padding: 0 28px;
    margin: 0 0 -56px;
    position: relative;
    z-index: 5;
    width: 0;
}
.brand-link {
    display: inline-block;
    text-decoration: none;
    font-size: 42px;
    line-height: 1;
    font-weight: 900;
    letter-spacing: -1.6px;
    white-space: nowrap;
    color: #f8fafc;
}
.brand-link:link,
.brand-link:visited,
.brand-link:hover,
.brand-link:active {
    color: #f8fafc;
    text-decoration: none;
}
.brand-link .draft {
    color: #f8fafc;
}
.brand-link .intel {
    color: #D50A0A;
}

/* ── Hero metrics ── */
.hero-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}
.metric-card {
    background: #0e1520;
    border: 1px solid #1c2840;
    border-radius: 8px;
    padding: 24px;
}
.m-val { font-size: 36px; font-weight: 800; color: #f1f5f9; line-height: 1; }
.m-val.accent { color: #D50A0A; }
.m-val.riser { color: #22c55e; }
.m-lbl { font-size: 11px; font-weight: 600; color: #4a6179; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 10px; }
.m-sub { font-size: 12px; color: #334a60; margin-top: 4px; }

/* ── Stat cards ── */
.stat-row { display: flex; gap: 12px; margin-bottom: 24px; }
.stat-card {
    flex: 1;
    background: #0e1520;
    border: 1px solid #1c2840;
    border-radius: 8px;
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.s-val { font-size: 28px; font-weight: 800; color: #f1f5f9; line-height: 1; }
.s-lbl { font-size: 10px; font-weight: 600; color: #4a6179; text-transform: uppercase; letter-spacing: 1.5px; }
.s-sub { font-size: 11px; color: #334a60; }

/* ── Section labels ── */
.sec-lbl {
    font-size: 11px;
    font-weight: 600;
    color: #4a6179;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 32px 0 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1c2840;
}

/* ── Player header ── */
.p-header {
    background: #0e1520;
    border: 1px solid #1c2840;
    border-left: 3px solid #D50A0A;
    border-radius: 8px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.p-name  { font-size: 28px; font-weight: 800; color: #f1f5f9; margin: 0 0 14px; letter-spacing: -0.3px; }
.badge   { display: inline-block; padding: 5px 14px; border-radius: 6px; font-size: 13px; font-weight: 700; margin-right: 8px; margin-bottom: 6px; letter-spacing: 0.2px; }
.b-pos   { background: #D50A0A; color: #fff; }
.b-src   { background: #1c2840; color: #94a3b8; border: 1px solid #2a3f5a; }
.b-info  { background: #0e1520; color: #64748b; border: 1px solid #1c2840; }

/* ── Prediction pill ── */
.pred-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin-top: 6px;
}

/* ── Chart label ── */
.chart-lbl {
    font-size: 11px;
    font-weight: 600;
    color: #4a6179;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}

/* ── Inline metric cards ── */
.inline-stats { display: flex; gap: 12px; margin: 16px 0; }
.i-card {
    background: #0e1520;
    border: 1px solid #1c2840;
    border-radius: 6px;
    padding: 14px 18px;
    text-align: center;
    min-width: 100px;
}
.i-val { font-size: 22px; font-weight: 800; color: #f1f5f9; }
.i-lbl { font-size: 10px; font-weight: 600; color: #4a6179; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* ── Sentence cards ── */
.sent-wrap { margin-top: 8px; }
.sent-card {
    background: #0e1520;
    border: 1px solid #1c2840;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 8px;
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.7;
}
.sent-pos   { border-left: 3px solid #3b82f6; }
.sent-neg   { border-left: 3px solid #D50A0A; }
.sent-empty { color: #334a60; font-style: italic; font-size: 13px; padding: 8px 0; }
.ph-tag     { display: inline-block; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.2); color: #93c5fd; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; margin: 5px 3px 0 0; }
.ph-neg     { background: rgba(213,10,10,0.08); border-color: rgba(213,10,10,0.2); color: #fca5a5; }

/* ── Ngram table ── */
.ng-table { width: 100%; border-collapse: collapse; }
.ng-table th { font-size: 10px; font-weight: 600; color: #4a6179; text-transform: uppercase; letter-spacing: 1.5px; padding: 8px 12px; border-bottom: 1px solid #1c2840; text-align: left; }
.ng-table td { padding: 9px 12px; border-bottom: 1px solid rgba(28,40,64,0.6); font-size: 13px; color: #94a3b8; }
.ng-table tr:hover td { background: rgba(255,255,255,0.015); color: #e2e8f0; }
.src-chip  { padding: 2px 7px; border-radius: 3px; font-size: 10px; font-weight: 700; letter-spacing: 0.3px; margin-right: 4px; }
.s-beast   { background: #1e3a5f; color: #93c5fd; }
.s-pff     { background: #1e293b; color: #94a3b8; border: 1px solid #2d3f57; }
.s-br      { background: #2d1515; color: #fca5a5; border: 1px solid #4a2020; }
.table-wrap { background:#0e1520; border:1px solid #1c2840; border-radius:10px; padding:8px 0; }

/* ── Similarity cards ── */
.sim-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 8px; }
.sim-card {
    background: linear-gradient(180deg, rgba(18,27,42,1), rgba(12,18,30,1));
    border: 1px solid #1c2840;
    border-radius: 10px;
    padding: 18px 18px 16px;
}
.sim-name { font-size: 15px; font-weight: 800; color: #f1f5f9; margin-bottom: 6px; line-height: 1.25; }
.sim-meta { font-size: 12px; color: #7b91a8; margin-bottom: 4px; }
.sim-college { font-size: 11px; color: #4a6179; margin-bottom: 6px; font-weight: 500; }
.sim-score { font-size: 22px; color: #f1f5f9; font-weight: 800; letter-spacing: 0.2px; margin-top: 8px; }
.sim-kicker { font-size: 10px; color: #4a6179; font-weight: 700; letter-spacing: 1.4px; text-transform: uppercase; }

/* ── Section card ── */
.section-card {
    background: #0e1520;
    border: 1px solid #1c2840;
    border-radius: 10px;
    padding: 20px 22px;
    margin-bottom: 8px;
}

/* ── Search input font ── */
div[data-testid="stTextInput"] input {
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
div[data-testid="stTextInput"] label p {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* ── Measurables grid ── */
.meas-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 8px; }
.meas-card { background: #0e1520; border: 1px solid #1c2840; border-radius: 6px; padding: 14px 16px; }
.meas-val  { font-size: 20px; font-weight: 700; color: #f1f5f9; }
.meas-lbl  { font-size: 10px; font-weight: 600; color: #4a6179; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

.live-board-card {
    background: linear-gradient(180deg, rgba(14,21,32,0.98), rgba(10,16,26,0.98));
    border: 1px solid #1c2840;
    border-radius: 10px;
    padding: 8px 0;
}
.live-board-head, .live-board-row {
    display: grid;
    grid-template-columns: 64px 86px minmax(150px, 0.9fr) minmax(220px, 1.1fr) 70px minmax(110px, 0.8fr);
    gap: 10px;
    align-items: center;
    padding: 10px 16px;
}
.live-board-head {
    border-bottom: 1px solid #1c2840;
}
.live-head-cell,
.live-body-cell {
    width: 100%;
    box-sizing: border-box;
    height: 54px;
    min-height: 54px;
    max-height: 54px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    border: 1px solid rgba(28,40,64,0.92);
    border-radius: 8px;
    padding: 10px 12px;
    overflow: hidden;
}
.live-head-cell {
    color: #4a6179;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    background: rgba(15, 23, 36, 0.75);
}
.live-body-cell {
    background: rgba(11, 18, 29, 0.9);
}
.live-cell-main {
    color: #f1f5f9;
    font-size: 13px;
    font-weight: 600;
    text-align: center;
}
.live-cell-sub {
    color: #94a3b8;
    font-size: 12px;
    text-align: center;
}
.live-player-link {
    width: 100%;
    height: 100%;
    min-height: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #f1f5f9 !important;
    text-decoration: none !important;
    font-size: 13px;
    font-weight: 600;
    padding: 0;
    margin: 0;
    box-sizing: border-box;
}

.live-body-cell p,
.live-head-cell p {
    margin: 0 !important;
}
.live-player-link:hover,
.live-player-link:visited,
.live-player-link:active {
    color: #ffffff !important;
    text-decoration: none !important;
}

/* ── Player summary grid ── */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 0 0 24px;
}
.summary-card {
    background: linear-gradient(180deg, rgba(15,23,36,1), rgba(11,17,28,1));
    border: 1px solid #1c2840;
    border-radius: 8px;
    padding: 18px 20px;
}
.summary-val { font-size: 30px; font-weight: 800; color: #f1f5f9; line-height: 1; }
.summary-val.pred { letter-spacing: 0.5px; }
.summary-lbl { font-size: 10px; font-weight: 600; color: #4a6179; text-transform: uppercase; letter-spacing: 1.4px; margin-top: 10px; }
.summary-sub { font-size: 11px; color: #64748b; margin-top: 4px; }

/* ── Media fallback ── */
.media-fallback {
    width: 120px; height: 120px; border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.18);
    background: linear-gradient(180deg, #132036, #0b1422);
    display: flex; align-items: center; justify-content: center;
    color: #f1f5f9; font-size: 34px; font-weight: 800; letter-spacing: 1px;
}

/* ── Radio (position selector) ── */
div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
div[data-testid="stRadio"] > div {
    flex-wrap: wrap !important;
    gap: 10px 12px !important;
}
div[data-testid="stRadio"] label {
    background: linear-gradient(180deg, rgba(14,21,32,0.96), rgba(8,13,24,0.96));
    border: 1px solid rgba(71, 85, 105, 0.85);
    border-radius: 999px;
    padding: 10px 16px !important;
    min-height: 0 !important;
    display: flex !important;
    align-items: center !important;
    box-shadow:
        inset 0 0 0 1px rgba(255,255,255,0.02),
        0 0 0 1px rgba(34,197,94,0.00),
        0 0 18px rgba(34,197,94,0.00);
    transition: all 0.18s ease;
}
div[data-testid="stRadio"] label:hover {
    border-color: rgba(74, 222, 128, 0.65);
    box-shadow:
        inset 0 0 0 1px rgba(255,255,255,0.03),
        0 0 0 1px rgba(34,197,94,0.18),
        0 0 16px rgba(34,197,94,0.12);
}
div[data-testid="stRadio"] label p {
    color: #dbe7f5 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.7px !important;
    text-transform: uppercase;
    margin: 0 !important;
    line-height: 1 !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] input:checked + div {
    background: transparent !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    border-color: #4ade80 !important;
    background:
        linear-gradient(180deg, rgba(16, 34, 24, 0.98), rgba(8, 18, 13, 0.98));
    box-shadow:
        inset 0 0 0 1px rgba(134,239,172,0.10),
        0 0 0 1px rgba(74,222,128,0.35),
        0 0 20px rgba(34,197,94,0.22);
}
div[data-testid="stRadio"] label:has(input:checked) p {
    color: #dcfce7 !important;
    text-shadow: 0 0 8px rgba(74,222,128,0.18);
}
div[data-testid="stRadio"] label [data-testid="stMarkdownContainer"] {
    margin-top: 0 !important;
}
div[data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}

/* ── Form controls ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label {
    font-size: 11px; font-weight: 600; color: #4a6179;
    text-transform: uppercase; letter-spacing: 1px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1c2840; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────────

BASE     = Path(__file__).parent
MEAS_RAW = ["ht_in","wt_lbs","arm_in","hand_in","wing_in","dash40","vj_in","bj_in","shuttle","cone3","bench"]
MEAS_LBL = ["Height","Weight","Arm","Hand","Wingspan","40-Yard","Vert Jump","Broad Jump","Shuttle","3-Cone","Bench"]
SPEED_MEAS = {"40-Yard", "Shuttle", "3-Cone"}  # lower = better → invert percentile for radar

RADAR_MEAS = ["ht_in","wt_lbs","dash40","vj_in","arm_in","cone3"]
RADAR_LBL  = ["Height","Weight","40-Yard","Vert Jump","Arm","3-Cone"]


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=None)
def load_data():
    exp   = pd.read_csv(BASE / "data/processed/explain_2026.csv")
    ng    = pd.read_csv(BASE / "data/processed/2026_draft_ngrams.csv")
    inf   = pd.read_csv(BASE / "data/processed/inference_2026.csv")
    all_p = pd.read_csv(BASE / "data/processed/all_prospects.csv")

    logos_path    = BASE / "data/processed/college_logos.csv"
    headshots_path = BASE / "data/raw/nfl_draft_2026_headshots.csv"

    # Merge raw measurables + extra fields from inference
    SCOUT_TEXT_COLS = [
        "beast_summary", "beast_strengths", "beast_weaknesses",
        "pff_overview", "pff_pros", "pff_cons", "pff_bottom_line",
        "br_positives", "br_negatives", "br_article_grade",
    ]
    extra = [
        "Player Name", "beast_grade", "br_pro_comparison", "has_br", "has_pff",
        "text_source", "college_logo_url",
    ] + MEAS_RAW + SCOUT_TEXT_COLS
    inf_sub = inf[[c for c in extra if c in inf.columns]].rename(
        columns={"Player Name": "player_name", "text_source": "inf_text_source"})
    df = exp.merge(inf_sub, on="player_name", how="left")

    # Include BOTR players (in inference but not in explain — no scouting text)
    _botr = inf[inf.get("text_source", pd.Series()).eq("none") if "text_source" in inf.columns
                else pd.Series(False, index=inf.index)].copy()
    _botr_new = _botr[~_botr["Player Name"].isin(df["player_name"])].copy()
    if len(_botr_new):
        _botr_rows = {col: np.nan for col in df.columns}
        _botr_rows["player_name"] = _botr_new["Player Name"].values
        _botr_df = pd.DataFrame({k: v if hasattr(v, "__len__") else [v]*len(_botr_new)
                                  for k, v in _botr_rows.items()})
        _botr_df["player_name"]    = _botr_new["Player Name"].values
        _botr_df["position"]       = _botr_new["Position"].values if "Position" in _botr_new else np.nan
        _botr_df["college"]        = _botr_new["College"].values  if "College"  in _botr_new else np.nan
        _botr_df["consensus"]      = _botr_new["consensus"].values if "consensus" in _botr_new else np.nan
        _botr_df["inf_text_source"] = "none"
        _botr_df["has_br"]         = False
        _botr_df["has_pff"]        = False
        for _c in MEAS_RAW + SCOUT_TEXT_COLS + ["beast_grade"]:
            if _c in _botr_new.columns:
                _botr_df[_c] = _botr_new[_c].values
        df = pd.concat([df, _botr_df], ignore_index=True)

    if "p_reach" in df.columns:
        df.rename(columns={"p_reach": "p_riser"}, inplace=True)
    if "draft_prediction" in df.columns:
        df["draft_prediction"] = df["draft_prediction"].replace("reach", "riser")

    # Merge RAS scores for 2026
    ras_path = BASE / "data/raw/RAS Scores.csv"
    if ras_path.exists():
        ras = pd.read_csv(ras_path)
        ras_2026 = ras[ras["Year"] == 2026][["Name", "RAS", "ALLTIME"]].copy()
        ras_2026 = ras_2026.rename(columns={"Name": "player_name", "RAS": "ras", "ALLTIME": "ras_alltime"})
        ras_2026["player_name"] = ras_2026["player_name"].str.strip()
        # Fuzzy-match RAS names to prospect names
        import re as _re_ras
        def _norm_ras(n):
            n = str(n).lower().strip()
            n = _re_ras.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", n)
            return _re_ras.sub(r"[^a-z ]", "", n).strip()

        prospect_names = df["player_name"].dropna().tolist()
        pname_norm = {_norm_ras(n): n for n in prospect_names}
        matched = []
        try:
            from rapidfuzz import process as _rfp_r, fuzz as _rff_r
            _use_fuzzy_ras = True
        except ImportError:
            _use_fuzzy_ras = False
        for _, rr in ras_2026.iterrows():
            key = _norm_ras(rr["player_name"])
            if key in pname_norm:
                matched.append({"player_name": pname_norm[key], "ras": rr["ras"], "ras_alltime": rr["ras_alltime"]})
            elif _use_fuzzy_ras:
                res = _rfp_r.extractOne(key, list(pname_norm.keys()), scorer=_rff_r.token_sort_ratio)
                if res and res[1] >= 85:
                    matched.append({"player_name": pname_norm[res[0]], "ras": rr["ras"], "ras_alltime": rr["ras_alltime"]})
        if matched:
            ras_df = pd.DataFrame(matched).drop_duplicates("player_name")
            df = df.merge(ras_df, on="player_name", how="left")
        if "ras" not in df.columns:
            df["ras"] = None
            df["ras_alltime"] = None
    else:
        df["ras"] = None
        df["ras_alltime"] = None

    # Merge headshots
    if headshots_path.exists():
        hs = pd.read_csv(headshots_path)[["player_name", "headshot_url"]].copy()
        hs["player_name"] = hs["player_name"].str.strip()
        # Alias map: prospect name → headshot CSV name (apostrophes, Jr/II/III, T.J. format, nicknames)
        _hs_aliases = {
            "AJ Haulcy": "A.J. Haulcy",
            "AJ Pena": "A.J. Pena",
            "AMarion McCoy": "A'Marion McCoy",
            "Alzillion Hamilton": "Al'zillion Hamilton",
            "Anthony Hill": "Anthony Hill Jr.",
            "Armaj Reed-Adams": "Ar'maj Reed-Adams",
            "Brian Parker": "Brian Parker II",
            "Bryan McCoy": "Bryan McCoy Jr.",
            "Bryan Thomas": "Bryan Thomas Jr.",
            "Byron Cardwell": "Byron Cardwell Jr.",
            "Chris Brazzell": "Chris Brazzell II",
            "Chris Hilton": "Chris Hilton Jr.",
            "DJ Graham": "DJ Graham II",
            "DaeQuan Wright": "Dae'Quan Wright",
            "Dangelo Ponds": "D'Angelo Ponds",
            "Darrell Jackson": "Darrell Jackson Jr.",
            "David Blay": "David Blay Jr.",
            "DeZhaun Stribling": "De'Zhaun Stribling",
            "Donaven Mcculley": "Donaven McCulley",
            "EJ Williams": "E.J. Williams Jr.",
            "Eddie Kelly": "Eddie Kelly Jr.",
            "Emmanuel Henderson": "Emmanuel Henderson Jr.",
            "Enrique Cruz": "Enrique Cruz Jr.",
            "Eric ONeill": "Eric O'Neill",
            "Faalili Faamoe": "Fa'alili Fa'amoe",
            "Fred Davis": "Fred Davis Ii",
            "Gary Smith": "Gary Smith III",
            "George Gumbs": "George Gumbs Jr.",
            "Harold Perkins": "Harold Perkins Jr.",
            "Harrison Wallace": "Harrison Wallace III",
            "JC Davis": "J.C. Davis",
            "JMari Taylor": "J'Mari Taylor",
            "JMichael Sturdivant": "J. Michael Sturdivant",
            "JQ Hardaway": "Jonquis Hardaway",
            "JaKobi Lane": "Ja'Kobi Lane",
            "JaMori Maclin": "Ja'Mori Maclin",
            "James Neal": "James Neal III",
            "James Thompson": "James Thompson Jr.",
            "Jeffrey Mba": "Jeffrey M'ba",
            "John Bock": "John Bock II",
            "Joseph Manjack": "Joseph Manjack IV",
            "Kaena Decambra": "Ka'ena Decambra",
            "Keith Abney": "Keith Abney II",
            "Kelvin Gilliam": "Kelvin Gilliam Jr.",
            "Kevin Coleman": "Kevin Coleman Jr.",
            "Kevin Concepcion": "KC Concepcion",
            "LJ Johnson": "L.J. Johnson Jr.",
            "Lance St Louis": "Lance St. Louis",
            "Latrell McCutchin": "Latrell McCutchin Sr.",
            "LeVeon Moss": "Le'Veon Moss",
            "Leon Lowery": "Leon Lowery Jr.",
            "Lorenzo Styles": "Lorenzo Styles Jr.",
            "Marcus Burris": "Marcus Burris Jr.",
            "Marvin Jones": "Marvin Jones Jr.",
            "Maurice Westmoreland": "Mo Westmoreland",
            "Max Tomzcak": "Max Tomczak",
            "Mike Washington": "Mike Washington Jr.",
            "Nick Degennaro": "Nick DeGennaro",
            "Nick Singleton": "Nicholas Singleton",
            "OMega Blake": "O'Mega Blake",
            "Omar Cooper": "Omar Cooper Jr.",
            "Raheem Anderson": "Raheem Anderson II",
            "Reggie Grimes": "Reggie Grimes II",
            "Reuben Fatheree": "Reuben Fatheree II",
            "Robert Henry": "Robert Henry Jr.",
            "Rueben Bain": "Rueben Bain Jr.",
            "Samuel MPemba": "Sam M'Pemba",
            "Shadrach Banks": "Shad Banks Jr.",
            "Stephen Dix": "Stephen Dix Jr.",
            "TJ Parker": "T.J. Parker",
            "Tay Yanta": "Tay Yanta Ii",
            "Thomas Castellanos": "Tommy Castellanos",
            "Tim Keenan": "Tim Keenan III",
            "Toriano Pride": "Toriano Pride Jr.",
            "TreVonte Citizen": "Tre'Vonte Citizen",
            "Trevion Cooley": "Trey Cooley",
            "Trey Zuhn": "Trey Zuhn III",
            "Tywone Malone": "Tywone Malone Jr.",
            "Vincent Anthony": "Vincent Anthony Jr.",
            "Vinny Anthony": "Vinny Anthony II",
            "Wendell Moe": "Wendell Moe Jr.",
            "Will Lee": "Will Lee III",
            "Wydett Williams": "Wydett Williams Jr.",
            "Xavian Sorey": "Xavian Sorey Jr.",
        }
        # Add alias rows so both names resolve to the same headshot
        alias_rows = []
        hs_lookup = hs.set_index("player_name")["headshot_url"].to_dict()
        for prospect_name, hs_name in _hs_aliases.items():
            if hs_name in hs_lookup and prospect_name not in hs_lookup:
                alias_rows.append({"player_name": prospect_name, "headshot_url": hs_lookup[hs_name]})
        if alias_rows:
            hs = pd.concat([hs, pd.DataFrame(alias_rows)], ignore_index=True)
        df = df.merge(hs, on="player_name", how="left")
    else:
        df["headshot_url"] = None

    if logos_path.exists():
        logos = pd.read_csv(logos_path)
        if {"college", "college_logo_url"}.issubset(logos.columns):
            logos = logos[["college", "college_logo_url"]].drop_duplicates(subset=["college"])
            df = df.merge(logos, on="college", how="left", suffixes=("", "_lookup"))
            if "college_logo_url_lookup" in df.columns:
                df["college_logo_url"] = df["college_logo_url"].fillna(df["college_logo_url_lookup"])
                df = df.drop(columns=["college_logo_url_lookup"])

    # Use inference's text_source (correct) over explain_2026's (all "beast" due to pipeline bug)
    df["text_source"] = df["inf_text_source"].fillna(df["text_source"])

    # Merge beast_rank from all_prospects (2026 only)
    p26 = (all_p[all_p["draft_year"] == 2026][["Player Name","beast_rank"]]
           .rename(columns={"Player Name": "player_name"}))
    df = df.merge(p26, on="player_name", how="left")

    # Rise/Fall: beast_rank is position-specific (#1 CB, #1 RB, etc.)
    # Compare to consensus rank within the same position group (apples to apples)
    df["consensus_pos_rank"] = df.groupby("position")["consensus"].rank(method="min")
    # positive = beast ranks higher within position (potential riser)
    df["rise_fall"] = df["consensus_pos_rank"] - df["beast_rank"]
    df["position_rank"] = df["beast_rank"].where(df["beast_rank"].notna(), df["consensus_pos_rank"])

    # Scout confidence: cross-source phrase count + BR coverage
    ng_counts = (ng.groupby("Player.Name").size().reset_index(name="ngram_count")
                 .rename(columns={"Player.Name": "player_name"}))
    df = df.merge(ng_counts, on="player_name", how="left")
    df["ngram_count"] = df["ngram_count"].fillna(0).astype(int)
    df = df.drop_duplicates(subset=["player_name"]).reset_index(drop=True)
    df["has_br"] = df["has_br"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)
    ng_max = max(df["ngram_count"].max(), 1)
    df["scout_conf"] = (df["ngram_count"] / ng_max * 70 + df["has_br"] * 30).clip(0, 100).round(1)

    # Position-group measurable percentiles (for radar)
    for col, lbl in zip(MEAS_RAW, MEAS_LBL):
        if col in df.columns:
            pct = df.groupby("position")[col].rank(pct=True, na_option="keep") * 100
            if lbl in SPEED_MEAS:
                pct = 100 - pct   # invert: lower time = better = higher radar value
            df[f"pct_{col}"] = pct

    # ── TF-IDF: 2026-only similarity + historical comps ──────────────────────
    BEAST_COLS = ["beast_summary","beast_strengths","beast_weaknesses"]
    PFF_COLS   = ["pff_overview","pff_pros","pff_cons","pff_bottom_line","pff_extra"]
    BR_COLS    = ["br_positives","br_negatives"]

    def make_text(src):
        bt  = src[[c for c in BEAST_COLS if c in src.columns]].fillna("").apply(
                  " ".join, axis=1).str.strip()
        pt  = src[[c for c in PFF_COLS  if c in src.columns]].fillna("").apply(
                  " ".join, axis=1).str.strip()
        brt = src[[c for c in BR_COLS   if c in src.columns]].fillna("").apply(
                  " ".join, axis=1).str.strip()
        base = bt.where(bt.str.len() > 0, pt)
        return (base + " " + brt).str.strip().where(brt.str.len() > 0, base)

    # 2026 texts (from inference)
    texts_26   = make_text(inf)
    names_26   = inf["Player Name"].tolist()

    # Historical texts (2019–2025, exclude 2026)
    hist = all_p[(all_p["draft_year"] < 2026) &
                 (all_p["beast_summary"].notna() | all_p["pff_overview"].notna())].copy()
    texts_hist = make_text(hist)
    names_hist = hist["Player Name"].tolist()

    # Fit TF-IDF on all text together, then split
    all_names  = names_26 + names_hist
    all_texts  = list(texts_26) + list(texts_hist)
    try:
        tfidf_v = TfidfVectorizer(ngram_range=(1, 2), max_features=6000,
                                  stop_words="english", min_df=2)
        X_all    = tfidf_v.fit_transform(all_texts)
        X_26     = X_all[: len(names_26)]
        X_hist   = X_all[len(names_26):]
        # 2026 vs 2026 similarity
        sim_mat  = cos_sim(X_26)
        sim_df   = pd.DataFrame(sim_mat, index=names_26, columns=names_26)
        # 2026 vs historical similarity
        hist_mat = cos_sim(X_26, X_hist)   # shape (378, n_hist)
        hist_sim_df = pd.DataFrame(hist_mat, index=names_26, columns=names_hist)
    except Exception:
        sim_df      = pd.DataFrame()
        hist_sim_df = pd.DataFrame()

    # Historical metadata lookup (name → row)
    hist_cols = ["Position","draft_year","round","pick","sc_contract_tier","sc_contract_type"]
    if "College" in hist.columns:
        hist_cols.append("College")
    hist_cols = [c for c in hist_cols if c in hist.columns]
    hist_meta = hist.set_index("Player Name")[hist_cols].copy()

    # #1 overall pick can't go higher — zero out riser probability and renormalize
    mask = df["consensus"] == 1
    df.loc[mask, "p_riser"] = 0.0
    total = df.loc[mask, ["p_slide","p_consensus"]].sum(axis=1)
    df.loc[mask, "p_slide"]     = df.loc[mask, "p_slide"]     / total * 100
    df.loc[mask, "p_consensus"] = df.loc[mask, "p_consensus"] / total * 100

    df["round_est"] = ((df["consensus"] - 1) // 32 + 1).clip(1, 7)
    return df, ng, sim_df, hist_sim_df, hist_meta

df, ng, sim_df, hist_sim_df, hist_meta = load_data()

# ── Strip name/college tokens from ngrams ──────────────────────────────────
def _build_entity_blocklist(dataframe):
    """Build a set of words that are player name parts or college name parts."""
    import re as _re
    # Common words that happen to appear in names/colleges but are real scouting terms
    _false_positives = {
        "college", "ball", "allen", "hill", "brown", "white", "green", "long",
        "short", "north", "south", "east", "west", "central", "state", "young",
        "cross", "man", "men", "back", "field", "land", "ford", "son", "tion",
        "strong", "late", "early", "run", "pass", "zone", "red",
    }
    tokens = set()
    for name in dataframe["player_name"].dropna():
        for tok in _re.split(r"\s+", name.lower()):
            tok = _re.sub(r"[^a-z]", "", tok)
            if len(tok) >= 4 and tok not in _false_positives:
                tokens.add(tok)
    for col in dataframe["college"].dropna() if "college" in dataframe.columns else []:
        for tok in _re.split(r"\s+", col.lower()):
            tok = _re.sub(r"[^a-z]", "", tok)
            if len(tok) >= 4 and tok not in _false_positives:
                tokens.add(tok)
    return tokens

_entity_tokens = _build_entity_blocklist(df)

def _ngram_has_entity(phrase, blocklist):
    words = set(re.sub(r"[^a-z\s]", "", str(phrase).lower()).split())
    return bool(words & blocklist)

ng = ng[~ng["ngram"].apply(_ngram_has_entity, blocklist=_entity_tokens)].copy()

qp_player = st.query_params.get("player")
if qp_player:
    player_match = df[df["player_name"] == qp_player]
    if len(player_match) > 0:
        st.session_state["pos_pills"] = player_match.iloc[0]["position"]
        st.session_state["player_sel"] = qp_player


# ── Chart helpers ──────────────────────────────────────────────────────────────

_BASE_LAYOUT = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(color="#94a3b8", family="Inter", size=12),
    margin=dict(l=16, r=16, t=44, b=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#64748b", size=11)),
    hoverlabel=dict(bgcolor="#0e1520", bordercolor="#1c2840",
                    font=dict(color="#f1f5f9", size=12)),
    xaxis=dict(gridcolor="#131e2e", linecolor="#1c2840",
               zerolinecolor="#1c2840", tickfont=dict(color="#4a6179")),
    yaxis=dict(gridcolor="#131e2e", linecolor="#1c2840",
               zerolinecolor="#1c2840", tickfont=dict(color="#4a6179")),
)

def styled(fig, height=380, **kw):
    fig.update_layout(height=height, **{**_BASE_LAYOUT, **kw})
    return fig


def parse_sentences(text):
    if pd.isna(text) or not str(text).strip():
        return []
    result = []
    for part in str(text).split(" || "):
        part = part.strip()
        if "[" in part and part.endswith("]"):
            sent    = part[:part.rfind("[")].strip().strip('"')
            phrases = [p.strip() for p in part[part.rfind("[")+1:-1].split(",") if p.strip()]
        else:
            sent, phrases = part.strip().strip('"'), []
        if sent:
            result.append((sent, phrases))
    return result


def render_sentences(sent_list, is_neg=False):
    cls  = "sent-neg" if is_neg else "sent-pos"
    ptag = "ph-neg"   if is_neg else "ph-tag"
    html = '<div class="sent-wrap">'
    for sent, phrases in sent_list:
        tags = "".join(f'<span class="{ptag}">{p}</span>' for p in phrases)
        html += f'<div class="sent-card {cls}">{sent}{"<br>" + tags if tags else ""}</div>'
    html += "</div>"
    return html


def fmt_meas(lbl, val):
    if pd.isna(val):
        return None
    v = float(val)
    lbl_l = lbl.lower().replace("-", "").replace(" ", "")
    if lbl_l == "height":
        ft, inch = int(v // 12), v % 12
        return f"{ft}'{inch:.1f}\""
    if lbl_l == "weight":    return f"{v:.0f} lbs"
    if lbl_l in ("40yard","shuttle","3cone"): return f"{v:.2f}s"
    if lbl_l == "bench":     return f"{v:.0f} reps"
    return f"{v:.1f}\""


def parse_word_str(words_str):
    if pd.isna(words_str) or not str(words_str).strip():
        return []
    result = []
    for item in str(words_str).split("|"):
        if ":" in item:
            word, score = item.rsplit(":", 1)
            try:
                word = word.strip()
                # Drop any word/phrase that contains a player name or college token
                if not _ngram_has_entity(word, _entity_tokens):
                    result.append((word, float(score)))
            except ValueError:
                pass
    return result


def render_word_chips(words_str, positive=True):
    items = parse_word_str(words_str)
    if not items:
        return ""
    if positive:
        bg, border, color = "rgba(41,121,255,0.10)", "rgba(41,121,255,0.22)", "#90caf9"
    else:
        bg, border, color = "rgba(213,10,10,0.08)", "rgba(213,10,10,0.22)", "#ef9a9a"
    chips = ""
    for word, score in items:
        chips += (f'<span style="display:inline-block;background:{bg};border:1px solid {border};'
                  f'color:{color};padding:4px 11px;border-radius:20px;font-size:12px;'
                  f'font-weight:600;margin:3px 3px 3px 0;">{word} ({score:+.4f})</span>')
    return chips


def render_inline_summary(pos_str, neg_str):
    pos = parse_word_str(pos_str)[:5]
    neg = parse_word_str(neg_str)[:3]
    lines = []
    if pos:
        words = " | ".join(f"{w}({s:+.4f})" for w, s in pos)
        lines.append(f'<span style="color:rgba(255,255,255,0.5);font-size:12px;">'
                     f'<b style="color:#90caf9;">Top words for prediction:</b> {words}</span>')
    if neg:
        words = " | ".join(f"{w}({s:+.4f})" for w, s in neg)
        lines.append(f'<span style="color:rgba(255,255,255,0.5);font-size:12px;">'
                     f'<b style="color:#ef9a9a;">Words against:</b> {words}</span>')
    return "<br>".join(lines)


def initials(text, n=2):
    parts = [p for p in str(text).replace(".", " ").split() if p]
    if not parts:
        return "NA"
    return "".join(p[0] for p in parts[:n]).upper()


def render_confidence_terms(ng_row, limit=3):
    if ng_row is None or len(ng_row) == 0:
        return "", "Confidence is lower because there are fewer repeated cross-source terms in the scouting language."

    top_terms = (ng_row.sort_values(["sources_present", "total_count"], ascending=False)
                     .head(limit))
    chips = []
    phrases = []
    for _, term in top_terms.iterrows():
        phrase = html.escape(str(term["ngram"]))
        src_n = int(term["sources_present"])
        phrases.append(f'"{phrase}"')
        chips.append(
            f'<span style="display:inline-block;background:rgba(59,130,246,0.10);'
            f'border:1px solid rgba(59,130,246,0.22);color:#bfdbfe;padding:4px 10px;'
            f'border-radius:999px;font-size:11px;font-weight:700;margin:3px 4px 0 0;">'
            f'{phrase} · {src_n} source{"s" if src_n != 1 else ""}</span>'
        )

    if len(phrases) == 1:
        summary = f'Scout confidence is supported by repeated phrasing like {phrases[0]}.'
    else:
        summary = f'Scout confidence is supported by repeated phrasing like {", ".join(phrases[:-1])}, and {phrases[-1]}.'
    return "".join(chips), summary


def show_player_logo_panel(player_name, college, college_logo_url=None, headshot_url=None):
    college_tag  = initials(college, n=3) if college else "COL"
    has_headshot = pd.notna(headshot_url) and str(headshot_url).strip()

    if has_headshot:
        hs = str(headshot_url).strip()
        st.markdown(
            f'<div style="text-align:center;">'
            f'<img src="{hs}" style="width:100%;max-width:220px;border-radius:12px;'
            f'border:1px solid #1c2840;display:block;margin:0 auto;"/>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f'<div class="media-fallback" style="margin:12px auto 0;">{college_tag}</div>',
                    unsafe_allow_html=True)


def college_logo_block(college_logo_url, college, size=64):
    """Returns an HTML snippet for a college logo in a white rounded container."""
    has_logo = pd.notna(college_logo_url) and str(college_logo_url).strip()
    if has_logo:
        return (f'<div style="background:#fff;border-radius:10px;padding:10px;'
                f'width:{size+20}px;height:{size+20}px;display:flex;align-items:center;'
                f'justify-content:center;flex-shrink:0;">'
                f'<img src="{str(college_logo_url).strip()}" '
                f'style="max-width:{size}px;max-height:{size}px;object-fit:contain;"/>'
                f'</div>')
    tag = initials(college, n=3) if college else "COL"
    return (f'<div style="background:#132036;border-radius:10px;padding:10px;'
            f'width:{size+20}px;height:{size+20}px;display:flex;align-items:center;'
            f'justify-content:center;flex-shrink:0;color:#94a3b8;font-size:13px;font-weight:700;">'
            f'{tag}</div>')


def aggregate_tier_words(tier_df, col, positive=True, n=12):
    """Aggregate word scores across players for a draft tier."""
    word_scores = {}
    for _, row in tier_df.iterrows():
        for word, score in parse_word_str(row.get(col, "")):
            if (score > 0) == positive:
                word_scores.setdefault(word, []).append(abs(score))
    result = [(w, float(np.mean(s)), len(s)) for w, s in word_scores.items() if len(s) >= 2]
    result.sort(key=lambda x: -x[1] * np.log1p(x[2]))
    return result[:n]


def radar_chart(row_data):
    """Build a measurables radar chart for a single player row."""
    labels, values = [], []
    for col, lbl in zip(RADAR_MEAS, RADAR_LBL):
        pct_col = f"pct_{col}"
        v = row_data.get(pct_col)
        if pd.notna(v):
            labels.append(lbl)
            values.append(float(v))
    if len(values) < 3:
        return None
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]
    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(213,10,10,0.12)",
        line=dict(color=NFL_RED, width=2),
        hovertemplate="<b>%{theta}</b><br>%{r:.0f}th %ile vs position<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(color="#4a6179", size=9),
                            gridcolor="#1c2840", linecolor="#1c2840",
                            tickvals=[25, 50, 75, 100],
                            ticktext=["25th", "50th", "75th", "100th"]),
            angularaxis=dict(tickfont=dict(color="#94a3b8", size=11),
                             linecolor="#1c2840", gridcolor="#1c2840"),
            bgcolor=CARD,
        ),
        paper_bgcolor=CARD,
        height=300,
        margin=dict(l=40, r=40, t=24, b=24),
        font=dict(family="Inter", color="#94a3b8"),
        hoverlabel=dict(bgcolor="#0e1520", bordercolor="#1c2840",
                        font=dict(color="#f1f5f9", size=12)),
    )
    return fig


def normalize_player_name(name):
    name = str(name or "").lower()
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", name)
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name


def normalize_team_name(name):
    name = str(name or "").lower().strip()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


@st.cache_data(ttl=86400)
def load_nfl_team_meta():
    urls = [
        "https://raw.githubusercontent.com/nflverse/nflverse-pbp/master/teams_colors_logos.csv",
        "https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/teams_colors_logos.csv",
    ]
    for url in urls:
        try:
            teams = pd.read_csv(url)
            if {"team_name", "team_logo_espn"}.issubset(teams.columns):
                meta = {}
                for _, row in teams.iterrows():
                    team_name = row.get("team_name")
                    if pd.isna(team_name):
                        continue
                    key = normalize_team_name(team_name)
                    meta[key] = {
                        "logo": row.get("team_logo_espn"),
                        "color_1": row.get("team_color"),
                        "color_2": row.get("team_color2"),
                        "color_3": row.get("team_color3"),
                        "color_4": row.get("team_color4"),
                    }
                return meta
        except Exception:
            pass
    return {}


def hex_to_rgba(value, alpha):
    if value is None or pd.isna(value):
        return None
    hex_value = str(value).strip().lstrip("#")
    if len(hex_value) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", hex_value):
        return None
    r = int(hex_value[0:2], 16)
    g = int(hex_value[2:4], 16)
    b = int(hex_value[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _hex_luminance(hex_color):
    hx = str(hex_color or "").strip().lstrip("#")
    if len(hx) != 6:
        return -1
    r, g, b = int(hx[0:2], 16) / 255, int(hx[2:4], 16) / 255, int(hx[4:6], 16) / 255
    def _lin(c): return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def readable_team_color(*hex_colors, fallback="#e2e8f0", threshold=0.08):
    """Try each color in order; return the first one readable on a dark background."""
    for c in hex_colors:
        if c and _hex_luminance(c) >= threshold:
            return c
    return fallback


@st.cache_data(ttl=300)
def load_live_wikipedia_board():
    url = "https://en.wikipedia.org/wiki/2026_NFL_draft"
    headers = {"User-Agent": "Mozilla/5.0 draftIntel/1.0"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    candidate = None
    for tbl in tables:
        cols = [str(c).strip() for c in tbl.columns]
        if {"Rnd.", "Pick"}.issubset(cols) and ({"NFL team", "Team"} & set(cols)):
            candidate = tbl.copy()
            break
    if candidate is None:
        return pd.DataFrame(columns=["Round", "Pick", "NFL Team", "Player", "Pos", "College"])

    rename_map = {
        "Rnd.": "Round",
        "NFL team": "NFL Team",
        "Team": "NFL Team",
        "Pos.": "Pos",
    }
    candidate = candidate.rename(columns=rename_map)
    keep_cols = [c for c in ["Round", "Pick", "NFL Team", "Player", "Pos", "College"] if c in candidate.columns]
    board = candidate[keep_cols].copy()

    for col in ["Round", "Pick", "NFL Team", "Player", "Pos", "College"]:
        if col not in board.columns:
            board[col] = ""

    board["Round"] = board["Round"].astype(str).str.replace(r"[^0-9]", "", regex=True)
    board["Pick"] = board["Pick"].astype(str).str.replace(r"[^0-9]", "", regex=True)
    board["NFL Team"] = board["NFL Team"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()
    for col in ["Player", "Pos", "College"]:
        board[col] = (
            board[col]
            .astype(str)
            .str.replace(r"\[.*?\]", "", regex=True)
            .str.strip()
            .replace({"nan": "", "NaN": "", "None": "", "<NA>": ""})
        )
    board = board[board["Round"].ne("") & board["Pick"].ne("")]
    board["Round"] = board["Round"].astype(int)
    board["Pick"] = board["Pick"].astype(int)
    return board.reset_index(drop=True)


def compute_actual_tier(consensus, actual_pick):
    """Return 'riser'/'slide'/'consensus'/None using the same threshold formula as the pick card."""
    import math as _math
    if not isinstance(consensus, int) or consensus > 257 or actual_pick is None:
        return None
    rise = consensus - actual_pick
    thresh = _math.floor(max(1, 2 * _math.log(consensus) + 0.07 * min(consensus, 200) ** 0.9))
    if rise >= thresh:
        return "riser"
    elif rise <= -thresh:
        return "slide"
    return "consensus"


def render_pick_card(dr_round, dr_pick, dr_team, dr_meta, consensus, predicted_tier):
    """Render the inline drafted card given pick info and model prediction."""
    import math as _math
    dr_raw_color = dr_meta.get("color_1") or dr_meta.get("color_2") or "#3b82f6"
    dr_color     = readable_team_color(dr_meta.get("color_1"), dr_meta.get("color_2"), fallback="#3b82f6")
    dr_logo      = dr_meta.get("team_logo_espn") or dr_meta.get("logo")

    if dr_pick <= 32:   pick_tier = "Round 1"
    elif dr_pick <= 64: pick_tier = "Round 2"
    elif dr_pick <= 105: pick_tier = "Round 3"
    elif dr_pick <= 140: pick_tier = "Round 4"
    elif dr_pick <= 175: pick_tier = "Round 5"
    elif dr_pick <= 215: pick_tier = "Round 6"
    else:               pick_tier = "Round 7 / UDFA"

    rise_fall_picks = consensus - dr_pick if isinstance(consensus, int) else None
    _undrafted_consensus = isinstance(consensus, int) and consensus > 257

    if _undrafted_consensus:
        accuracy_label, accuracy_color, accuracy_icon = "Beat undrafted projection", "#22c55e", "↑"
        rf_label, rf_color = "Projected undrafted", "#94a3b8"
    elif rise_fall_picks is not None:
        _tier_thresh = _math.floor(max(1, 2 * _math.log(consensus) + 0.07 * min(consensus, 200) ** 0.9))
        actual_tier = "riser" if rise_fall_picks >= _tier_thresh else ("slide" if rise_fall_picks <= -_tier_thresh else "consensus")
        if actual_tier == predicted_tier:
            accuracy_label, accuracy_color, accuracy_icon = "Model called it", "#22c55e", "✓"
        elif actual_tier == "riser":
            accuracy_label, accuracy_color, accuracy_icon = "Went earlier than model predicted", "#f59e0b", "↑"
        elif actual_tier == "slide":
            accuracy_label, accuracy_color, accuracy_icon = "Went later than model predicted", "#ef4444", "↓"
        else:
            accuracy_label, accuracy_color, accuracy_icon = "Closer to consensus than predicted", "#94a3b8", "→"
        if rise_fall_picks > 0:
            rf_label, rf_color = f"Rose {rise_fall_picks} spots vs consensus", "#22c55e"
        elif rise_fall_picks < 0:
            rf_label, rf_color = f"Fell {abs(rise_fall_picks)} spots vs consensus", "#ef4444"
        else:
            rf_label, rf_color = "Picked exactly at consensus", "#94a3b8"
    else:
        accuracy_label, accuracy_color, accuracy_icon = "—", "#94a3b8", ""
        rf_label, rf_color = "—", "#94a3b8"

    pred_lbl = DRAFT_LABELS.get(predicted_tier, predicted_tier.title() if predicted_tier else "—")
    cons_lbl = f"#{consensus}" if isinstance(consensus, int) else "—"
    team_logo_html = (
        f'<img src="{html.escape(str(dr_logo))}" style="height:44px;object-fit:contain;display:block;" />'
        if dr_logo else
        f'<div style="font-size:13px;font-weight:700;color:#f1f5f9;">{html.escape(dr_team)}</div>'
    )
    dr_rgba   = hex_to_rgba(dr_raw_color, 0.18) or "rgba(59,130,246,0.18)"
    dr_border = hex_to_rgba(dr_raw_color, 0.5)  or "rgba(59,130,246,0.5)"

    st.markdown(f"""
    <div style="background:{dr_rgba};border:1px solid {dr_border};border-radius:10px;
                padding:18px 20px;margin:8px 0 4px;">
      <div style="font-size:10px;font-weight:700;color:{dr_color};text-transform:uppercase;
                  letter-spacing:1.2px;margin-bottom:12px;">DRAFTED</div>
      <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px;min-width:80px;">
          {team_logo_html}
          <div style="font-size:11px;color:#94a3b8;">{html.escape(dr_team)}</div>
        </div>
        <div style="display:flex;gap:16px;flex-wrap:wrap;flex:1;">
          <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:12px 18px;text-align:center;min-width:90px;">
            <div style="font-size:24px;font-weight:900;color:{dr_color};">R{dr_round}</div>
            <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Round</div>
          </div>
          <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:12px 18px;text-align:center;min-width:90px;">
            <div style="font-size:24px;font-weight:900;color:{dr_color};">#{dr_pick}</div>
            <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Overall</div>
          </div>
          <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:12px 18px;text-align:center;min-width:90px;">
            <div style="font-size:15px;font-weight:700;color:#f1f5f9;">{pick_tier}</div>
            <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Pick Value</div>
          </div>
          <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:12px 18px;text-align:center;min-width:120px;">
            <div style="font-size:15px;font-weight:700;color:{accuracy_color};">{accuracy_icon} {accuracy_label}</div>
            <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">vs Model Prediction ({pred_lbl})</div>
          </div>
          <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:12px 18px;text-align:center;min-width:120px;">
            <div style="font-size:15px;font-weight:700;color:{rf_color};">{rf_label}</div>
            <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">vs Consensus {cons_lbl}</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_live_board(df, live_board, live_error=None):
    st.markdown('<div class="sec-lbl" style="margin-top:0">Live Draft Board</div>', unsafe_allow_html=True)
    if live_error:
        st.markdown(f'<div class="sent-empty">Live board unavailable right now: {html.escape(live_error)}</div>',
                    unsafe_allow_html=True)
        return
    if live_board.empty:
        st.markdown('<div class="sent-empty">No live draft rows available yet.</div>', unsafe_allow_html=True)
        return

    _prospect_names = df["player_name"].dropna().tolist()
    name_lookup = {normalize_player_name(n): n for n in _prospect_names}
    # Last-name fallback: "Vega Ioane" → last token "ioane" → "Olaivavega Ioane"
    _lastname_lookup = {}
    for n in _prospect_names:
        parts = n.strip().split()
        if parts:
            _lastname_lookup.setdefault(parts[-1].lower(), []).append(n)

    # Fuzzy fallback: for live-board names that don't exact-match after normalization
    try:
        from rapidfuzz import process as _rfp, fuzz as _rff
        _fuzzy_cache = {}
        def _fuzzy_match_name(raw):
            key = normalize_player_name(raw)
            if key in name_lookup:
                return name_lookup[key]
            if key in _fuzzy_cache:
                return _fuzzy_cache[key]
            # Last-name exact fallback
            raw_parts = str(raw).strip().split()
            if raw_parts:
                last = raw_parts[-1].lower()
                ln_matches = _lastname_lookup.get(last, [])
                if len(ln_matches) == 1:
                    _fuzzy_cache[key] = ln_matches[0]
                    return ln_matches[0]
            # Full fuzzy — use both token_sort and partial, take best
            keys = list(name_lookup.keys())
            r1 = _rfp.extractOne(key, keys, scorer=_rff.token_sort_ratio)
            r2 = _rfp.extractOne(key, keys, scorer=_rff.partial_ratio)
            best = max([r for r in [r1, r2] if r], key=lambda x: x[1], default=None)
            matched = name_lookup[best[0]] if best and best[1] >= 80 else None
            _fuzzy_cache[key] = matched
            return matched
    except ImportError:
        def _fuzzy_match_name(raw):
            return name_lookup.get(normalize_player_name(raw))
    team_meta_lookup = load_nfl_team_meta()
    h1, h2, h3, h4, h5, h6 = st.columns([0.62, 0.86, 1.18, 1.75, 0.68, 0.92], vertical_alignment="center")
    with h1:
        st.markdown('<div class="live-head-cell">Round</div>', unsafe_allow_html=True)
    with h2:
        st.markdown('<div class="live-head-cell">Pick</div>', unsafe_allow_html=True)
    with h3:
        st.markdown('<div class="live-head-cell">NFL Team</div>', unsafe_allow_html=True)
    with h4:
        st.markdown('<div class="live-head-cell">Player</div>', unsafe_allow_html=True)
    with h5:
        st.markdown('<div class="live-head-cell">Pos</div>', unsafe_allow_html=True)
    with h6:
        st.markdown('<div class="live-head-cell">College</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    for _, draft_row in live_board.iterrows():
        round_txt = int(draft_row["Round"])
        pick_txt = int(draft_row["Pick"])
        team_txt = html.escape(str(draft_row["NFL Team"]))
        team_meta = team_meta_lookup.get(normalize_team_name(draft_row["NFL Team"]), {})
        team_logo_url = team_meta.get("logo")
        row_bg = (
            hex_to_rgba(team_meta.get("color_1"), 0.28)
            or "rgba(11, 18, 29, 0.9)"
        )
        row_border = (
            hex_to_rgba(team_meta.get("color_1"), 0.72)
            or "rgba(28,40,64,0.92)"
        )
        def clean_display_value(value):
            if pd.isna(value):
                return ""
            text = str(value).strip()
            if text.lower() in ["nan", "none", "<na>", "nat"]:
                return ""
            return text

        player_txt = clean_display_value(draft_row["Player"])
        pos_txt = html.escape(clean_display_value(draft_row["Pos"]))
        college_txt = html.escape(clean_display_value(draft_row["College"]))
        match_name = _fuzzy_match_name(player_txt) if player_txt else None
        cell_style = f' style="background:{row_bg};border-color:{row_border};"'

        c1, c2, c3, c4, c5, c6 = st.columns([0.62, 0.86, 1.18, 1.75, 0.68, 0.92], vertical_alignment="center")
        with c1:
            st.markdown(f'<div class="live-body-cell"{cell_style}><div class="live-cell-main">R{round_txt}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="live-body-cell"{cell_style}><div class="live-cell-main">{pick_txt}</div></div>', unsafe_allow_html=True)
        with c3:
            if team_logo_url:
                st.markdown(
                    f'<div class="live-body-cell"{cell_style}><img src="{html.escape(str(team_logo_url))}" '
                    f'style="max-height:32px;max-width:110px;object-fit:contain;display:block;margin:0 auto;" '
                    f'alt="{team_txt}" title="{team_txt}" /></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f'<div class="live-body-cell"{cell_style}><div class="live-cell-sub">{team_txt}</div></div>', unsafe_allow_html=True)
        with c4:
            if match_name:
                qp_name = quote(match_name)
                st.markdown(
                    f'<div class="live-body-cell"{cell_style}>'
                    f'<a class="live-player-link" href="?player={qp_name}" target="_self">{html.escape(player_txt)}</a>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f'<div class="live-body-cell"{cell_style}><div class="live-cell-main">{html.escape(player_txt) if player_txt else "—"}</div></div>',
                            unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="live-body-cell"{cell_style}><div class="live-cell-sub">{pos_txt or "—"}</div></div>', unsafe_allow_html=True)
        with c6:
            st.markdown(f'<div class="live-body-cell"{cell_style}><div class="live-cell-sub">{college_txt or "—"}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sent-empty" style="padding-top:8px;">Clicking a matched player preloads the Player Card selection.</div>',
                unsafe_allow_html=True)


# ── Tabs / Player Deeplink Router ──────────────────────────────────────────────

qp_player = st.query_params.get("player")
player_deeplink_active = False

if qp_player:
    player_match = df[df["player_name"] == qp_player]
    if len(player_match) > 0:
        st.session_state["pos_pills"] = player_match.iloc[0]["position"]
        st.session_state["player_sel"] = qp_player
        player_deeplink_active = True

st.markdown("""
<div class="brand-row">
  <a class="brand-link" href="?" target="_self">
    <span class="draft">draft</span><span class="intel">Intel.</span>
  </a>
</div>
""", unsafe_allow_html=True)

tab0, tab6, tab7, tab4, tab1, tab2, tab3, tab5 = st.tabs([
    "NFL Draft Live",
    "Team Board",
    "Top Undrafted",
    "Player Card",
    "Class Overview",
    "Draft Board",
    "Scouting Language",
    "Contract Outlook",
])

if player_deeplink_active:
    components.html(
        """
        <script>
        function clickPlayerCardTab() {
            const doc = window.parent.document;
            const tabs = Array.from(doc.querySelectorAll('button[data-baseweb="tab"]'));
            const playerTab = tabs.find(tab => (tab.innerText || "").trim() === "Player Card");

            if (playerTab && playerTab.getAttribute("aria-selected") !== "true") {
                playerTab.click();
            }
        }

        setTimeout(clickPlayerCardTab, 50);
        setTimeout(clickPlayerCardTab, 250);
        setTimeout(clickPlayerCardTab, 750);
        setTimeout(clickPlayerCardTab, 1250);
        </script>
        """,
        height=0,
    )

# ── Load live board once, share across tabs ────────────────────────────────
live_board = pd.DataFrame()
live_error = None
try:
    live_board = load_live_wikipedia_board()
except Exception as exc:
    live_error = str(exc)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 0 — MAIN BOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab0:
    render_live_board(df, live_board, live_error)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — TEAM BOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown('<div class="sec-lbl" style="margin-top:0">Team Draft Board</div>', unsafe_allow_html=True)
    if live_board.empty:
        st.markdown('<div class="sent-empty">No draft picks available yet.</div>', unsafe_allow_html=True)
    else:
        # Build name→model-row lookup
        _tb_name_lookup = {normalize_player_name(n): n for n in df["player_name"].dropna().tolist()}
        try:
            from rapidfuzz import process as _rfp_tb, fuzz as _rff_tb
            _tb_fuzzy_cache = {}
            def _tb_match(raw):
                key = normalize_player_name(raw)
                if key in _tb_name_lookup:
                    return _tb_name_lookup[key]
                if key in _tb_fuzzy_cache:
                    return _tb_fuzzy_cache[key]
                keys = list(_tb_name_lookup.keys())
                r1 = _rfp_tb.extractOne(key, keys, scorer=_rff_tb.token_sort_ratio)
                r2 = _rfp_tb.extractOne(key, keys, scorer=_rff_tb.partial_ratio)
                best = max([r for r in [r1, r2] if r], key=lambda x: x[1], default=None)
                result = _tb_name_lookup[best[0]] if best and best[1] >= 80 else None
                _tb_fuzzy_cache[key] = result
                return result
        except ImportError:
            def _tb_match(raw):
                return _tb_name_lookup.get(normalize_player_name(raw))

        # Team selector
        _teams = sorted(live_board["NFL Team"].dropna().unique().tolist())
        _sel_team = st.selectbox("Select Team", _teams, key="team_board_sel")

        _team_picks = live_board[live_board["NFL Team"] == _sel_team].copy()
        team_meta_tb = load_nfl_team_meta()
        _tmeta = team_meta_tb.get(normalize_team_name(_sel_team), {})
        _team_color = _tmeta.get("color_1") or "#3b82f6"
        _team_logo  = _tmeta.get("team_logo_espn") or _tmeta.get("logo")

        # Team header
        logo_html = f'<img src="{html.escape(str(_team_logo))}" style="height:48px;object-fit:contain;vertical-align:middle;margin-right:12px;" />' if _team_logo else ""
        st.markdown(
            f'<div style="display:flex;align-items:center;margin:12px 0 20px;">'
            f'{logo_html}<span style="font-size:1.4rem;font-weight:700;color:#e2e8f0;">{html.escape(_sel_team)}</span>'
            f'<span style="margin-left:12px;color:#64748b;font-size:0.95rem;">{len(_team_picks)} pick{"s" if len(_team_picks)!=1 else ""}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Predicted vs Actual summary badges
        _pred_counts   = {"riser": 0, "consensus": 0, "slide": 0}
        _actual_counts = {"riser": 0, "consensus": 0, "slide": 0}
        for _, _pr in _team_picks.iterrows():
            if not pd.notna(_pr.get("Player")): continue
            _mn = _tb_match(str(_pr["Player"]))
            if not _mn: continue
            _mr = df[df["player_name"] == _mn]
            if not len(_mr): continue
            _mr = _mr.iloc[0]
            _pt = _mr["draft_prediction"]
            if _pt in _pred_counts:
                _pred_counts[_pt] += 1
            _c = int(_mr["consensus"]) if pd.notna(_mr["consensus"]) else None
            _at = compute_actual_tier(_c, int(_pr["Pick"]))
            if _at:
                _actual_counts[_at] += 1

        def _badge_row(label, counts):
            badges = "".join([
                f'<span style="background:{DRAFT_COLORS.get(t,"#334155")};color:#fff;border-radius:6px;'
                f'padding:3px 10px;font-size:0.8rem;font-weight:600;margin-right:6px;">'
                f'{DRAFT_LABELS.get(t, t.title() if isinstance(t, str) else "—")}: {c}</span>'
                for t, c in counts.items() if c > 0
            ])
            return (
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">'
                f'<span style="color:#64748b;font-size:0.75rem;font-weight:600;min-width:64px;">{label}</span>'
                f'{badges}</div>'
            )

        st.markdown(
            _badge_row("Predicted", _pred_counts) + _badge_row("Actual", _actual_counts),
            unsafe_allow_html=True
        )

        _pick_meta = load_nfl_team_meta().get(normalize_team_name(_sel_team), {})
        for _pick_i, (_, _pr) in enumerate(_team_picks.iterrows()):
            _player_raw = str(_pr["Player"]) if pd.notna(_pr["Player"]) else ""
            if not _player_raw or _player_raw.lower() in ("nan", "none", ""):
                continue
            _mn   = _tb_match(_player_raw)
            _mrow = df[df["player_name"] == _mn].iloc[0] if _mn and len(df[df["player_name"] == _mn]) else None
            _tier     = _mrow["draft_prediction"] if _mrow is not None else None
            _tier_col = DRAFT_COLORS.get(_tier, "#475569") if _tier else "#475569"
            _tier_lbl = DRAFT_LABELS.get(_tier, "—") if _tier else "—"
            _cons_int = int(_mrow["consensus"]) if _mrow is not None and pd.notna(_mrow["consensus"]) else None
            _cons_val = f"#{_cons_int}" if _cons_int else "—"
            _p_s = f'{_mrow["p_slide"]:.0f}%' if _mrow is not None else "—"
            _p_r = f'{_mrow["p_riser"]:.0f}%' if _mrow is not None else "—"
            _pos_raw = str(_pr.get("Pos", "") or "")

            _toggle_key = f"tb_expand_{_sel_team}_{_pick_i}"
            if _toggle_key not in st.session_state:
                st.session_state[_toggle_key] = False

            _tier_badge_html = (
                f'<span style="background:{_tier_col};color:#fff;border-radius:4px;'
                f'padding:1px 8px;font-size:0.75rem;font-weight:600;margin-left:10px;">{_tier_lbl}</span>'
                if _tier else ""
            )
            _row_bg     = hex_to_rgba(_team_color, 0.12) or "rgba(11,18,29,0.9)"
            _row_border = hex_to_rgba(_team_color, 0.35) or "rgba(28,40,64,0.92)"

            _rc1, _rc2 = st.columns([0.92, 0.08], vertical_alignment="center")
            with _rc1:
                st.markdown(
                    f'<div style="background:{_row_bg};border:1px solid {_row_border};border-radius:8px;'
                    f'padding:10px 14px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
                    f'<span style="color:#64748b;font-size:0.8rem;min-width:32px;">R{int(_pr["Round"])}</span>'
                    f'<span style="font-weight:700;color:#f1f5f9;min-width:36px;">#{int(_pr["Pick"])}</span>'
                    f'<span style="font-weight:600;color:#e2e8f0;flex:1;">{html.escape(_player_raw)}</span>'
                    f'<span style="color:#64748b;font-size:0.82rem;">{html.escape(_pos_raw)}</span>'
                    f'<span style="color:#94a3b8;font-size:0.82rem;">{_cons_val}</span>'
                    f'{_tier_badge_html}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with _rc2:
                if st.button("▼" if not st.session_state[_toggle_key] else "▲",
                             key=f"btn_{_toggle_key}", use_container_width=True):
                    st.session_state[_toggle_key] = not st.session_state[_toggle_key]
                    st.rerun()

            if st.session_state[_toggle_key]:
                if _mrow is not None:
                    st.markdown(
                        f'<div style="display:flex;gap:16px;align-items:center;margin:4px 0 0 8px;flex-wrap:wrap;">'
                        f'<span style="background:{_tier_col};color:#fff;border-radius:5px;padding:3px 10px;font-size:0.82rem;font-weight:600;">{_tier_lbl}</span>'
                        f'<span style="color:#94a3b8;font-size:0.82rem;">Consensus {_cons_val}</span>'
                        f'<span style="color:#94a3b8;font-size:0.82rem;">P(Slide) {_p_s}</span>'
                        f'<span style="color:#94a3b8;font-size:0.82rem;">P(Riser) {_p_r}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                render_pick_card(
                    dr_round=int(_pr["Round"]),
                    dr_pick=int(_pr["Pick"]),
                    dr_team=_sel_team,
                    dr_meta=_pick_meta,
                    consensus=_cons_int,
                    predicted_tier=_tier,
                )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — TOP UNDRAFTED
# ══════════════════════════════════════════════════════════════════════════════

with tab7:
    st.markdown('<div class="sec-lbl" style="margin-top:0">Still on the Board</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#64748b;font-size:0.85rem;margin-bottom:16px;">'
        'Prospects in our model who have not been picked. Sorted by consensus rank.</div>',
        unsafe_allow_html=True
    )

    if live_board.empty:
        _picked_names = set()
    else:
        # Build set of matched prospect names already picked
        _ud_name_lookup = {normalize_player_name(n): n for n in df["player_name"].dropna().tolist()}
        try:
            from rapidfuzz import process as _rfp_ud, fuzz as _rff_ud
            _ud_picked = set()
            for _pn in live_board["Player"].dropna():
                key = normalize_player_name(str(_pn))
                if key in _ud_name_lookup:
                    _ud_picked.add(_ud_name_lookup[key])
                else:
                    keys = list(_ud_name_lookup.keys())
                    r1 = _rfp_ud.extractOne(key, keys, scorer=_rff_ud.token_sort_ratio)
                    r2 = _rfp_ud.extractOne(key, keys, scorer=_rff_ud.partial_ratio)
                    best = max([r for r in [r1, r2] if r], key=lambda x: x[1], default=None)
                    if best and best[1] >= 80:
                        _ud_picked.add(_ud_name_lookup[best[0]])
            _picked_names = _ud_picked
        except ImportError:
            _picked_names = {
                _ud_name_lookup[normalize_player_name(str(p))]
                for p in live_board["Player"].dropna()
                if normalize_player_name(str(p)) in _ud_name_lookup
            }

    _still_on_board = df[~df["player_name"].isin(_picked_names)].copy()

    # Position filter
    _ud_positions = ["All"] + sorted(_still_on_board["position"].dropna().unique().tolist())
    _ud_col1, _ud_col2 = st.columns([1, 3])
    with _ud_col1:
        _ud_pos_sel = st.selectbox("Position", _ud_positions, key="ud_pos_sel")
    with _ud_col2:
        _ud_tier_opts = ["All", "Slide", "Consensus", "Riser"]
        _ud_tier_sel = st.selectbox("Model Tier", _ud_tier_opts, key="ud_tier_sel")

    _ud_show = _still_on_board.copy()
    if _ud_pos_sel != "All":
        _ud_show = _ud_show[_ud_show["position"] == _ud_pos_sel]
    if _ud_tier_sel != "All":
        _ud_show = _ud_show[_ud_show["draft_prediction"] == _ud_tier_sel.lower()]
    _ud_show = _ud_show.sort_values("consensus").reset_index(drop=True)

    n_total_picked = len(live_board) if not live_board.empty else 0
    st.markdown(
        f'<div style="color:#94a3b8;font-size:0.82rem;margin-bottom:12px;">'
        f'{n_total_picked} prospects picked so far</div>',
        unsafe_allow_html=True
    )

    if _ud_show.empty:
        st.markdown('<div class="sent-empty">No prospects match this filter.</div>', unsafe_allow_html=True)
    else:
        for _, _ur in _ud_show.iterrows():
            _u_name    = _ur["player_name"]
            _u_pos     = _ur.get("position", "")
            _u_college = _ur.get("college", "") or ""
            _u_cons    = int(_ur["consensus"]) if pd.notna(_ur.get("consensus")) else None
            _u_br      = int(_ur["beast_rank"]) if pd.notna(_ur.get("beast_rank")) else None
            _u_grade   = str(_ur.get("beast_grade", "") or "").strip()
            _u_grade   = "" if _u_grade.lower() in ("nan", "none") else _u_grade
            _u_tier    = _ur.get("draft_prediction")
            _u_tc      = DRAFT_COLORS.get(_u_tier, "#475569") if _u_tier else "#475569"
            _u_tl      = DRAFT_LABELS.get(_u_tier, "—") if _u_tier else "—"
            _u_ps      = float(_ur["p_slide"])   if pd.notna(_ur.get("p_slide"))   else 0
            _u_pr      = float(_ur["p_riser"])   if pd.notna(_ur.get("p_riser"))   else 0
            _u_pc      = float(_ur["p_consensus"]) if pd.notna(_ur.get("p_consensus")) else 0
            _u_sc      = float(_ur.get("scout_conf", 0) or 0)
            _u_sc_col  = "#22c55e" if _u_sc >= 60 else "#3b82f6" if _u_sc >= 30 else "#64748b"
            _u_comp    = str(_ur.get("br_pro_comparison", "") or "").strip()
            _u_comp    = "" if _u_comp.lower() in ("nan", "none") else _u_comp
            _u_logo    = _ur.get("college_logo_url", "")
            _u_hs      = _ur.get("headshot_url", "")

            _u_badges = f'<span class="badge b-pos">{html.escape(_u_pos)}</span>'
            if _u_college:
                _u_badges += f'<span class="badge b-info">{html.escape(_u_college)}</span>'
            if _u_cons:
                _u_badges += f'<span class="badge b-info">Consensus #{_u_cons}</span>'
            if _u_br:
                _u_badges += f'<span class="badge b-src">Beast #{_u_br}</span>'
            if _u_grade:
                _u_badges += f'<span class="badge b-src">Beast {html.escape(_u_grade)}</span>'
            if _u_comp and str(_u_comp) not in ("", "nan"):
                _u_badges += f'<span class="badge b-info">Comp: {html.escape(str(_u_comp))}</span>'

            _u_tier_pill = (
                f'<span style="background:{_u_tc}22;color:{_u_tc};border:1px solid {_u_tc}55;'
                f'border-radius:6px;padding:3px 10px;font-size:0.82rem;font-weight:600;">{_u_tl}</span>'
            )
            _u_logo_html = college_logo_block(_u_logo, _u_college, size=52) if _u_logo else ""

            _u_col1, _u_col2 = st.columns([1, 4])
            with _u_col1:
                show_player_logo_panel(_u_name, _u_college,
                                       college_logo_url=_u_logo if _u_logo else None,
                                       headshot_url=_u_hs if _u_hs else None)
            with _u_col2:
                st.markdown(f"""
                <div class="p-header" style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;">
                  <div style="flex:1;min-width:0;">
                    <p class="p-name" style="font-size:1.2rem;margin-bottom:8px;">{html.escape(_u_name)}</p>
                    {_u_badges}
                    <div style="margin-top:10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                      <div class="chart-lbl" style="margin:0;">DRAFT TIER</div>
                      {_u_tier_pill}
                    </div>
                    <div style="margin-top:10px;display:flex;gap:16px;flex-wrap:wrap;">
                      <span style="font-size:0.8rem;color:#64748b;">P(Slide) <span style="color:#ef4444;font-weight:600;">{_u_ps:.0f}%</span></span>
                      <span style="font-size:0.8rem;color:#64748b;">P(Riser) <span style="color:#22c55e;font-weight:600;">{_u_pr:.0f}%</span></span>
                      <span style="font-size:0.8rem;color:#64748b;">P(Consensus) <span style="color:#3b82f6;font-weight:600;">{_u_pc:.0f}%</span></span>
                      <span style="font-size:0.8rem;color:#64748b;">Scout Conf <span style="color:{_u_sc_col};font-weight:600;">{_u_sc:.0f}</span></span>
                    </div>
                  </div>
                  {_u_logo_html}
                </div>
                """, unsafe_allow_html=True)

            qp = quote(_u_name)
            st.markdown(
                f'<a href="?player={qp}" target="_self" style="font-size:0.78rem;color:#475569;'
                f'text-decoration:none;margin-left:4px;">→ Full player card</a>',
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CLASS OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    n_total   = len(df)
    _draftable_mask = df["consensus"].apply(lambda x: pd.notna(x) and int(x) <= 257)
    n_slides  = ((df["draft_prediction"] == "slide") & _draftable_mask).sum()
    n_reaches = ((df["draft_prediction"] == "riser") & _draftable_mask).sum()
    n_3source = (df["ngram_count"] > 0).sum()
    pct_3src  = n_3source / n_total * 100

    st.markdown(f"""
    <div class="hero-grid">
      <div class="metric-card">
        <div class="m-val">{n_total}</div>
        <div class="m-lbl">Prospects Evaluated</div>
        <div class="m-sub">Beast / PFF / BR text available</div>
      </div>
      <div class="metric-card">
        <div class="m-val accent">{n_slides}</div>
        <div class="m-lbl">Predicted Slides</div>
        <div class="m-sub">Expected to fall vs. consensus</div>
      </div>
      <div class="metric-card">
        <div class="m-val riser">{n_reaches}</div>
        <div class="m-lbl">Predicted Risers</div>
        <div class="m-sub">Model says above consensus value</div>
      </div>
      <div class="metric-card">
        <div class="m-val">{n_3source}</div>
        <div class="m-lbl">Multi-Source Scouted</div>
        <div class="m-sub">{pct_3src:.0f}% with cross-validated phrases</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Predicted vs Actual outcome badges (draftable players with live board data) ──
    if not live_board.empty:
        _co_name_lookup = {normalize_player_name(n): n for n in df["player_name"].dropna().tolist()}
        try:
            from rapidfuzz import process as _rfp_co, fuzz as _rff_co
            def _co_match(raw):
                key = normalize_player_name(raw)
                if key in _co_name_lookup: return _co_name_lookup[key]
                keys = list(_co_name_lookup.keys())
                r1 = _rfp_co.extractOne(key, keys, scorer=_rff_co.token_sort_ratio)
                r2 = _rfp_co.extractOne(key, keys, scorer=_rff_co.partial_ratio)
                best = max([r for r in [r1, r2] if r], key=lambda x: x[1], default=None)
                return _co_name_lookup[best[0]] if best and best[1] >= 80 else None
        except ImportError:
            def _co_match(raw): return _co_name_lookup.get(normalize_player_name(raw))

        _co_pred = {"riser": 0, "consensus": 0, "slide": 0}
        _co_actual = {"riser": 0, "consensus": 0, "slide": 0}
        for _, _lbr in live_board.iterrows():
            if not pd.notna(_lbr.get("Player")): continue
            _mn = _co_match(str(_lbr["Player"]))
            if not _mn: continue
            _mr = df[df["player_name"] == _mn]
            if not len(_mr): continue
            _mr = _mr.iloc[0]
            _pt = _mr["draft_prediction"]
            if _pt in _co_pred:
                _co_pred[_pt] += 1
            _c = int(_mr["consensus"]) if pd.notna(_mr["consensus"]) else None
            _at = compute_actual_tier(_c, int(_lbr["Pick"]))
            if _at:
                _co_actual[_at] += 1

        def _co_badge_row(label, counts):
            badges = "".join([
                f'<span style="background:{DRAFT_COLORS.get(t,"#334155")};color:#fff;border-radius:6px;'
                f'padding:3px 10px;font-size:0.8rem;font-weight:600;margin-right:6px;">'
                f'{DRAFT_LABELS.get(t, t.title() if isinstance(t, str) else "—")}: {c}</span>'
                for t, c in counts.items() if c > 0
            ])
            return (
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">'
                f'<span style="color:#64748b;font-size:0.75rem;font-weight:600;min-width:64px;">{label}</span>'
                f'{badges}</div>'
            )

        _drafted_so_far = sum(_co_pred.values())
        st.markdown(
            f'<div style="margin:12px 0 4px;color:#94a3b8;font-size:0.78rem;">'
            f'{_drafted_so_far} draftable prospects picked so far</div>'
            + _co_badge_row("Predicted", _co_pred)
            + _co_badge_row("Actual", _co_actual),
            unsafe_allow_html=True
        )

    # ── Donuts ──
    c1, c2 = st.columns(2)

    # Exclude projected undrafted players from tier distribution charts —
    # model tiers are not meaningful for consensus > 257
    _draftable = df[df["consensus"].apply(lambda x: pd.notna(x) and int(x) <= 257)]

    with c1:
        dc = _draftable["draft_prediction"].value_counts().reset_index()
        dc.columns = ["tier", "count"]
        dc["color"] = dc["tier"].map(DRAFT_COLORS)
        dc["label"] = dc["tier"].map(DRAFT_LABELS)
        fig = go.Figure(go.Pie(
            labels=dc["label"], values=dc["count"], hole=0.62,
            marker=dict(colors=dc["color"].tolist(), line=dict(color=BG, width=3)),
            textinfo="label+percent",
            textfont=dict(color="white", size=12),
            hovertemplate="<b>%{label}</b><br>%{value} players · %{percent}<extra></extra>",
        ))
        fig.add_annotation(text="<b>Draft Tier</b>", x=0.5, y=0.5,
                           font=dict(color="white", size=13), showarrow=False)
        styled(fig, height=400, title_text="Draft Outcome Distribution",
               showlegend=False, margin=dict(l=16, r=16, t=50, b=16))
        st.plotly_chart(fig, use_container_width=True, key="draft_donut")

    with c2:
        # Source coverage donut — accurate 5-way split
        is_beast  = df["text_source"].eq("beast")
        is_pff    = df["text_source"].eq("pff")
        has_br_s  = df["has_br"].astype(bool)  if "has_br"  in df.columns else pd.Series(False, index=df.index)
        has_pff_s = df["has_pff"].astype(bool) if "has_pff" in df.columns else pd.Series(False, index=df.index)
        n_beast_only    = (is_beast & ~has_pff_s & ~has_br_s).sum()
        n_beast_pff     = (is_beast &  has_pff_s & ~has_br_s).sum()
        n_beast_br      = (is_beast & ~has_pff_s &  has_br_s).sum()
        n_beast_pff_br  = (is_beast &  has_pff_s &  has_br_s).sum()
        n_pff_only      = (is_pff   & ~has_br_s).sum()
        src_labels = ["Beast Only","Beast + PFF","Beast + BR","Beast + PFF + BR","PFF Only"]
        src_vals   = [n_beast_only, n_beast_pff, n_beast_br, n_beast_pff_br, n_pff_only]
        src_colors = ["#1e3a5f","#2d5a8f","#2d4a7a","#3d6aaa","#334155"]
        fig = go.Figure(go.Pie(
            labels=src_labels, values=[max(v, 0) for v in src_vals], hole=0.62,
            marker=dict(colors=src_colors, line=dict(color=BG, width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value} players (%{percent})<extra></extra>",
        ))
        fig.add_annotation(text="<b>Scout Coverage</b>", x=0.5, y=0.5,
                           font=dict(color="white", size=13), showarrow=False)
        styled(fig, height=400, title_text="Source Coverage",
               showlegend=True,
               legend=dict(orientation="v", x=1.02, y=0.5,
                           font=dict(color="#94a3b8", size=11)),
               margin=dict(l=16, r=130, t=50, b=16))
        st.plotly_chart(fig, use_container_width=True, key="src_donut")

    # ── Draft tier by position ──
    pos_grp = _draftable.groupby(["position","draft_prediction"]).size().reset_index(name="n")
    pos_order = (_draftable.groupby("position")["p_riser"].mean()
                   .sort_values(ascending=False).index.tolist())
    fig = go.Figure()
    for tier in ["riser","consensus","slide"]:
        sub = pos_grp[pos_grp["draft_prediction"] == tier]
        fig.add_trace(go.Bar(
            x=sub["position"], y=sub["n"],
            name=DRAFT_LABELS[tier],
            marker_color=DRAFT_COLORS[tier],
            hovertemplate="<b>%{x}</b> — " + DRAFT_LABELS[tier] + "<br>%{y} players<extra></extra>",
        ))
    styled(fig, height=320, title_text="Draft Tier by Position Group",
           barmode="stack",
           xaxis=dict(categoryorder="array", categoryarray=pos_order,
                      gridcolor="#0d2a52", linecolor="#1a3a6b"),
           yaxis=dict(gridcolor="#0d2a52", linecolor="#1a3a6b"))
    st.plotly_chart(fig, use_container_width=True, key="pos_stack")

    # ── Rise/Fall scatter ──
    rf_df = _draftable[_draftable["rise_fall"].notna()].copy()
    rf_df["rise_fall_fmt"] = rf_df["rise_fall"].round(1)
    rf_df["Draft Tier"] = rf_df["draft_prediction"].map(DRAFT_LABELS)
    rf_df["scout_conf_sz"] = rf_df["scout_conf"].clip(lower=1).astype(float)
    _has_beast_rank = "beast_rank" in rf_df.columns and rf_df["beast_rank"].notna().any()
    fig = px.scatter(
        rf_df,
        x="consensus", y="rise_fall",
        color="draft_prediction",
        color_discrete_map=DRAFT_COLORS,
        size="scout_conf_sz",
        size_max=16,
        hover_name="player_name",
        hover_data={"consensus": True, "rise_fall": ":.0f",
                    "position": True, "Draft Tier": True, "draft_prediction": False,
                    **( {"beast_rank": True, "consensus_pos_rank": ":.0f"} if _has_beast_rank else {}),
                    "scout_conf": ":.0f", "scout_conf_sz": False},
        labels={"consensus": "Consensus Rank", "rise_fall": "Rise/Fall Score",
                "draft_prediction": "Draft Tier", "scout_conf": "Scout Confidence"},
        title="Rise / Fall — Consensus Rank vs Beast Model Delta  (bubble = scout confidence)",
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0)))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1)
    fig.add_annotation(x=rf_df["consensus"].max() * 0.8, y=rf_df["rise_fall"].max() * 0.85,
                       text="Rising stock", font=dict(color="rgba(34,197,94,0.5)", size=11),
                       showarrow=False)
    fig.add_annotation(x=rf_df["consensus"].max() * 0.8, y=rf_df["rise_fall"].min() * 0.85,
                       text="Falling stock", font=dict(color="rgba(213,10,10,0.5)", size=11),
                       showarrow=False)
    for tier in ["riser","consensus","slide"]:
        fig.for_each_trace(lambda t, _tier=tier: t.update(name=DRAFT_LABELS[_tier])
                           if t.name == _tier else None)
    styled(fig, height=440)
    st.plotly_chart(fig, use_container_width=True, key="rise_fall_scatter")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — DRAFT BOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab2:

    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        pos_opts = sorted(df["position"].dropna().unique())
        sel_pos  = st.multiselect("Position", pos_opts, default=[], key="v_pos")
    with fc2:
        tier_opts = ["Riser","Consensus","Slide"]
        sel_tier  = st.multiselect("Draft Tier", tier_opts, default=[], key="v_tier")
    with fc3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Reset", key="v_reset"):
            sel_pos, sel_tier = [], []

    fdf = _draftable.copy()
    if sel_pos:
        fdf = fdf[fdf["position"].isin(sel_pos)]
    label_to_tier = {v: k for k, v in DRAFT_LABELS.items()}
    if sel_tier:
        fdf = fdf[fdf["draft_prediction"].isin([label_to_tier.get(t, t.lower()) for t in sel_tier])]

    # ── Main scatter: consensus vs P(slide), sized by rise_fall magnitude ──
    fig = px.scatter(
        fdf,
        x="consensus", y="p_slide",
        color="draft_prediction",
        color_discrete_map=DRAFT_COLORS,
        hover_name="player_name",
        hover_data={"consensus": True, "p_slide": ":.1f",
                    "p_riser": ":.1f", "position": True,
                    "beast_rank": True, "rise_fall": ":.0f"},
        labels={"consensus": "Consensus Rank", "p_slide": "P(Slide) %",
                "draft_prediction": "Draft Tier"},
        title="Draft Board — Consensus Rank vs P(Slide)",
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0)))
    fig.add_hline(y=33, line_dash="dot", line_color="rgba(255,255,255,0.12)", line_width=1)
    fig.add_annotation(x=fdf["consensus"].max() * 0.75, y=70,
                       text="High slide risk", font=dict(color="rgba(213,10,10,0.4)", size=11),
                       showarrow=False)
    for tier in ["riser","consensus","slide"]:
        fig.for_each_trace(lambda t, _tier=tier: t.update(name=DRAFT_LABELS[_tier])
                           if t.name == _tier else None)
    styled(fig, height=400)
    st.plotly_chart(fig, use_container_width=True, key="board_scatter")

    bc1, bc2, bc3 = st.columns(3)

    with bc1:
        # Top predicted risers by model P(riser)
        risers = (fdf[fdf["p_riser"].notna()]
                  .nlargest(15, "p_riser")
                  [["player_name","position","consensus","p_riser","p_slide","p_consensus","draft_prediction"]])
        fig = go.Figure(go.Bar(
            x=risers["p_riser"],
            y=risers["player_name"],
            orientation="h",
            marker_color="#22c55e",
            text=[f"#{int(c) if pd.notna(c) else '?'} consensus | {p:.0f}% riser"
                  for c, p in zip(risers["consensus"], risers["p_riser"])],
            textposition="outside",
            textfont=dict(color="rgba(255,255,255,0.45)", size=9),
            hovertemplate="<b>%{y}</b><br>P(Riser): %{x:.1f}%<extra></extra>",
        ))
        styled(fig, height=400, title_text="Top Predicted Risers",
               yaxis=dict(autorange="reversed", gridcolor="#0d2a52", linecolor="#1a3a6b"),
               xaxis=dict(gridcolor="#0d2a52", linecolor="#1a3a6b", ticksuffix="%"),
               margin=dict(l=16, r=130, t=44, b=16))
        st.plotly_chart(fig, use_container_width=True, key="risers_bar")

    with bc2:
        # Top predicted sliders by model P(slide)
        sliders = (fdf[fdf["p_slide"].notna()]
                   .nlargest(15, "p_slide")
                   [["player_name","position","consensus","p_slide","p_riser","p_consensus","draft_prediction"]])
        fig = go.Figure(go.Bar(
            x=sliders["p_slide"],
            y=sliders["player_name"],
            orientation="h",
            marker_color=NFL_RED,
            text=[f"#{int(c) if pd.notna(c) else '?'} consensus | {p:.0f}% slide"
                  for c, p in zip(sliders["consensus"], sliders["p_slide"])],
            textposition="outside",
            textfont=dict(color="rgba(255,255,255,0.45)", size=9),
            hovertemplate="<b>%{y}</b><br>P(Slide): %{x:.1f}%<extra></extra>",
        ))
        styled(fig, height=400, title_text="Top Predicted Sliders",
               yaxis=dict(autorange="reversed", gridcolor="#0d2a52", linecolor="#1a3a6b"),
               xaxis=dict(gridcolor="#0d2a52", linecolor="#1a3a6b", ticksuffix="%"),
               margin=dict(l=16, r=130, t=44, b=16))
        st.plotly_chart(fig, use_container_width=True, key="sliders_bar")

    with bc3:
        # Sleepers: undrafted consensus but Beast ranks them within the draftable range at their position
        _beast_cutoff = (
            _draftable[_draftable["beast_rank"].notna()]
            .groupby("position")["beast_rank"]
            .max()
            .rename("beast_draft_cutoff")
        )
        _undrafted = fdf[fdf["consensus"].apply(lambda x: pd.notna(x) and int(x) > 257)].copy()
        _undrafted = _undrafted.merge(_beast_cutoff.reset_index(), on="position", how="left")
        sleepers = (
            _undrafted[
                _undrafted["beast_rank"].notna() &
                _undrafted["beast_draft_cutoff"].notna() &
                (_undrafted["beast_rank"] <= _undrafted["beast_draft_cutoff"])
            ]
            .nlargest(15, "rise_fall")
            [["player_name","position","consensus","beast_rank","beast_draft_cutoff","rise_fall"]]
        )
        if len(sleepers):
            fig = go.Figure(go.Bar(
                x=sleepers["rise_fall"],
                y=sleepers["player_name"],
                orientation="h",
                marker_color="#f59e0b",
                text=[f"#{int(b):.0f} Beast pos / consensus #{int(c)}"
                      for b, c in zip(sleepers["beast_rank"], sleepers["consensus"])],
                textposition="outside",
                textfont=dict(color="rgba(255,255,255,0.45)", size=9),
                hovertemplate="<b>%{y}</b><br>Beast rise score: %{x:.0f}<extra></extra>",
            ))
            styled(fig, height=400, title_text="Sleepers (Undrafted → Beast Says Drafted)",
                   yaxis=dict(autorange="reversed", gridcolor="#0d2a52", linecolor="#1a3a6b"),
                   xaxis=dict(gridcolor="#0d2a52", linecolor="#1a3a6b"),
                   margin=dict(l=16, r=150, t=44, b=16))
            st.plotly_chart(fig, use_container_width=True, key="sleepers_bar")
        else:
            st.markdown('<div style="color:#64748b;padding:2rem 0">No sleepers found for this filter.</div>',
                        unsafe_allow_html=True)

    # ── Full table ──
    st.markdown('<div class="sec-lbl">All Prospects</div>', unsafe_allow_html=True)
    show_cols = ["player_name","position","college","consensus","beast_rank","rise_fall",
                 "draft_prediction","p_slide","p_consensus","p_riser","ngram_count","scout_conf"]
    rename_map = {
        "player_name":"Player","position":"Pos","college":"College",
        "consensus":"Consensus","beast_rank":"Beast Rank","rise_fall":"Rise/Fall",
        "draft_prediction":"Draft Tier","p_slide":"P(Slide) %",
        "p_consensus":"P(Consensus) %","p_riser":"P(Riser) %",
        "ngram_count":"Cross-Source Phrases","scout_conf":"Scout Conf",
    }
    tbl = fdf[show_cols].rename(columns=rename_map).sort_values("Consensus")
    st.dataframe(
        tbl,
        use_container_width=True,
        height=380,
        column_config={
            "P(Slide) %":     st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "P(Consensus) %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "P(Riser) %":     st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "Scout Conf":     st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        },
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — SCOUTING LANGUAGE
# ══════════════════════════════════════════════════════════════════════════════

with tab3:

    # ── Hero metrics ──
    total_phrases   = len(ng)
    avg_phrases     = ng.groupby("Player.Name").size().mean()
    n_3src_players  = (ng[ng["sources_present"] >= 3]["Player.Name"].nunique())
    max_phrases     = ng.groupby("Player.Name").size().max()

    st.markdown(f"""
    <div class="hero-grid">
      <div class="metric-card">
        <div class="m-val">{total_phrases}</div>
        <div class="m-lbl">Cross-Source Phrases</div>
        <div class="m-sub">Validated across 2+ scouting sources</div>
      </div>
      <div class="metric-card">
        <div class="m-val accent">{n_3source}</div>
        <div class="m-lbl">Players Covered</div>
        <div class="m-sub">With at least 1 cross-source phrase</div>
      </div>
      <div class="metric-card">
        <div class="m-val">{avg_phrases:.1f}</div>
        <div class="m-lbl">Avg Phrases / Player</div>
        <div class="m-sub">Among scouted prospects</div>
      </div>
      <div class="metric-card">
        <div class="m-val accent">{max_phrases}</div>
        <div class="m-lbl">Max Phrases (1 Player)</div>
        <div class="m-sub">Highest cross-source agreement</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    lc1, lc2 = st.columns(2)

    with lc1:
        # ── Top cross-source phrases (class-wide) ──
        top_ng = (ng.groupby("ngram")
                  .agg(total=("total_count","sum"),
                       players=("Player.Name","nunique"),
                       sources=("sources_present","max"))
                  .reset_index()
                  .sort_values(["sources","players","total"], ascending=False)
                  .head(20))
        top_ng["label"] = top_ng["ngram"]
        fig = go.Figure(go.Bar(
            x=top_ng["players"],
            y=top_ng["label"],
            orientation="h",
            marker_color=[NFL_RED if s >= 3 else "#3b82f6" for s in top_ng["sources"]],
            text=[f"{s} sources" for s in top_ng["sources"]],
            textposition="outside",
            textfont=dict(color="rgba(255,255,255,0.45)", size=9),
            hovertemplate="<b>%{y}</b><br>%{x} players · %{text}<extra></extra>",
        ))
        styled(fig, height=520, title_text="Top Phrases — Class-Wide Frequency",
               yaxis=dict(autorange="reversed", gridcolor="#0d2a52", linecolor="#1a3a6b"),
               xaxis=dict(title="# Players", gridcolor="#0d2a52", linecolor="#1a3a6b"),
               margin=dict(l=16, r=80, t=44, b=16))
        st.plotly_chart(fig, use_container_width=True, key="top_phrases_bar")

    with lc2:
        # ── Scout confidence leaderboard ──
        top_conf = (df[df["ngram_count"] > 0]
                    .nlargest(20, "ngram_count")
                    [["player_name","position","consensus","ngram_count","has_br","draft_prediction"]]
                    .sort_values("ngram_count"))
        top_conf["color"] = top_conf["draft_prediction"].map(DRAFT_COLORS).fillna("#3b82f6")
        fig = go.Figure(go.Bar(
            x=top_conf["ngram_count"],
            y=top_conf["player_name"],
            orientation="h",
            marker_color=top_conf["color"],
            text=[f"#{int(c)}" for c in top_conf["consensus"]],
            textposition="outside",
            textfont=dict(color="rgba(255,255,255,0.45)", size=9),
            hovertemplate="<b>%{y}</b><br>Cross-source phrases: %{x}<extra></extra>",
        ))
        styled(fig, height=520, title_text="Most Scouted — Cross-Source Phrase Count",
               yaxis=dict(autorange="reversed", gridcolor="#0d2a52", linecolor="#1a3a6b"),
               xaxis=dict(title="# Phrases", gridcolor="#0d2a52", linecolor="#1a3a6b"),
               margin=dict(l=16, r=60, t=44, b=16))
        st.plotly_chart(fig, use_container_width=True, key="scout_conf_bar")

    # ── Word signals by draft tier ──
    st.markdown('<div class="sec-lbl">Draft Model Word Signals by Tier</div>', unsafe_allow_html=True)

    wc1, wc2, wc3 = st.columns(3)
    tier_col_map = [
        ("riser",     "#22c55e", wc1),
        ("consensus", "#3b82f6", wc2),
        ("slide",     "#D50A0A", wc3),
    ]
    for tier, color, col in tier_col_map:
        tier_df = df[df["draft_prediction"] == tier]
        pos_words = aggregate_tier_words(tier_df, "draft_pos_words", positive=True)
        neg_words = aggregate_tier_words(tier_df, "draft_neg_words", positive=False)

        with col:
            st.markdown(f'<div style="font-size:11px;font-weight:700;color:{color};'
                        f'text-transform:uppercase;letter-spacing:1.5px;'
                        f'margin-bottom:12px;">{DRAFT_LABELS[tier]} ({len(tier_df)} players)</div>',
                        unsafe_allow_html=True)

            if pos_words:
                pos_df = pd.DataFrame(pos_words, columns=["word","avg_score","freq"])
                fig = go.Figure(go.Bar(
                    x=pos_df["avg_score"] * pos_df["freq"].apply(np.log1p),
                    y=pos_df["word"],
                    orientation="h",
                    marker_color=color,
                    hovertemplate="<b>%{y}</b><br>Score × freq: %{x:.3f}<extra></extra>",
                ))
                styled(fig, height=280, title_text="",
                       margin=dict(l=8, r=8, t=8, b=8),
                       yaxis=dict(autorange="reversed", tickfont=dict(size=10),
                                  gridcolor="#0d2a52", linecolor="#1a3a6b"),
                       xaxis=dict(showticklabels=False,
                                  gridcolor="#0d2a52", linecolor="#1a3a6b"))
                st.plotly_chart(fig, use_container_width=True, key=f"wt_{tier}")
            else:
                st.markdown('<div class="sent-empty">No signal data.</div>',
                            unsafe_allow_html=True)

    # ── Source coverage by position ──
    st.markdown('<div class="sec-lbl">Source Coverage by Position</div>', unsafe_allow_html=True)
    cov = df.groupby("position").agg(
        n_total=("player_name","count"),
        n_beast=("text_source", lambda x: (x == "beast").sum()),
        n_pff_text=("has_pff", "sum"),
        n_br=("has_br", "sum"),
    ).reset_index().sort_values("n_total", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Beast", x=cov["position"], y=cov["n_beast"],
                         marker_color="#1e3a5f",
                         hovertemplate="<b>%{x}</b><br>Beast primary: %{y}<extra></extra>"))
    fig.add_trace(go.Bar(name="+ PFF", x=cov["position"], y=cov["n_pff_text"],
                         marker_color="#2d5a8f",
                         hovertemplate="<b>%{x}</b><br>Also has PFF: %{y}<extra></extra>"))
    fig.add_trace(go.Bar(name="+ BR", x=cov["position"], y=cov["n_br"],
                         marker_color="#5a1a1a",
                         hovertemplate="<b>%{x}</b><br>Also has BR: %{y}<extra></extra>"))
    styled(fig, height=300, barmode="group",
           xaxis=dict(gridcolor="#0d2a52", linecolor="#1a3a6b"),
           yaxis=dict(gridcolor="#0d2a52", linecolor="#1a3a6b"))
    st.plotly_chart(fig, use_container_width=True, key="src_coverage_pos")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — PLAYER CARD
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    # ── Two-step search: position → player ──
    if st.query_params.get("player"):
        st.markdown(
            """
            <a href="?" target="_self" style="
                display:inline-flex;
                align-items:center;
                justify-content:center;
                padding:8px 14px;
                border-radius:8px;
                border:1px solid #1c2840;
                background:#0e1520;
                color:#f1f5f9;
                font-size:13px;
                font-weight:700;
                text-decoration:none;
                margin-bottom:18px;
            ">
                ← Back to NFL Draft Live
        </a>
        """,
        unsafe_allow_html=True,
    )
    POS_ORDER_DISPLAY = ["QB","RB","WR","TE","OT","IOL","DL","EDGE","LB","CB","S"]
    POS_FULL_NAMES = {
        "QB": "Quarterback",
        "RB": "Running Back",
        "WR": "Wide Receiver",
        "TE": "Tight End",
        "OT": "Offensive Tackle",
        "IOL": "Interior OL",
        "DL": "Defensive Line",
        "EDGE": "Edge Rusher",
        "LB": "Linebacker",
        "CB": "Cornerback",
        "S": "Safety",
    }
    all_pos   = df["position"].dropna().unique()
    positions = [p for p in POS_ORDER_DISPLAY if p in all_pos] + \
                [p for p in sorted(all_pos) if p not in POS_ORDER_DISPLAY]
    st.markdown('<div class="sec-lbl" style="margin-top:0">Filter by position</div>',
                unsafe_allow_html=True)
    sel_pos_card = st.radio("", positions, index=0, key="pos_pills",
                            label_visibility="collapsed", horizontal=True,
                            format_func=lambda p: POS_FULL_NAMES.get(p, p))

    pos_df = df[df["position"] == sel_pos_card].sort_values("consensus")
    st.markdown('<div class="sec-lbl">Select player</div>', unsafe_allow_html=True)
    sel = st.selectbox("", options=pos_df["player_name"].tolist(),
                       key="player_sel", label_visibility="collapsed")

    row    = df[df["player_name"] == sel].iloc[0]
    ng_row = ng[ng["Player.Name"] == sel].copy()

    college   = row.get("college", "") or ""
    consensus = int(row["consensus"]) if pd.notna(row.get("consensus")) else "?"
    beast_rnk = int(row["beast_rank"]) if pd.notna(row.get("beast_rank")) else None
    pos_rank  = int(row["position_rank"]) if pd.notna(row.get("position_rank")) else None
    bg_grade  = row.get("beast_grade", "")
    pro_comp  = row.get("br_pro_comparison", "")
    dc_color  = DRAFT_COLORS.get(row["draft_prediction"], "#3b82f6")
    rf_val    = row.get("rise_fall")
    _dp = row["draft_prediction"]
    pred_lbl  = DRAFT_LABELS.get(_dp, _dp.title() if isinstance(_dp, str) else "—")
    if isinstance(consensus, int) and consensus > 257:
        pred_lbl = "Undrafted"
        dc_color = "#64748b"
    # ── Resolve drafted_row early so we can show actual tier in header ──
    drafted_row = None
    if not live_board.empty:
        try:
            from rapidfuzz import process as _rfp2, fuzz as _rff2
            _lb_names = live_board["Player"].dropna().tolist()
            _r1 = _rfp2.extractOne(sel, _lb_names, scorer=_rff2.token_sort_ratio)
            _r2 = _rfp2.extractOne(sel, _lb_names, scorer=_rff2.partial_ratio)
            _res = max([r for r in [_r1, _r2] if r], key=lambda x: x[1], default=None)
            if _res and _res[1] >= 80:
                drafted_row = live_board[live_board["Player"] == _res[0]].iloc[0]
        except ImportError:
            _norm_sel = normalize_player_name(sel)
            for _, _lr in live_board.iterrows():
                if normalize_player_name(str(_lr["Player"])) == _norm_sel:
                    drafted_row = _lr
                    break

    _pc_actual_tier = None
    if drafted_row is not None:
        _pc_actual_tier = compute_actual_tier(
            consensus if isinstance(consensus, int) else None, int(drafted_row["Pick"])
        )

    # ── Player header ──
    badges_html = f'<span class="badge b-pos">{row["position"]}</span>'
    if college:
        badges_html += f'<span class="badge b-info">{html.escape(college)}</span>'
    badges_html += f'<span class="badge b-info">Consensus #{consensus}</span>'
    if beast_rnk:
        badges_html += f'<span class="badge b-src">Beast #{beast_rnk}</span>'
    if isinstance(bg_grade, str) and bg_grade.strip() and bg_grade.lower() not in ("nan", "none"):
        badges_html += f'<span class="badge b-src">Beast {html.escape(bg_grade)}</span>'
    if pro_comp and str(pro_comp).strip() not in ("", "nan"):
        badges_html += f'<span class="badge b-info">Comp: {html.escape(str(pro_comp))}</span>'

    _pred_pill = (
        f'<span style="background:{dc_color}22;color:{dc_color};'
        f'border:1px solid {dc_color}55;border-radius:6px;padding:3px 10px;font-size:0.82rem;font-weight:600;">'
        f'{pred_lbl}</span>'
    )
    _actual_pill = ""
    if _pc_actual_tier is not None:
        _ac = DRAFT_COLORS.get(_pc_actual_tier, "#475569")
        _al = DRAFT_LABELS.get(_pc_actual_tier, _pc_actual_tier.title() if isinstance(_pc_actual_tier, str) else "—")
        _actual_pill = (
            f'<span style="background:{_ac};color:#fff;'
            f'border-radius:6px;padding:3px 10px;font-size:0.82rem;font-weight:600;">{_al}</span>'
        )

    _pc_college_logo_url = row.get("college_logo_url")
    _pc_headshot_url     = row.get("headshot_url")
    _has_logo_url = pd.notna(_pc_college_logo_url) and str(_pc_college_logo_url).strip() not in ("", "nan")
    _pc_logo_html = college_logo_block(_pc_college_logo_url, college, size=60) if _has_logo_url else ""

    _ph1, _ph2 = st.columns([1, 3])
    with _ph1:
        show_player_logo_panel(sel, college, college_logo_url=_pc_college_logo_url,
                               headshot_url=_pc_headshot_url)
    with _ph2:
        st.markdown(f"""
        <div class="p-header" style="display:flex;align-items:flex-start;justify-content:space-between;gap:20px;">
          <div style="flex:1;min-width:0;">
            <p class="p-name">{html.escape(sel)}</p>
            {badges_html}
            <div style="margin-top:14px;display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start;">
              <div>
                <div class="chart-lbl">PREDICTED</div>
                <div style="margin-top:4px;">{_pred_pill}</div>
              </div>
              {f'<div><div class="chart-lbl">ACTUAL</div><div style="margin-top:4px;">{_actual_pill}</div></div>' if _actual_pill else ''}
            </div>
          </div>
          {_pc_logo_html}
        </div>
        """, unsafe_allow_html=True)

        pos_rank_str  = f"#{pos_rank}" if pos_rank is not None else "N/A"
        consensus_str = f"#{consensus}" if consensus != "?" else "N/A"
        st.markdown(f"""
        <div class="summary-grid">
          <div class="summary-card">
            <div class="summary-val">{consensus_str}</div>
            <div class="summary-lbl">Consensus Rank</div>
            <div class="summary-sub">Overall board placement</div>
          </div>
          <div class="summary-card">
            <div class="summary-val">{pos_rank_str}</div>
            <div class="summary-lbl">{html.escape(f"{row['position']} Rank")}</div>
            <div class="summary-sub">Position-specific rank from Beast; falls back to consensus-in-position when missing.</div>
          </div>
          <div class="summary-card">
            <div class="summary-val pred" style="color:{dc_color};">{pred_lbl.upper()}</div>
            <div class="summary-lbl">Prediction</div>
            <div class="summary-sub">Model view vs. current consensus</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if drafted_row is not None:
        dr_round  = int(drafted_row["Round"])
        dr_pick   = int(drafted_row["Pick"])
        dr_team   = str(drafted_row.get("NFL Team", "")).strip()
        team_meta_for_card = load_nfl_team_meta()
        dr_meta   = team_meta_for_card.get(normalize_team_name(dr_team), {})
        # Raw color for background tinting — always use a team color, dark is fine at low opacity
        dr_raw_color = dr_meta.get("color_1") or dr_meta.get("color_2") or "#3b82f6"
        # Readable color for text — tries color_1 then color_2, falls back to neutral
        dr_color  = readable_team_color(dr_meta.get("color_1"), dr_meta.get("color_2"), fallback="#3b82f6")
        dr_logo   = dr_meta.get("team_logo_espn") or dr_meta.get("logo")

        # Pick value tier
        if dr_pick <= 32:
            pick_tier = "Round 1"
        elif dr_pick <= 64:
            pick_tier = "Round 2"
        elif dr_pick <= 105:
            pick_tier = "Round 3"
        elif dr_pick <= 140:
            pick_tier = "Round 4"
        elif dr_pick <= 175:
            pick_tier = "Round 5"
        elif dr_pick <= 215:
            pick_tier = "Round 6"
        else:
            pick_tier = "Round 7 / UDFA"

        # Rise / fall vs consensus (positive = went earlier = rose)
        rise_fall_picks = consensus - dr_pick if isinstance(consensus, int) else None

        _undrafted_consensus = isinstance(consensus, int) and consensus > 257
        predicted_tier = row["draft_prediction"]

        if _undrafted_consensus:
            # Consensus had this player outside the draft — getting drafted at all is the story.
            accuracy_label = "Beat undrafted projection"
            accuracy_color = "#22c55e"
            accuracy_icon  = "↑"
            rf_label = "Projected undrafted"
            rf_color = "#94a3b8"
        elif rise_fall_picks is not None:
            # Prediction vs reality: derive actual tier from pick vs consensus rank.
            # Threshold scales with rank — top picks need tight accuracy, later picks
            # have fuzzier boards so a wider band still counts as "consensus."
            import math as _math
            _tier_thresh = _math.floor(max(1, 2 * _math.log(consensus) + 0.07 * min(consensus, 200) ** 0.9))
            if rise_fall_picks >= _tier_thresh:
                actual_tier = "riser"
            elif rise_fall_picks <= -_tier_thresh:
                actual_tier = "slide"
            else:
                actual_tier = "consensus"
            if actual_tier == predicted_tier:
                accuracy_label = "Model called it"
                accuracy_color = "#22c55e"
                accuracy_icon  = "✓"
            elif actual_tier == "riser":
                accuracy_label = "Went earlier than model predicted"
                accuracy_color = "#f59e0b"
                accuracy_icon  = "↑"
            elif actual_tier == "slide":
                accuracy_label = "Went later than model predicted"
                accuracy_color = "#ef4444"
                accuracy_icon  = "↓"
            else:
                accuracy_label = "Closer to consensus than predicted"
                accuracy_color = "#94a3b8"
                accuracy_icon  = "→"
            if rise_fall_picks > 0:
                rf_label = f"Rose {rise_fall_picks} spots vs consensus"
                rf_color = "#22c55e"
            elif rise_fall_picks < 0:
                rf_label = f"Fell {abs(rise_fall_picks)} spots vs consensus"
                rf_color = "#ef4444"
            else:
                rf_label = "Picked exactly at consensus"
                rf_color = "#94a3b8"
        else:
            accuracy_label = "—"
            accuracy_color = "#94a3b8"
            accuracy_icon  = ""
            rf_label = "—"
            rf_color = "#94a3b8"

        team_logo_html = (
            f'<img src="{html.escape(str(dr_logo))}" style="height:44px;object-fit:contain;display:block;" />'
            if dr_logo else
            f'<div style="font-size:13px;font-weight:700;color:#f1f5f9;">{html.escape(dr_team)}</div>'
        )

        dr_rgba   = hex_to_rgba(dr_raw_color, 0.18) or "rgba(59,130,246,0.18)"
        dr_border = hex_to_rgba(dr_raw_color, 0.5) or "rgba(59,130,246,0.5)"

        st.markdown(f"""
        <div style="background:{dr_rgba};border:1px solid {dr_border};border-radius:10px;
                    padding:18px 20px;margin:16px 0 4px;">
          <div style="font-size:10px;font-weight:700;color:{dr_color};text-transform:uppercase;
                      letter-spacing:1.2px;margin-bottom:12px;">DRAFTED</div>
          <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">
            <div style="display:flex;flex-direction:column;align-items:center;gap:6px;min-width:80px;">
              {team_logo_html}
              <div style="font-size:11px;color:#94a3b8;">{html.escape(dr_team)}</div>
            </div>
            <div style="display:flex;gap:16px;flex-wrap:wrap;flex:1;">
              <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:12px 18px;text-align:center;min-width:90px;">
                <div style="font-size:24px;font-weight:900;color:{dr_color};">R{dr_round}</div>
                <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Round</div>
              </div>
              <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:12px 18px;text-align:center;min-width:90px;">
                <div style="font-size:24px;font-weight:900;color:{dr_color};">#{dr_pick}</div>
                <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Overall</div>
              </div>
              <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:12px 18px;text-align:center;min-width:90px;">
                <div style="font-size:15px;font-weight:700;color:#f1f5f9;">{pick_tier}</div>
                <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Pick Value</div>
              </div>
              <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:12px 18px;text-align:center;min-width:120px;">
                <div style="font-size:15px;font-weight:700;color:{accuracy_color};">{accuracy_icon} {accuracy_label}</div>
                <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">vs Model Prediction ({pred_lbl})</div>
              </div>
              <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:12px 18px;text-align:center;min-width:120px;">
                <div style="font-size:15px;font-weight:700;color:{rf_color};">{rf_label}</div>
                <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">vs Consensus #{consensus}</div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Draft tier probs | Scout Confidence ──
    pc1, pc2 = st.columns(2)

    with pc1:
        st.markdown('<div class="chart-lbl">DRAFT TIER PROBABILITIES</div>', unsafe_allow_html=True)
        if isinstance(consensus, int) and consensus > 257:
            st.markdown('<div style="color:#64748b;font-size:13px;padding:32px 8px;text-align:center;">Model not applicable for projected undrafted players.</div>', unsafe_allow_html=True)
        else:
            dv_df = pd.DataFrame({
                "tier":  ["Slide","Consensus","Riser"],
                "prob":  [row["p_slide"], row["p_consensus"], row["p_riser"]],
                "color": [DRAFT_COLORS["slide"], DRAFT_COLORS["consensus"], DRAFT_COLORS["riser"]],
            }).sort_values("prob")
            fig = go.Figure(go.Bar(
                x=dv_df["prob"], y=dv_df["tier"], orientation="h",
                marker_color=dv_df["color"],
                text=[f"{p:.1f}%" for p in dv_df["prob"]],
                textposition="outside", textfont=dict(color="rgba(255,255,255,0.6)", size=11),
                hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
            ))
            styled(fig, height=220, margin=dict(l=16, r=48, t=8, b=8),
                   xaxis=dict(range=[0, 110], gridcolor="#0d2a52", linecolor="#1a3a6b",
                              showticklabels=False),
                   yaxis=dict(gridcolor="#0d2a52", linecolor="#1a3a6b"))
            st.plotly_chart(fig, use_container_width=True, key="pc_dv")

    with pc2:
        st.markdown('<div class="chart-lbl">SCOUT CONFIDENCE</div>', unsafe_allow_html=True)
        sc_val     = float(row.get("scout_conf", 0))
        ng_cnt     = int(row.get("ngram_count", 0))
        has_br_val = int(row.get("has_br", 0))
        sc_color   = "#22c55e" if sc_val >= 60 else "#3b82f6" if sc_val >= 30 else "#64748b"
        sc_tier    = "High" if sc_val >= 60 else "Medium" if sc_val >= 30 else "Low"
        br_str     = "BR coverage included" if has_br_val else "No BR coverage"
        conf_terms_html, conf_summary = render_confidence_terms(ng_row)
        st.markdown(f"""
        <div style="background:#0e1520;border:1px solid #1c2840;border-radius:8px;
                    padding:20px;text-align:center;min-height:200px;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;">
          <div style="font-size:48px;font-weight:900;color:{sc_color};line-height:1;">{sc_val:.0f}</div>
          <div style="font-size:11px;font-weight:700;color:{sc_color};margin-top:4px;
                      text-transform:uppercase;letter-spacing:1px;">{sc_tier} Confidence</div>
          <div style="font-size:12px;color:#94a3b8;margin-top:6px;">{ng_cnt} cross-source phrase{"s" if ng_cnt != 1 else ""}</div>
          <div style="font-size:11px;color:#4a6179;margin-top:2px;">{br_str}</div>
          <div style="font-size:11px;color:#cbd5e1;line-height:1.6;margin-top:12px;max-width:360px;">{conf_summary}</div>
          <div style="margin-top:10px;max-width:360px;">{conf_terms_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Measurables radar + grid ──
    radar_fig = radar_chart(row)
    has_meas  = any(pd.notna(row.get(c)) for c in MEAS_RAW)

    if has_meas:
        st.markdown('<div class="sec-lbl">Measurables vs Position Group Peers</div>',
                    unsafe_allow_html=True)
        mr1, mr2 = st.columns([1, 2])

        with mr1:
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True, key="radar_chart")

        with mr2:
            meas_items = []
            for raw, lbl in zip(MEAS_RAW, MEAS_LBL):
                val  = row.get(raw)
                fmtd = fmt_meas(lbl, val)
                if fmtd:
                    pct_col = f"pct_{raw}"
                    pct_v   = row.get(pct_col)
                    pct_str = f" · {pct_v:.0f}th %ile" if pd.notna(pct_v) else ""
                    meas_items.append((lbl, fmtd, pct_str))
            if meas_items:
                cards_html = '<div class="meas-grid">'
                for lbl, val, pct in meas_items:
                    cards_html += (f'<div class="meas-card">'
                                   f'<div class="meas-val">{val}</div>'
                                   f'<div class="meas-lbl">{lbl}{pct}</div></div>')
                # RAS card
                ras_val = row.get("ras")
                if pd.notna(ras_val):
                    ras_float = float(ras_val)
                    ras_color = ("#22c55e" if ras_float >= 8.0
                                 else "#3b82f6" if ras_float >= 5.0
                                 else "#ef4444")
                    ras_alltime = row.get("ras_alltime")
                    ras_sub = f"All-time: {float(ras_alltime):.2f}" if pd.notna(ras_alltime) else ""
                    cards_html += (f'<div class="meas-card" style="border-color:{ras_color}44;">'
                                   f'<div class="meas-val" style="color:{ras_color};">{ras_float:.2f}</div>'
                                   f'<div class="meas-lbl">RAS · /10{(" · " + ras_sub) if ras_sub else ""}</div></div>')
                cards_html += "</div>"
                st.markdown(cards_html, unsafe_allow_html=True)

    # ── Structural drivers (draft model) ──
    st.markdown('<div class="sec-lbl">Structural Drivers — Draft Model</div>',
                unsafe_allow_html=True)
    feat_vals  = [
        row["draft_feat_consensus"],
        row["draft_top_measurable_contrib"],
    ]
    feat_lbls  = [
        "Consensus Rank",
        f"Top Measurable ({row['draft_top_measurable']})" if pd.notna(row.get("draft_top_measurable")) else "Top Measurable",
    ]
    feat_df = pd.DataFrame({"feature": feat_lbls, "contribution": feat_vals})
    feat_df["color"] = feat_df["contribution"].apply(lambda v: "#2979FF" if v >= 0 else "#D50A0A")
    feat_df = feat_df.sort_values("contribution")
    fig = go.Figure(go.Bar(
        x=feat_df["contribution"], y=feat_df["feature"],
        orientation="h", marker_color=feat_df["color"],
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    styled(fig, height=160, margin=dict(l=16, r=24, t=8, b=8),
           xaxis=dict(gridcolor="#0d2a52", linecolor="#1a3a6b",
                      zerolinecolor="rgba(255,255,255,0.3)"),
           yaxis=dict(gridcolor="#0d2a52", linecolor="#1a3a6b"))
    st.plotly_chart(fig, use_container_width=True, key="feat_bar")

    # ── Historical Comps ──
    st.markdown('<div class="sec-lbl">Most Similarly Scouted — Past Drafts</div>',
                unsafe_allow_html=True)
    if not hist_sim_df.empty and sel in hist_sim_df.index:
        hist_sims = hist_sim_df.loc[sel].sort_values(ascending=False).head(30)
        # Use dict for safe position lookup (avoids ambiguous Series truth value)
        pos_lkp  = hist_meta["Position"].to_dict()
        same_pos = [n for n in hist_sims.index if pos_lkp.get(n) == row["position"]]
        if len(same_pos) >= 3:
            top_hist = hist_sims[hist_sims.index.isin(same_pos)].head(5)
        else:
            top_hist = hist_sims.head(5)

        cards_html = '<div class="sim-grid">'
        for hname, hscore in top_hist.items():
            if hname not in hist_meta.index:
                continue
            hrow     = hist_meta.loc[hname]
            hyr      = int(hrow["draft_year"]) if pd.notna(hrow.get("draft_year")) else "?"
            hpos     = hrow.get("Position", "")
            hcollege = hrow.get("College", "") or ""
            try:    hrnd = int(float(hrow["round"])) if pd.notna(hrow.get("round")) else "?"
            except: hrnd = "UDFA"
            try:    hpick = int(float(hrow["pick"])) if pd.notna(hrow.get("pick")) else "?"
            except: hpick = "?"
            pick_str = ("UDFA" if hrnd == "UDFA"
                        else f"Rd {hrnd}, Pick {hpick}" if hrnd != "?" else "—")
            college_line = f'<div class="sim-college">{html.escape(hcollege)}</div>' if hcollege else ""
            cards_html += f"""
            <div class="sim-card">
              <div class="sim-kicker">Historical comp</div>
              <div class="sim-name">{hname}</div>
              <div class="sim-meta">{hpos} · {hyr}</div>
              {college_line}
              <div class="sim-meta">{pick_str}</div>
              <div class="sim-score">{hscore:.2f}</div>
              <div class="sim-kicker" style="margin-top:4px;">Similarity score</div>
            </div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="sent-empty">Historical comp data not available.</div>',
                    unsafe_allow_html=True)

    # ── Similar Prospects (2026 class) ──
    st.markdown('<div class="sec-lbl">Most Similarly Scouted — 2026 Class</div>',
                unsafe_allow_html=True)
    if not sim_df.empty and sel in sim_df.index:
        all_sims = sim_df.loc[sel].drop(labels=[sel], errors="ignore").sort_values(ascending=False)
        # prefer same position group, fall back to all positions
        pos_players = df[df["position"] == sel_pos_card]["player_name"].tolist()
        sims_pos    = all_sims[all_sims.index.isin(pos_players)].head(5)
        if len(sims_pos) < 3:
            sims_pos = all_sims.head(5)

        cards_html = '<div class="sim-grid">'
        for name, score in sims_pos.items():
            sim_row = df[df["player_name"] == name]
            if len(sim_row) == 0:
                continue
            sim_row      = sim_row.iloc[0]
            sim_cons     = int(sim_row["consensus"]) if pd.notna(sim_row.get("consensus")) else "?"
            sim_pos      = sim_row.get("position", "")
            sim_tier     = sim_row.get("draft_prediction", "")
            sim_color    = DRAFT_COLORS.get(sim_tier, "#64748b")
            sim_college  = sim_row.get("college", "") or ""
            sim_hs       = sim_row.get("headshot_url", "")
            sim_logo     = sim_row.get("college_logo_url", "")
            has_sim_hs   = pd.notna(sim_hs) and str(sim_hs).strip()
            has_sim_logo = pd.notna(sim_logo) and str(sim_logo).strip()

            # Headshot square (left)
            if has_sim_hs:
                hs_html = (f'<div style="width:82px;height:82px;flex-shrink:0;border-radius:8px;'
                           f'overflow:hidden;border:1px solid #1c2840;background:#0a1020;">'
                           f'<img src="{str(sim_hs).strip()}" style="width:100%;height:100%;'
                           f'object-fit:contain;display:block;"/></div>')
            else:
                ini = initials(name)
                hs_html = (f'<div style="width:82px;height:82px;flex-shrink:0;border-radius:8px;'
                           f'background:#132036;display:flex;align-items:center;justify-content:center;'
                           f'font-size:22px;font-weight:800;color:#f1f5f9;">{ini}</div>')

            # Logo top-right
            if has_sim_logo:
                logo_html = (f'<div style="background:#fff;border-radius:8px;padding:6px;'
                             f'width:48px;height:48px;display:flex;align-items:center;'
                             f'justify-content:center;flex-shrink:0;">'
                             f'<img src="{str(sim_logo).strip()}" style="max-width:36px;max-height:36px;'
                             f'object-fit:contain;"/></div>')
            else:
                logo_html = ""

            cards_html += f"""
            <div class="sim-card">
              <div class="sim-kicker">This class</div>
              <div style="display:flex;gap:12px;align-items:flex-start;margin:10px 0 10px;">
                {hs_html}
                <div style="flex:1;min-width:0;display:flex;flex-direction:column;gap:2px;position:relative;">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:6px;">
                    <div class="sim-name" style="margin-bottom:4px;">{name}</div>
                    {logo_html}
                  </div>
                  <div class="sim-meta">{sim_pos} · #{sim_cons}</div>
                  <div class="sim-meta" style="color:{sim_color};font-weight:700;">{DRAFT_LABELS.get(sim_tier, sim_tier.title() if isinstance(sim_tier, str) else "—")}</div>
                </div>
              </div>
              <div class="sim-score">{score:.2f}</div>
              <div class="sim-kicker" style="margin-top:4px;">Similarity score</div>
            </div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="sent-empty">Similarity data not available.</div>',
                    unsafe_allow_html=True)

    # ── Cross-source phrases (has native widget — use st.container) ──
    st.markdown('<div class="sec-lbl">Cross-Source Scouting Terms</div>', unsafe_allow_html=True)
    with st.container(border=False):
        if len(ng_row) > 0:
            phrase_query = st.text_input("Search repeated scouting terms",
                                         key=f"phrase_search_{sel}",
                                         placeholder="Type a phrase like short arms or burst")
            ng_sorted = ng_row.sort_values(["sources_present", "total_count"], ascending=False).copy()
            if phrase_query:
                ng_sorted = ng_sorted[ng_sorted["ngram"].str.contains(phrase_query, case=False, na=False)]
            rows_html = ""
            for _, nr in ng_sorted.iterrows():
                chips = ""
                if nr["beast_n"] > 0:
                    chips += f'<span class="src-chip s-beast">BEAST ×{int(nr["beast_n"])}</span>'
                if nr["pff_n"] > 0:
                    chips += f'<span class="src-chip s-pff">PFF ×{int(nr["pff_n"])}</span>'
                if nr["br_n"] > 0:
                    chips += f'<span class="src-chip s-br">BR ×{int(nr["br_n"])}</span>'
                rows_html += (f'<tr><td>{nr["ngram"]}</td><td>{chips}</td>'
                              f'<td style="color:#94a3b8;font-size:12px">{int(nr["sources_present"])} sources</td></tr>')
            if rows_html:
                st.markdown(f'<div class="table-wrap"><table class="ng-table">'
                            f'<thead><tr><th>Term</th><th>Sources</th><th>Agreement</th></tr></thead>'
                            f'<tbody>{rows_html}</tbody></table></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sent-empty">No repeated scouting terms matched that search.</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="sent-empty">No cross-source phrases found for this player.</div>',
                        unsafe_allow_html=True)

    # ── Scouting sentences (pure HTML — render as single block) ──
    dv_pos = parse_sentences(row.get("draft_key_sentences", ""))
    dv_neg = parse_sentences(row.get("draft_concerns", ""))
    dv_pos_chips = render_word_chips(row.get("draft_pos_words", ""), positive=True)
    dv_neg_chips = render_word_chips(row.get("draft_neg_words", ""), positive=False)
    dv_summary = render_inline_summary(row.get("draft_pos_words", ""), row.get("draft_neg_words", ""))
    st.markdown('<div class="sec-lbl">Key Scouting Language</div>', unsafe_allow_html=True)
    _sl_inner = ""
    if dv_summary:
        _sl_inner += f'<div style="margin-bottom:12px;line-height:1.8;font-size:13px;">{dv_summary}</div>'
    if dv_pos_chips:
        _sl_inner += '<div class="chart-lbl" style="margin:0 0 6px;">Supports prediction</div>' + dv_pos_chips
    if dv_neg_chips:
        _sl_inner += '<div class="chart-lbl" style="margin:12px 0 6px;">Pushes against prediction</div>' + dv_neg_chips
    if dv_pos_chips or dv_neg_chips:
        _sl_inner += '<div style="height:12px"></div>'
    _sl_inner += render_sentences(dv_pos, is_neg=False) if dv_pos else '<div class="sent-empty">No high-signal positive sentences found.</div>'
    if dv_neg:
        _sl_inner += '<div class="sec-lbl" style="margin-top:20px">Concerns</div>' + render_sentences(dv_neg, is_neg=True)
    st.markdown(f'<div class="section-card">{_sl_inner}</div>', unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — CONTRACT OUTLOOK
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:320px;text-align:center;gap:16px;">

      <div style="font-size:22px;font-weight:800;color:#f1f5f9;letter-spacing:-0.3px;">Coming Soon</div>
      <div style="font-size:14px;color:#64748b;max-width:420px;line-height:1.7;">
        Once the 2026 class has had time to develop, this tab will show
        projected second contract value, confidence ranges, and the likelihood
        each player earns a real deal.
      </div>
    </div>
    """, unsafe_allow_html=True)
