"""
2026 NFL Draft Intelligence — Streamlit app
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import html
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

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
    "reach":     "#22c55e",
    "consensus": "#3b82f6",
    "slide":     "#D50A0A",
}
DRAFT_LABELS = {
    "reach":     "Riser",
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

/* ── Nav ── */
.top-nav {
    background: #0d1626;
    border-bottom: 1px solid #1c2840;
    padding: 0 40px;
    height: 56px;
    display: flex;
    align-items: center;
    gap: 32px;
}
.nav-wordmark { font-size: 15px; font-weight: 800; color: #ffffff; letter-spacing: 0.5px; }
.nav-wordmark span { color: #D50A0A; }
.nav-divider { width: 1px; height: 20px; background: #1c2840; }
.nav-sub { font-size: 12px; color: #4a6179; font-weight: 400; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a1020;
    border-bottom: 1px solid #1c2840;
    padding: 0 32px;
    gap: 0;
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
div[data-testid="stRadio"] > div {
    flex-wrap: wrap !important;
    gap: 4px 20px !important;
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
    extra = [
        "Player Name", "beast_grade", "br_pro_comparison", "has_br", "has_pff",
        "text_source", "college_logo_url",
    ] + MEAS_RAW
    inf_sub = inf[[c for c in extra if c in inf.columns]].rename(
        columns={"Player Name": "player_name", "text_source": "inf_text_source"})
    df = exp.merge(inf_sub, on="player_name", how="left")

    # Merge headshots
    if headshots_path.exists():
        hs = pd.read_csv(headshots_path)[["player_name", "headshot_url"]].copy()
        hs["player_name"] = hs["player_name"].str.strip()
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
    df["has_br"] = df["has_br"].fillna(0)
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
    hist_meta = hist.set_index("Player Name")[hist_cols].copy()

    # #1 overall pick can't go higher — zero out riser probability and renormalize
    mask = df["consensus"] == 1
    df.loc[mask, "p_reach"] = 0.0
    total = df.loc[mask, ["p_slide","p_consensus"]].sum(axis=1)
    df.loc[mask, "p_slide"]     = df.loc[mask, "p_slide"]     / total * 100
    df.loc[mask, "p_consensus"] = df.loc[mask, "p_consensus"] / total * 100

    df["round_est"] = ((df["consensus"] - 1) // 32 + 1).clip(1, 7)
    return df, ng, sim_df, hist_sim_df, hist_meta

df, ng, sim_df, hist_sim_df, hist_meta = load_data()


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
                result.append((word.strip(), float(score)))
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


# ── Nav bar ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="top-nav">
  <div class="nav-wordmark">DRAFT<span>INTEL</span> &nbsp;2026</div>
  <div class="nav-divider"></div>
  <div class="nav-sub">NLP + Machine Learning &nbsp;&middot;&nbsp; {len(df)} Prospects &nbsp;&middot;&nbsp; Beast / PFF / Bleacher Report</div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab4, tab1, tab2, tab3 = st.tabs([
    "Player Card",
    "Class Overview",
    "Draft Board",
    "Scouting Language",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CLASS OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:

    n_total   = len(df)
    n_slides  = (df["draft_prediction"] == "slide").sum()
    n_reaches = (df["draft_prediction"] == "reach").sum()
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

    # ── Donuts ──
    c1, c2 = st.columns(2)

    with c1:
        dc = df["draft_prediction"].value_counts().reset_index()
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
    pos_grp = df.groupby(["position","draft_prediction"]).size().reset_index(name="n")
    pos_order = (df.groupby("position")["p_reach"].mean()
                   .sort_values(ascending=False).index.tolist())
    fig = go.Figure()
    for tier in ["reach","consensus","slide"]:
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
    rf_df = df[df["rise_fall"].notna()].copy()
    rf_df["rise_fall_fmt"] = rf_df["rise_fall"].round(1)
    fig = px.scatter(
        rf_df,
        x="consensus", y="rise_fall",
        color="draft_prediction",
        color_discrete_map=DRAFT_COLORS,
        size="scout_conf",
        size_max=16,
        hover_name="player_name",
        hover_data={"consensus": True, "rise_fall": ":.0f",
                    "position": True, "draft_prediction": True,
                    "beast_rank": True, "consensus_pos_rank": ":.0f", "scout_conf": ":.0f"},
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
    for tier in ["reach","consensus","slide"]:
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

    fdf = df.copy()
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
                    "p_reach": ":.1f", "position": True,
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
    for tier in ["reach","consensus","slide"]:
        fig.for_each_trace(lambda t, _tier=tier: t.update(name=DRAFT_LABELS[_tier])
                           if t.name == _tier else None)
    styled(fig, height=400)
    st.plotly_chart(fig, use_container_width=True, key="board_scatter")

    bc1, bc2 = st.columns(2)

    with bc1:
        # Top risers: consensus >> beast_rank (consensus undervalues)
        risers = (fdf[fdf["rise_fall"].notna()]
                  .nlargest(15, "rise_fall")
                  [["player_name","position","consensus","beast_rank","consensus_pos_rank","rise_fall","draft_prediction"]])
        fig = go.Figure(go.Bar(
            x=risers["rise_fall"],
            y=risers["player_name"],
            orientation="h",
            marker_color="#22c55e",
            text=[f"#{int(b)} Beast / #{int(c)} consensus pos"
                  for b, c in zip(risers["beast_rank"], risers["consensus_pos_rank"])],
            textposition="outside",
            textfont=dict(color="rgba(255,255,255,0.45)", size=9),
            hovertemplate="<b>%{y}</b><br>Rise score: %{x:.0f}<extra></extra>",
        ))
        styled(fig, height=400, title_text="Top Risers (Beast > Consensus)",
               yaxis=dict(autorange="reversed", gridcolor="#0d2a52", linecolor="#1a3a6b"),
               xaxis=dict(gridcolor="#0d2a52", linecolor="#1a3a6b"),
               margin=dict(l=16, r=100, t=44, b=16))
        st.plotly_chart(fig, use_container_width=True, key="risers_bar")

    with bc2:
        # Top sliders: beast_rank >> consensus (consensus overvalues)
        sliders = (fdf[fdf["rise_fall"].notna()]
                   .nsmallest(15, "rise_fall")
                   [["player_name","position","consensus","beast_rank","consensus_pos_rank","rise_fall","draft_prediction"]])
        fig = go.Figure(go.Bar(
            x=sliders["rise_fall"].abs(),
            y=sliders["player_name"],
            orientation="h",
            marker_color=NFL_RED,
            text=[f"#{int(c)} consensus pos / #{int(b)} Beast"
                  for b, c in zip(sliders["beast_rank"], sliders["consensus_pos_rank"])],
            textposition="outside",
            textfont=dict(color="rgba(255,255,255,0.45)", size=9),
            hovertemplate="<b>%{y}</b><br>Fall score: %{x:.0f}<extra></extra>",
        ))
        styled(fig, height=400, title_text="Top Sliders (Consensus > Beast)",
               yaxis=dict(autorange="reversed", gridcolor="#0d2a52", linecolor="#1a3a6b"),
               xaxis=dict(gridcolor="#0d2a52", linecolor="#1a3a6b"),
               margin=dict(l=16, r=100, t=44, b=16))
        st.plotly_chart(fig, use_container_width=True, key="sliders_bar")

    # ── Full table ──
    st.markdown('<div class="sec-lbl">All Prospects</div>', unsafe_allow_html=True)
    show_cols = ["player_name","position","college","consensus","beast_rank","rise_fall",
                 "draft_prediction","p_slide","p_consensus","p_reach","ngram_count","scout_conf"]
    rename_map = {
        "player_name":"Player","position":"Pos","college":"College",
        "consensus":"Consensus","beast_rank":"Beast Rank","rise_fall":"Rise/Fall",
        "draft_prediction":"Draft Tier","p_slide":"P(Slide) %",
        "p_consensus":"P(Consensus) %","p_reach":"P(Reach) %",
        "ngram_count":"Cross-Source Phrases","scout_conf":"Scout Conf",
    }
    tbl = fdf[show_cols].rename(columns=rename_map).sort_values("Consensus")
    st.dataframe(
        tbl,
        use_container_width=True,
        height=380,
        column_config={
            "P(Slide) %":     st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "P(Reach) %":     st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
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
        ("reach",     "#22c55e", wc1),
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
    POS_ORDER_DISPLAY = ["QB","RB","WR","TE","OT","IOL","DL","EDGE","LB","CB","S"]
    all_pos   = df["position"].dropna().unique()
    positions = [p for p in POS_ORDER_DISPLAY if p in all_pos] + \
                [p for p in sorted(all_pos) if p not in POS_ORDER_DISPLAY]
    st.markdown('<div class="sec-lbl" style="margin-top:0">Filter by position</div>',
                unsafe_allow_html=True)
    sel_pos_card = st.radio("", positions, index=0, key="pos_pills",
                            label_visibility="collapsed", horizontal=True)

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
    pred_lbl  = DRAFT_LABELS.get(row["draft_prediction"], row["draft_prediction"].title())
    pos_label = f"{row['position']} Rank"
    college_logo_url = row.get("college_logo_url")
    headshot_url_val = row.get("headshot_url")

    badges = f'<span class="badge b-pos">{row["position"]}</span>'
    if college:
        badges += f'<span class="badge b-info">{college}</span>'
    badges += f'<span class="badge b-info">Consensus #{consensus}</span>'
    if beast_rnk:
        badges += f'<span class="badge b-src">Beast #{beast_rnk}</span>'
    if bg_grade:
        badges += f'<span class="badge b-src">Beast {bg_grade}</span>'
    if pro_comp:
        badges += f'<span class="badge b-info">Comp: {pro_comp}</span>'

    hero1, hero2 = st.columns([1, 3])
    with hero1:
        show_player_logo_panel(sel, college, college_logo_url=college_logo_url,
                               headshot_url=headshot_url_val)

    with hero2:
        _logo_html = college_logo_block(college_logo_url, college, size=60)
        st.markdown(f"""
        <div class="p-header" style="display:flex;align-items:flex-start;justify-content:space-between;gap:20px;">
          <div style="flex:1;min-width:0;">
            <p class="p-name">{sel}</p>
            {badges}
            <div style="margin-top:14px;">
              <div class="chart-lbl">DRAFT TIER</div>
              <span class="pred-pill" style="background:{dc_color}22;color:{dc_color};border:1px solid {dc_color}55">
                {pred_lbl}
              </span>
            </div>
          </div>
          {_logo_html}
        </div>
        """, unsafe_allow_html=True)

        pos_rank_str = f"#{pos_rank}" if pos_rank is not None else "N/A"
        consensus_str = f"#{consensus}" if consensus != "?" else "N/A"
        summary_sub = "Position-specific rank from Beast; falls back to consensus-in-position when missing."
        st.markdown(f"""
        <div class="summary-grid">
          <div class="summary-card">
            <div class="summary-val">{consensus_str}</div>
            <div class="summary-lbl">Consensus Rank</div>
            <div class="summary-sub">Overall board placement</div>
          </div>
          <div class="summary-card">
            <div class="summary-val">{pos_rank_str}</div>
            <div class="summary-lbl">{html.escape(pos_label)}</div>
            <div class="summary-sub">{summary_sub}</div>
          </div>
          <div class="summary-card">
            <div class="summary-val pred" style="color:{dc_color};">{pred_lbl.upper()}</div>
            <div class="summary-lbl">Prediction</div>
            <div class="summary-sub">Model view vs. current consensus</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Draft tier probs | Scout Confidence ──
    pc1, pc2 = st.columns(2)

    with pc1:
        st.markdown('<div class="chart-lbl">DRAFT TIER PROBABILITIES</div>', unsafe_allow_html=True)
        dv_df = pd.DataFrame({
            "tier":  ["Slide","Consensus","Riser"],
            "prob":  [row["p_slide"], row["p_consensus"], row["p_reach"]],
            "color": [DRAFT_COLORS["slide"], DRAFT_COLORS["consensus"], DRAFT_COLORS["reach"]],
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
    SC_TIER_COLORS = {
        "cornerstone":   "#D50A0A",
        "53_man":        "#3b82f6",
        "roster_bubble": "#64748b",
        "out_of_league": "#334155",
    }
    SC_TIER_LABELS = {
        "cornerstone":   "Cornerstone",
        "53_man":        "53-Man",
        "roster_bubble": "Roster Bubble",
        "out_of_league": "Out of League",
    }
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
            htier    = hrow.get("sc_contract_tier", "")
            htier    = htier if pd.notna(htier) else ""
            htcolor  = SC_TIER_COLORS.get(htier, "#334155")
            htlabel  = SC_TIER_LABELS.get(htier, "2nd contract pending" if hyr >= 2023 else "—")
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
              <div class="sim-meta" style="color:{htcolor};font-weight:600;">{htlabel}</div>
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
                  <div class="sim-meta" style="color:{sim_color};font-weight:700;">{DRAFT_LABELS.get(sim_tier, sim_tier.title())}</div>
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

    # ── Word contribution chips (pure HTML — render as single block) ──
    dv_pos_chips = render_word_chips(row.get("draft_pos_words", ""), positive=True)
    dv_neg_chips = render_word_chips(row.get("draft_neg_words", ""), positive=False)
    st.markdown('<div class="sec-lbl">Draft Model Word Signals</div>', unsafe_allow_html=True)
    if dv_pos_chips or dv_neg_chips:
        _wc_inner = ""
        if dv_pos_chips:
            _wc_inner += '<div class="chart-lbl" style="margin-bottom:6px;">Supports prediction</div>' + dv_pos_chips
        if dv_neg_chips:
            _wc_inner += '<div class="chart-lbl" style="margin:12px 0 6px;">Pushes against prediction</div>' + dv_neg_chips
        st.markdown(f'<div class="section-card">{_wc_inner}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-card"><div class="sent-empty">No significant word signals found.</div></div>',
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
    dv_summary = render_inline_summary(row.get("draft_pos_words", ""), row.get("draft_neg_words", ""))
    st.markdown('<div class="sec-lbl">Key Scouting Language</div>', unsafe_allow_html=True)
    _sl_inner = ""
    if dv_summary:
        _sl_inner += f'<div style="margin-bottom:12px;line-height:1.8;font-size:13px;">{dv_summary}</div>'
    _sl_inner += render_sentences(dv_pos, is_neg=False) if dv_pos else '<div class="sent-empty">No high-signal positive sentences found.</div>'
    if dv_neg:
        _sl_inner += '<div class="sec-lbl" style="margin-top:20px">Concerns</div>' + render_sentences(dv_neg, is_neg=True)
    st.markdown(f'<div class="section-card">{_sl_inner}</div>', unsafe_allow_html=True)
