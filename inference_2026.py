"""
Run inference on 2026 draft prospects using saved models.

Outputs ranked tables (and inference_2026.csv):
  1. sc_tier: P(cornerstone) — projected NFL contract tier from scouting language
              classes: out_of_league / roster_bubble / 53_man / cornerstone
  2. draft_value: P(slide/consensus/reach) — probability player is drafted above/below consensus
  3. apy_pct: predicted APY percentile for players who get a real contract
              1.0 = highest-paid player in NFL on signing date, 0.0 = lowest
              expected_apy_pct = P(real_contract) × pred_apy_pct
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer

from features import (
    Z_COLS, unified_text, text_source_flag, has_br_flag,
    consensus_feature, beast_rank_feature, text_length_feature,
    round_feature, position_feature,
)

warnings.filterwarnings("ignore")

BASE = Path("/Users/varunramanathan/Downloads/sentiment-analysis")


# ── load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(BASE / "data/processed/all_prospects.csv")

p26 = df[df["draft_year"] == 2026].copy().reset_index(drop=True)
has_text = p26["beast_summary"].notna() | p26["pff_overview"].notna()
p26 = p26[has_text].reset_index(drop=True)
print(f"2026 prospects with scouting text: {len(p26)}")
print(f"  Beast: {p26['beast_summary'].notna().sum()} | PFF-only: {(p26['beast_summary'].isna() & p26['pff_overview'].notna()).sum()}")

# ── load models ───────────────────────────────────────────────────────────────

print("Loading models...")
sc_bundle  = joblib.load(BASE / "models/model_sc_tier.pkl")
dv_bundle  = joblib.load(BASE / "models/model_draft_value.pkl")
apy_bundle = joblib.load(BASE / "models/model_apy_pct.pkl")

st = SentenceTransformer(sc_bundle["st_model"])

# ── encode features ───────────────────────────────────────────────────────────

print("Encoding text features...")
texts   = unified_text(p26)
X_src   = text_source_flag(p26)
X_br    = has_br_flag(p26)
X_emb   = st.encode(texts.tolist(), batch_size=64,
                    show_progress_bar=False, normalize_embeddings=True)
X_meas  = p26[sc_bundle["z_cols"]].fillna(0.0).values

# Compute consensus z-score using only draftable players (consensus <= 257).
# Training data only contained drafted players, so its mu/sigma matched this range.
# Using the full 2026 pool (mean ~357) would produce z-scores far outside training range.
_p26_cons = p26.copy()
_draftable_mask = _p26_cons["consensus"].apply(lambda x: pd.notna(x) and float(x) <= 257)
_p26_cons.loc[~_draftable_mask, "consensus"] = np.nan  # undraftable → 0-imputed (mean)
X_cons     = consensus_feature(_p26_cons)
X_pos      = position_feature(p26)
X_rank     = beast_rank_feature(p26)
X_tlen     = text_length_feature(texts)
X_rnd      = round_feature(p26)   # all 2026 rows imputed from consensus (round col is NaN)
X_extra_sc = np.hstack([X_cons, X_br, X_pos, X_rank, X_tlen, X_rnd])  # sc: full extras
X_extra_dv = np.hstack([X_cons, X_br])                                  # dv: unchanged

X_tfidf_sc = sc_bundle["svd"].transform(sc_bundle["tfidf"].transform(texts))
X_tfidf_dv = dv_bundle["svd"].transform(dv_bundle["tfidf"].transform(texts))

X_sc = np.hstack([X_tfidf_sc, X_emb, X_meas, X_src,                    X_extra_sc])
X_dv = np.hstack([X_tfidf_dv, X_emb, X_meas, np.zeros((len(p26), 1)), X_extra_dv])

# ── predict ───────────────────────────────────────────────────────────────────

print("Predicting...")
le_sc      = sc_bundle["le"]
sc_classes = list(le_sc.classes_)   # ['53_man', 'cornerstone', 'out_of_league', 'roster_bubble']
if sc_bundle.get("ensemble") and sc_bundle.get("clf2") is not None:
    sc_proba = (sc_bundle["clf"].predict_proba(X_sc) + 2 * sc_bundle["clf2"].predict_proba(X_sc)) / 3
else:
    sc_proba = sc_bundle["clf"].predict_proba(X_sc)
sc_pred    = le_sc.inverse_transform(sc_proba.argmax(axis=1))

le_dv      = dv_bundle["le"]
dv_classes = list(le_dv.classes_)   # ['consensus', 'reach', 'slide']
dv_proba   = dv_bundle["clf"].predict_proba(X_dv)
dv_pred    = le_dv.inverse_transform(dv_bundle["clf"].predict(X_dv))

# APY percentile — same feature matrix as sc (identical extra features)
X_tfidf_ap  = apy_bundle["svd"].transform(apy_bundle["tfidf"].transform(texts))
X_ap        = np.hstack([X_tfidf_ap, X_emb, X_meas, X_src, X_extra_sc])
if apy_bundle.get("ensemble") and apy_bundle.get("clf2") is not None:
    apy_preds = ((apy_bundle["clf"].predict(X_ap) + 2 * apy_bundle["clf2"].predict(X_ap)) / 3).clip(0, 1)
else:
    apy_preds = apy_bundle["clf"].predict(X_ap).clip(0, 1)
apy_resid   = apy_bundle["resid_std"]

print(f"sc_tier classes: {sc_classes}")
print(f"draft_tier classes: {dv_classes}")

# ── build results ─────────────────────────────────────────────────────────────

MEAS_COLS = ["ht_in", "wt_lbs", "arm_in", "hand_in", "wing_in",
             "dash40", "vj_in", "bj_in", "shuttle", "cone3", "bench"]
TEXT_COLS = ["beast_summary", "beast_strengths", "beast_weaknesses",
             "pff_overview", "pff_pros", "pff_cons", "pff_bottom_line",
             "br_positives", "br_negatives"]

results = p26[["Player Name", "Position", "College", "consensus",
               "beast_grade", "br_article_grade", "br_pro_comparison"]].copy()
for col in MEAS_COLS + TEXT_COLS:
    if col in p26.columns:
        results[col] = p26[col]

idx_53man       = sc_classes.index("53_man")
idx_roster_bubble  = sc_classes.index("roster_bubble")
idx_cornerstone    = sc_classes.index("cornerstone")
idx_out_of_league  = sc_classes.index("out_of_league")

idx_slide     = dv_classes.index("slide")
idx_consensus = dv_classes.index("consensus")
idx_reach     = dv_classes.index("reach")

results["p_cornerstone"]       = (sc_proba[:, idx_cornerstone] * 100).round(1)
results["p_53_man"]            = (sc_proba[:, idx_53man]       * 100).round(1)
results["p_roster_bubble"]     = (sc_proba[:, idx_roster_bubble]  * 100).round(1)
results["p_out_of_league"]     = (sc_proba[:, idx_out_of_league]  * 100).round(1)
results["sc_prediction"]       = sc_pred
results["p_slide"]             = (dv_proba[:, idx_slide]       * 100).round(1)
results["p_consensus"]         = (dv_proba[:, idx_consensus]   * 100).round(1)
results["p_reach"]             = (dv_proba[:, idx_reach]       * 100).round(1)
results["draft_prediction"]    = dv_pred
results["text_source"]         = ["pff" if f else "beast" for f in X_src[:, 0]]
results["has_br"]  = p26["br_positives"].notna()
results["has_pff"] = p26["pff_overview"].notna()

# APY percentile columns
p_real_contract              = (sc_proba[:, idx_53man] + sc_proba[:, idx_cornerstone])
results["pred_apy_pct"]      = apy_preds.round(3)
results["pred_apy_pct_lo"]   = (apy_preds - apy_resid).clip(0, 1).round(3)
results["pred_apy_pct_hi"]   = (apy_preds + apy_resid).clip(0, 1).round(3)
results["p_real_contract"]   = (p_real_contract * 100).round(1)
results["expected_apy_pct"]  = (p_real_contract * apy_preds).round(3)

results.to_csv(BASE / "data/processed/inference_2026.csv", index=False)
print(f"Saved → inference_2026.csv\n")

# ── display ───────────────────────────────────────────────────────────────────

pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 22)

def show(title, ranked, cols):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print("="*70)
    print(ranked[cols].to_string(index=False))


# Top 20 by P(cornerstone)
top_cornerstone = results.nlargest(20, "p_cornerstone")
show("TOP 20 — Highest P(cornerstone)  [elite second contract projection]",
     top_cornerstone,
     ["Player Name","Position","consensus","p_cornerstone","p_53_man","p_roster_bubble","p_out_of_league","text_source"])

# Top 20 by P(slide)
top_slide = results.nlargest(20, "p_slide")
show("TOP 20 — Highest P(slide)  [likely to fall below consensus]",
     top_slide,
     ["Player Name","Position","consensus","p_slide","p_consensus","p_reach","p_cornerstone"])

# Potential steals: outside top 100, high P(cornerstone)
steals = results[results["consensus"] > 100].nlargest(15, "p_cornerstone")
show("POTENTIAL STEALS — Outside top 100 consensus, high P(cornerstone)",
     steals,
     ["Player Name","Position","consensus","p_cornerstone","p_53_man","p_slide"])

# Best per position by P(cornerstone)
print(f"\n{'='*70}")
print(f"  BEST PROSPECT PER POSITION — by P(cornerstone)")
print("="*70)
for pos in sorted(results["Position"].unique()):
    sub = results[results["Position"] == pos].nlargest(1, "p_cornerstone")
    if len(sub):
        r = sub.iloc[0]
        print(f"  {pos:<6} {r['Player Name']:<25} consensus={int(r['consensus']) if pd.notna(r['consensus']) else '?':>4}  "
              f"P(cornerstone)={r['p_cornerstone']:>5.1f}%  P(slide)={r['p_slide']:>5.1f}%")

# Top 20 by expected APY percentile (P(real_contract) × pred_apy_pct)
top_apy = results.nlargest(20, "expected_apy_pct")
show("TOP 20 — Highest Expected APY Percentile  [P(real contract) × predicted standing]",
     top_apy,
     ["Player Name","Position","consensus","p_real_contract","pred_apy_pct","pred_apy_pct_lo","pred_apy_pct_hi","expected_apy_pct"])

print(f"\nNote: pred_apy_pct is predicted rank among all active NFL players on signing date")
print(f"      (1.0 = highest-paid, 0.0 = lowest-paid). Interval = ±{apy_resid:.3f} (±1.25×CV MAE).")
print(f"\nFull results → inference_2026.csv")
