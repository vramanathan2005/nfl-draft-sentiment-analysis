"""
Run inference on 2026 draft prospects using saved models.

Outputs two ranked tables (and inference_2026.csv):
  1. sc_binary: P(made_it) — NFL success trajectory from scouting language
  2. draft_value: P(fell) — probability player gets drafted above consensus (team reaches)
"""

import re
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

BASE = Path("/Users/varunramanathan/Downloads/sentiment-analysis")

def beast_text(df):
    cols = ["beast_summary", "beast_strengths", "beast_weaknesses"]
    return df[cols].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()

def pff_text(df):
    cols = ["pff_overview", "pff_pros", "pff_cons", "pff_bottom_line", "pff_extra"]
    return df[cols].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()

def unified_text(df):
    bt = beast_text(df)
    pt = pff_text(df)
    return bt.where(bt.str.len() > 0, pt)

def text_source_flag(df):
    bt = beast_text(df)
    return (bt.str.len() == 0).astype(float).values.reshape(-1, 1)

def beast_grade_feature(df):
    def parse(val):
        if pd.isna(val): return 0.0
        s = str(val).lower()
        if any(x in s for x in ["undrafted", "priority free", "udfa", "free agent"]): return 8.0
        m = re.search(r"(\d+)(?:st|nd|rd|th)", s)
        return float(m.group(1)) if m else 0.0
    return df["beast_grade"].map(parse).values.reshape(-1, 1)


# ── load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(BASE / "all_prospects.csv")

p26 = df[df["draft_year"] == 2026].copy().reset_index(drop=True)
has_text = p26["beast_summary"].notna() | p26["pff_overview"].notna()
p26 = p26[has_text].reset_index(drop=True)
print(f"2026 prospects with scouting text: {len(p26)}")
print(f"  Beast: {p26['beast_summary'].notna().sum()} | PFF-only: {(p26['beast_summary'].isna() & p26['pff_overview'].notna()).sum()}")

# ── load models ───────────────────────────────────────────────────────────────

print("Loading models...")
sc_bundle  = joblib.load(BASE / "model_sc_tier.pkl")
dv_bundle  = joblib.load(BASE / "model_draft_value.pkl")

st = SentenceTransformer(sc_bundle["st_model"])

# ── encode features ───────────────────────────────────────────────────────────

print("Encoding text features...")
texts   = unified_text(p26)
X_src   = text_source_flag(p26)
X_grade = beast_grade_feature(p26)
X_emb   = st.encode(texts.tolist(), batch_size=64,
                    show_progress_bar=False, normalize_embeddings=True)
X_meas  = p26[sc_bundle["z_cols"]].fillna(0.0).values

X_tfidf_sc = sc_bundle["svd"].transform(sc_bundle["tfidf"].transform(texts))

texts_dv   = beast_text(p26)
X_emb_dv   = st.encode(texts_dv.tolist(), batch_size=64,
                        show_progress_bar=False, normalize_embeddings=True)
X_tfidf_dv = dv_bundle["svd"].transform(dv_bundle["tfidf"].transform(texts_dv))

dv_grade = X_grade if dv_bundle.get("use_grade") else np.zeros((len(p26), 0))

X_sc = np.hstack([X_tfidf_sc, X_emb, X_meas, X_src, X_grade])
X_dv = np.hstack([X_tfidf_dv, X_emb_dv, X_meas, np.zeros((len(p26), 1)), dv_grade])

# ── predict ───────────────────────────────────────────────────────────────────

print("Predicting...")
le = sc_bundle["le"]
is_binary = sc_bundle.get("binary", True)

sc_proba  = sc_bundle["clf"].predict_proba(X_sc)
sc_pred   = le.inverse_transform(sc_bundle["clf"].predict(X_sc))

dv_proba   = dv_bundle["clf"].predict_proba(X_dv)
le_dv      = dv_bundle["le"]
dv_classes = list(le_dv.classes_)  # ['around', 'fell', 'rose']
dv_pred    = le_dv.inverse_transform(dv_bundle["clf"].predict(X_dv))

class_order = list(le.classes_)
print(f"sc_tier classes: {class_order}")
print(f"draft_tier classes: {dv_classes}")

results = p26[["Player Name", "Position", "College", "consensus",
               "beast_grade", "br_article_grade", "br_pro_comparison"]].copy()

if is_binary:
    idx_made  = class_order.index("made_it")
    idx_didnt = class_order.index("didnt")
    results["p_made_it"] = (sc_proba[:, idx_made]  * 100).round(1)
    results["p_didnt"]   = (sc_proba[:, idx_didnt] * 100).round(1)
    results["p_made_it_rank"] = results["p_made_it"]  # unified ranking col
else:
    idx_real = class_order.index("real_contract")
    idx_prac = class_order.index("practice_only")
    idx_out  = class_order.index("out_of_league")
    results["p_real_contract"] = (sc_proba[:, idx_real] * 100).round(1)
    results["p_practice_only"] = (sc_proba[:, idx_prac] * 100).round(1)
    results["p_out_of_league"] = (sc_proba[:, idx_out]  * 100).round(1)
    results["p_made_it_rank"]  = results["p_real_contract"]  # unified ranking col

idx_higher = dv_classes.index("mocked_higher")
idx_near   = dv_classes.index("mocked_near_consensus")
idx_lower  = dv_classes.index("mocked_lower")

results["sc_prediction"]          = sc_pred
results["p_mocked_higher"]        = (dv_proba[:, idx_higher] * 100).round(1)
results["p_mocked_near_consensus"]= (dv_proba[:, idx_near]   * 100).round(1)
results["p_mocked_lower"]         = (dv_proba[:, idx_lower]  * 100).round(1)
results["draft_prediction"]       = dv_pred
results["text_source"]      = ["pff" if f else "beast" for f in X_src[:, 0]]
results["has_br"]  = p26["br_positives"].notna()
results["has_pff"] = p26["pff_overview"].notna()

results.drop(columns=["p_made_it_rank"], errors="ignore").to_csv(BASE / "inference_2026.csv", index=False)
print(f"Saved → inference_2026.csv\n")

# ── display ───────────────────────────────────────────────────────────────────

pd.set_option("display.width", 130)
pd.set_option("display.max_colwidth", 22)

def show(title, ranked, cols):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print("="*70)
    print(ranked[cols].to_string(index=False))


# Top 20 by P(real_contract / made_it)
top_made = results.nlargest(20, "p_made_it_rank")
if is_binary:
    sc_cols = ["Player Name","Position","consensus","p_made_it","p_didnt","text_source"]
    label = "P(made_it)"
else:
    sc_cols = ["Player Name","Position","consensus","p_real_contract","p_practice_only","p_out_of_league","text_source"]
    label = "P(real_contract)"
show(f"TOP 20 — Highest {label}  [predicted NFL contributors]", top_made, sc_cols)

# Top 20 by P(mocked_higher) — consensus overrated them vs where teams drafted
top_mocked_higher = results.nlargest(20, "p_mocked_higher")
rc_col = "p_real_contract" if not is_binary else "p_made_it"
show("TOP 20 — Highest P(mocked_higher)  [consensus ranked them above where teams drafted]",
     top_mocked_higher,
     ["Player Name","Position","consensus","p_mocked_higher","p_mocked_near_consensus","p_mocked_lower", rc_col])

# Potential steals: outside top 100 consensus but high P(real_contract)
steals = results[results["consensus"] > 100].nlargest(15, "p_made_it_rank")
show("POTENTIAL STEALS — Outside top 100 consensus, high P(real_contract)",
     steals,
     ["Player Name","Position","consensus", rc_col, "p_mocked_higher"])

# Best per position
rc_label = "P(real_contract)" if not is_binary else "P(made_it)"
print(f"\n{'='*70}")
print(f"  BEST PROSPECT PER POSITION — by {rc_label}")
print("="*70)
for pos in sorted(results["Position"].unique()):
    sub = results[results["Position"] == pos].nlargest(1, "p_made_it_rank")
    if len(sub):
        r = sub.iloc[0]
        print(f"  {pos:<6} {r['Player Name']:<25} consensus={int(r['consensus']) if pd.notna(r['consensus']) else '?':>4}  "
              f"{rc_label}={r[rc_col]:>5.1f}%  P(mocked_higher)={r['p_mocked_higher']:>5.1f}%")

print(f"\nFull results → inference_2026.csv")
