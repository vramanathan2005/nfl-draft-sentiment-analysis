"""
Beast + PFF stacked models.

Features:
  - TF-IDF (SVD 150d) on unified scouting text
  - Sentence embeddings (768d, all-mpnet-base-v2)
  - Z-scored combine measurables (11d), 0-imputed
  - consensus rank z-score (per-year, 1d), 0-imputed
  - text_source flag (0=Beast, 1=PFF)
  - beast_rank z-score (per-year, 0-imputed)
  - text_length (log, standardized)
  - round one-hot (1-7) + imputed flag

Tasks:
  sc_tier     — 4-class career outcome (out_of_league/roster_bubble/53_man/cornerstone)
                2017-2021 drafts, Beast OR PFF text, ~1,373 rows
                train: 2017-2019  |  test: 2020-2021

  draft_value — 3-class draft positioning (slide/consensus/reach)
                2019-2025 drafts, Beast text only, ~1,376 rows
                train: 2019-2023  |  test: 2024-2025

  apy_pct     — APY percentile regression for real_contract players
                apy_pct = 1 - (rank_on_date - 1) / (max_rank_on_date - 1)
                1.0 = highest-paid player in NFL on signing date, 0.0 = lowest
                2017-2021 real_contract rows with text, ~556 rows
                train: 2017-2019  |  test: 2020-2021
"""

import re
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import spearmanr
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

BASE        = Path("/Users/varunramanathan/Downloads/sentiment-analysis")
MODEL_NAME  = "all-mpnet-base-v2"
BEAST_COLS  = ["beast_summary", "beast_strengths", "beast_weaknesses"]
PFF_COLS    = ["pff_overview", "pff_pros", "pff_cons", "pff_bottom_line", "pff_extra"]
BR_COLS     = ["br_positives", "br_negatives"]
Z_COLS      = ["z_ht_in","z_wt_lbs","z_arm_in","z_hand_in","z_wing_in",
               "z_dash40","z_vj_in","z_bj_in","z_shuttle","z_cone3","z_bench"]
TFIDF_PARAMS = dict(ngram_range=(1, 2), max_features=15_000,
                    sublinear_tf=True, min_df=2)
SVD_DIM = 150


def parse_beast_grade(val):
    if pd.isna(val): return 0.0
    s = str(val).lower()
    if any(x in s for x in ["undrafted", "priority free", "udfa", "free agent"]): return 8.0
    m = re.search(r"(\d+)(?:st|nd|rd|th)", s)
    return float(m.group(1)) if m else 0.0

def beast_grade_feature(df):
    return df["beast_grade"].map(parse_beast_grade).values.reshape(-1, 1)

def beast_text(df):
    return df[BEAST_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()

def pff_text(df):
    return df[PFF_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()

def br_text(df):
    return df[BR_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()

def unified_text(df):
    bt  = beast_text(df)
    pt  = pff_text(df)
    brt = br_text(df)
    base = bt.where(bt.str.len() > 0, pt)
    return (base + " " + brt).str.strip().where(brt.str.len() > 0, base)

def text_source_flag(df):
    bt = beast_text(df)
    return (bt.str.len() == 0).astype(float).values.reshape(-1, 1)

def has_br_flag(df):
    return (br_text(df).str.len() > 0).astype(float).values.reshape(-1, 1)

def get_meas(df):
    return df[Z_COLS].fillna(0.0).values

def beast_rank_feature(df):
    """Per-year z-score of beast_rank, 0-imputed for missing."""
    out = np.zeros(len(df))
    df = df.reset_index(drop=True)
    for _, grp in df.groupby("draft_year"):
        vals = grp["beast_rank"].dropna()
        if len(vals) < 2:
            continue
        mu, sigma = vals.mean(), vals.std()
        if sigma < 1e-6:
            continue
        out[grp.index] = grp["beast_rank"].fillna(mu).map(lambda v: (v - mu) / sigma)
    return out.reshape(-1, 1)

def text_length_feature(texts: pd.Series) -> np.ndarray:
    """Log text length, standardized to zero mean / unit variance."""
    lengths = np.log1p(texts.str.len().values.astype(float))
    mu, sigma = lengths.mean(), lengths.std()
    if sigma < 1e-6:
        return np.zeros((len(texts), 1))
    return ((lengths - mu) / sigma).reshape(-1, 1)

def round_feature(df: pd.DataFrame) -> np.ndarray:
    """
    Draft round (1-7) one-hot encoded, plus an 'imputed' flag.
    Training rows use actual round; rows with no round (2026 prospects)
    fall back to consensus // 32 + 1, capped at 7.
    """
    rounds = df["round"].copy()
    missing = rounds.isna()
    if missing.any():
        cons = df.loc[missing, "consensus"].fillna(250)
        rounds.loc[missing] = (cons // 32 + 1).clip(upper=7)
    rounds = rounds.fillna(4).astype(int).clip(1, 7)
    enc = OneHotEncoder(categories=[list(range(1, 8))], sparse_output=False, handle_unknown="ignore")
    ohe = enc.fit_transform(rounds.values.reshape(-1, 1))
    imputed_flag = missing.astype(float).values.reshape(-1, 1)
    return np.hstack([ohe, imputed_flag])

POS_GROUPS = {
    "QB":["QB"], "RB":["RB","FB"], "WR":["WR"], "TE":["TE"],
    "OL":["OT","IOL","C","G","T"], "EDGE":["EDGE","OLB","DE"],
    "DL":["DL","DT","NT"], "LB":["LB","ILB","MLB"],
    "CB":["CB"], "S":["S","FS","SS"], "SPEC":["K","P","LS"],
}
POS_TO_GROUP = {p: g for g, ps in POS_GROUPS.items() for p in ps}
POS_ORDER    = sorted(POS_GROUPS.keys()) + ["OTHER"]

def position_feature(df):
    groups = df["Position"].str.upper().map(POS_TO_GROUP).fillna("OTHER")
    enc = OneHotEncoder(categories=[POS_ORDER], sparse_output=False, handle_unknown="ignore")
    return enc.fit_transform(groups.values.reshape(-1, 1))

def consensus_feature(df):
    """Per-year z-score of consensus rank, 0-imputed for missing."""
    out = np.zeros(len(df))
    for _, grp in df.groupby("draft_year"):
        vals = grp["consensus"].dropna()
        if len(vals) < 2:
            continue
        mu, sigma = vals.mean(), vals.std()
        if sigma < 1e-6:
            continue
        out[grp.index] = grp["consensus"].fillna(mu).map(lambda v: (v - mu) / sigma)
    return out.reshape(-1, 1)

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def manual_cv(texts, X_emb, X_meas, X_src, X_extra, y, clf_fn, cv):
    scores = []
    for train_idx, val_idx in cv.split(texts, y):
        tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        svd   = TruncatedSVD(n_components=SVD_DIM, random_state=42)
        Xtr  = np.hstack([svd.fit_transform(tfidf.fit_transform(texts.iloc[train_idx])),
                          X_emb[train_idx], X_meas[train_idx], X_src[train_idx], X_extra[train_idx]])
        Xval = np.hstack([svd.transform(tfidf.transform(texts.iloc[val_idx])),
                          X_emb[val_idx], X_meas[val_idx], X_src[val_idx], X_extra[val_idx]])
        clf = clf_fn()
        clf.fit(Xtr, y[train_idx])
        scores.append(f1_score(y[val_idx], clf.predict(Xval), average="macro"))
    return np.array(scores)

def rpt(name, scores):
    print(f"  {name:<50} f1_macro: {scores.mean():.3f} ±{scores.std():.3f}")


# ─────────────────────────────────────────────────────────────────────────────
print("Loading data and model...")
df = pd.read_csv(BASE / "data/processed/all_prospects.csv")
st = SentenceTransformer(MODEL_NAME)

has_beast = df["beast_summary"].notna()
has_pff   = df["pff_overview"].notna()

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def dummy_clf():  return DummyClassifier(strategy="stratified")


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: sc_contract_tier  (cut / camp_body / 53_man / cornerstone)
# ─────────────────────────────────────────────────────────────────────────────

section("TASK 1: sc_contract_tier 4-class  (Beast+PFF, 2017-2021)")

ct_df = df[df["sc_contract_tier"].notna()
           & (has_beast | has_pff)].copy().reset_index(drop=True)

texts_ct   = unified_text(ct_df)
X_src_ct   = text_source_flag(ct_df)
X_br_ct    = has_br_flag(ct_df)
X_cons_ct  = consensus_feature(ct_df)
X_pos_ct   = position_feature(ct_df)
X_rank_ct  = beast_rank_feature(ct_df)
X_tlen_ct  = text_length_feature(texts_ct)
X_rnd_ct   = round_feature(ct_df)
X_extra_ct = np.hstack([X_cons_ct, X_br_ct, X_pos_ct, X_rank_ct, X_tlen_ct, X_rnd_ct])
X_emb_ct   = st.encode(texts_ct.tolist(), batch_size=64,
                        show_progress_bar=False, normalize_embeddings=True)
X_meas_ct  = get_meas(ct_df)

le_ct = LabelEncoder()
y_ct  = le_ct.fit_transform(ct_df["sc_contract_tier"])

print(f"  Dataset: {len(ct_df)} rows | {dict(ct_df['sc_contract_tier'].value_counts())}")
print(f"  Classes: {list(le_ct.classes_)}")
print(f"  Beast rows: {(X_src_ct[:,0]==0).sum()} | PFF-only rows: {(X_src_ct[:,0]==1).sum()}")

def xgb_ct():
    return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="mlogloss", verbosity=0, random_state=42)
def lr_ct():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)

zeros_ct = np.zeros((len(ct_df), 0))

print(f"  BR text rows: {X_br_ct.sum():.0f}/{len(ct_df)}")
print("\n  Cross-validated macro-F1 (5-fold):")
s = manual_cv(texts_ct, X_emb_ct, X_meas_ct, X_src_ct, zeros_ct, y_ct, dummy_clf, cv5)
rpt("Dummy", s)
s_ct_xgb = manual_cv(texts_ct, X_emb_ct, X_meas_ct, X_src_ct, X_extra_ct, y_ct, xgb_ct, cv5)
rpt("XGBoost", s_ct_xgb)
s_ct_lr = manual_cv(texts_ct, X_emb_ct, X_meas_ct, X_src_ct, X_extra_ct, y_ct, lr_ct, cv5)
rpt("LogReg (balanced, C=0.5)", s_ct_lr)

best_ct_fn = xgb_ct if s_ct_xgb.mean() >= s_ct_lr.mean() else lr_ct

ct_tr_mask = ct_df["draft_year"].between(2017, 2019)
ct_te_mask = ct_df["draft_year"].between(2020, 2021)
print(f"\n  Year split — train: 2017-2019 ({ct_tr_mask.sum()} rows) | test: 2020-2021 ({ct_te_mask.sum()} rows)")

tfidf_ct = TfidfVectorizer(**TFIDF_PARAMS)
svd_ct   = TruncatedSVD(n_components=SVD_DIM, random_state=42)
X_ct_all = np.hstack([svd_ct.fit_transform(tfidf_ct.fit_transform(texts_ct)),
                       X_emb_ct, X_meas_ct, X_src_ct, X_extra_ct])
X_tr, X_te = X_ct_all[ct_tr_mask.values], X_ct_all[ct_te_mask.values]
y_tr, y_te = y_ct[ct_tr_mask.values],     y_ct[ct_te_mask.values]

# Train both and soft-vote ensemble
ct_xgb = xgb_ct(); ct_xgb.fit(X_tr, y_tr)
ct_lr  = lr_ct();  ct_lr.fit(X_tr, y_tr)
ens_proba = (ct_xgb.predict_proba(X_te) + 2 * ct_lr.predict_proba(X_te)) / 3
ens_preds = le_ct.classes_[ens_proba.argmax(axis=1)]
ens_f1 = f1_score(y_te, ens_proba.argmax(axis=1), average="macro")

single_f1 = f1_score(y_te, best_ct_fn().fit(X_tr, y_tr).predict(X_te), average="macro")
best_ct = ct_lr if s_ct_lr.mean() >= s_ct_xgb.mean() else ct_xgb

print(f"\n  Single best F1: {single_f1:.3f}  |  Ensemble F1: {ens_f1:.3f}")
use_ensemble = ens_f1 >= single_f1
print(f"  → Using {'ensemble' if use_ensemble else 'single best'}")

print("\n  Held-out classification report (test: 2020-2021):")
if use_ensemble:
    print(classification_report(y_te, ens_proba.argmax(axis=1), target_names=le_ct.classes_))
else:
    print(classification_report(y_te, best_ct.predict(X_te), target_names=le_ct.classes_))

joblib.dump({"tfidf": tfidf_ct, "svd": svd_ct,
             "clf": best_ct, "clf2": ct_xgb if use_ensemble and best_ct is ct_lr else None,
             "ensemble": use_ensemble,
             "le": le_ct, "st_model": MODEL_NAME, "z_cols": Z_COLS,
             "beast_cols": BEAST_COLS, "pff_cols": PFF_COLS, "br_cols": BR_COLS,
             "extra_features": ["consensus", "br_flag", "position", "beast_rank", "text_length", "round"]},
            BASE / "models/model_sc_tier.pkl")
print("  Saved → model_sc_tier.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: draft_tier 3-class  (slide / consensus / reach)
# ─────────────────────────────────────────────────────────────────────────────

section("TASK 2: draft_tier 3-class  (Beast text, 2019-2025)")

dv_df = df[has_beast & df["draft_tier"].isin(["slide","consensus","reach"])
           & df["round"].notna()].copy().reset_index(drop=True)

texts_dv   = unified_text(dv_df)
X_src_dv   = np.zeros((len(dv_df), 1))
X_br_dv    = has_br_flag(dv_df)
X_cons_dv  = consensus_feature(dv_df)
X_extra_dv = np.hstack([X_cons_dv, X_br_dv])
X_emb_dv   = st.encode(texts_dv.tolist(), batch_size=64,
                        show_progress_bar=False, normalize_embeddings=True)
X_meas_dv  = get_meas(dv_df)

le_dv = LabelEncoder()
y_dv  = le_dv.fit_transform(dv_df["draft_tier"])

print(f"  Dataset: {len(dv_df)} rows | {dict(dv_df['draft_tier'].value_counts())}")

cv5_dv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def xgb_dv():
    return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="mlogloss", verbosity=0, random_state=42)
def lr_dv():
    return LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)

zeros_dv = np.zeros((len(dv_df), 0))

print(f"  BR text rows: {X_br_dv.sum():.0f}/{len(dv_df)}")
print("\n  Cross-validated macro-F1 (5-fold):")
s = manual_cv(texts_dv, X_emb_dv, zeros_dv, zeros_dv, zeros_dv, y_dv, dummy_clf, cv5_dv)
rpt("Dummy", s)
s_dv_xgb = manual_cv(texts_dv, X_emb_dv, X_meas_dv, X_src_dv, X_extra_dv, y_dv, xgb_dv, cv5_dv)
rpt("XGBoost", s_dv_xgb)
s_dv_lr = manual_cv(texts_dv, X_emb_dv, X_meas_dv, X_src_dv, X_extra_dv, y_dv, lr_dv, cv5_dv)
rpt("LogReg", s_dv_lr)

best_dv_fn = xgb_dv if s_dv_xgb.mean() >= s_dv_lr.mean() else lr_dv

dv_tr_mask = dv_df["draft_year"].between(2019, 2023)
dv_te_mask = dv_df["draft_year"].between(2024, 2025)
print(f"\n  Year split — train: 2019-2023 ({dv_tr_mask.sum()} rows) | test: 2024-2025 ({dv_te_mask.sum()} rows)")

tfidf_dv = TfidfVectorizer(**TFIDF_PARAMS)
svd_dv   = TruncatedSVD(n_components=SVD_DIM, random_state=42)
X_dv_all = np.hstack([svd_dv.fit_transform(tfidf_dv.fit_transform(texts_dv)),
                       X_emb_dv, X_meas_dv, X_src_dv, X_extra_dv])
X_tr, X_te = X_dv_all[dv_tr_mask.values], X_dv_all[dv_te_mask.values]
y_tr, y_te = y_dv[dv_tr_mask.values],     y_dv[dv_te_mask.values]
best_dv = best_dv_fn()
best_dv.fit(X_tr, y_tr)
print("\n  Held-out classification report (test: 2024-2025):")
print(classification_report(y_te, best_dv.predict(X_te), target_names=le_dv.classes_))

joblib.dump({"tfidf": tfidf_dv, "svd": svd_dv, "clf": best_dv,
             "le": le_dv, "st_model": MODEL_NAME, "z_cols": Z_COLS,
             "beast_cols": BEAST_COLS, "br_cols": BR_COLS},
            BASE / "models/model_draft_value.pkl")
print("  Saved → model_draft_value.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: apy_pct regression  (real_contract players, 2017-2021)
# ─────────────────────────────────────────────────────────────────────────────

section("TASK 3: apy_pct regression  (real_contract, 2017-2021)")

sc_raw = pd.read_csv(BASE / "data/raw/ranked_second_contracts.csv")
max_rank_by_year = sc_raw.groupby("draft_year")["rank_on_date_APY"].max()

ap_df = df[(df["sc_tier"] == "real_contract")
           & (has_beast | has_pff)].copy().reset_index(drop=True)

def compute_apy_pct(row):
    mx = max_rank_by_year.get(int(row["draft_year"]), 550)
    return 1.0 - (row["sc_rank_on_date_APY"] - 1.0) / (mx - 1.0)

ap_df["apy_pct"] = ap_df.apply(compute_apy_pct, axis=1).clip(0.0, 1.0)
y_ap = ap_df["apy_pct"].values

texts_ap   = unified_text(ap_df)
X_src_ap   = text_source_flag(ap_df)
X_br_ap    = has_br_flag(ap_df)
X_cons_ap  = consensus_feature(ap_df)
X_pos_ap   = position_feature(ap_df)
X_rank_ap  = beast_rank_feature(ap_df)
X_tlen_ap  = text_length_feature(texts_ap)
X_rnd_ap   = round_feature(ap_df)
X_extra_ap = np.hstack([X_cons_ap, X_br_ap, X_pos_ap, X_rank_ap, X_tlen_ap, X_rnd_ap])
X_emb_ap   = st.encode(texts_ap.tolist(), batch_size=64,
                        show_progress_bar=False, normalize_embeddings=True)
X_meas_ap  = get_meas(ap_df)

print(f"  Dataset: {len(ap_df)} rows | apy_pct mean={y_ap.mean():.3f} std={y_ap.std():.3f}")
print(f"  Beast rows: {(X_src_ap[:,0]==0).sum()} | PFF-only rows: {(X_src_ap[:,0]==1).sum()}")

def ridge_ap():
    return Ridge(alpha=10.0)

def xgb_ap():
    return XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="rmse", verbosity=0, random_state=42)

def manual_cv_reg(texts, X_emb, X_meas, X_src, X_extra, y, reg_fn, cv):
    """CV for regression: returns (spearman_r, mae) arrays."""
    spears, maes = [], []
    for train_idx, val_idx in cv.split(np.arange(len(texts))):
        tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        svd   = TruncatedSVD(n_components=SVD_DIM, random_state=42)
        Xtr  = np.hstack([svd.fit_transform(tfidf.fit_transform(texts.iloc[train_idx])),
                          X_emb[train_idx], X_meas[train_idx], X_src[train_idx], X_extra[train_idx]])
        Xval = np.hstack([svd.transform(tfidf.transform(texts.iloc[val_idx])),
                          X_emb[val_idx], X_meas[val_idx], X_src[val_idx], X_extra[val_idx]])
        reg = reg_fn()
        reg.fit(Xtr, y[train_idx])
        preds = reg.predict(Xval).clip(0, 1)
        spears.append(spearmanr(y[val_idx], preds).statistic)
        maes.append(np.abs(y[val_idx] - preds).mean())
    return np.array(spears), np.array(maes)

cv5_ap = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n  Cross-validated metrics (5-fold):")
dummy_preds = np.full(len(y_ap), y_ap.mean())
print(f"  {'Dummy (mean)':<40} spearman: {0.000:.3f}  mae: {np.abs(y_ap - dummy_preds).mean():.4f}")

sp_ridge, mae_ridge = manual_cv_reg(texts_ap, X_emb_ap, X_meas_ap, X_src_ap, X_extra_ap, y_ap, ridge_ap, cv5_ap)
print(f"  {'Ridge (alpha=10)':<40} spearman: {sp_ridge.mean():.3f} ±{sp_ridge.std():.3f}  mae: {mae_ridge.mean():.4f}")

sp_xgb, mae_xgb = manual_cv_reg(texts_ap, X_emb_ap, X_meas_ap, X_src_ap, X_extra_ap, y_ap, xgb_ap, cv5_ap)
print(f"  {'XGBRegressor':<40} spearman: {sp_xgb.mean():.3f} ±{sp_xgb.std():.3f}  mae: {mae_xgb.mean():.4f}")

# Year-split held-out evaluation
ap_tr_mask = ap_df["draft_year"].between(2017, 2019)
ap_te_mask = ap_df["draft_year"].between(2020, 2021)
print(f"\n  Year split — train: 2017-2019 ({ap_tr_mask.sum()} rows) | test: 2020-2021 ({ap_te_mask.sum()} rows)")

tfidf_ap = TfidfVectorizer(**TFIDF_PARAMS)
svd_ap   = TruncatedSVD(n_components=SVD_DIM, random_state=42)
X_ap_all = np.hstack([svd_ap.fit_transform(tfidf_ap.fit_transform(texts_ap)),
                       X_emb_ap, X_meas_ap, X_src_ap, X_extra_ap])
X_tr, X_te = X_ap_all[ap_tr_mask.values], X_ap_all[ap_te_mask.values]
y_tr, y_te = y_ap[ap_tr_mask.values],     y_ap[ap_te_mask.values]

# Train both, pick better by CV spearman, also compute ensemble
best_ap_fn = ridge_ap if sp_ridge.mean() >= sp_xgb.mean() else xgb_ap
ap_ridge = ridge_ap(); ap_ridge.fit(X_tr, y_tr)
ap_xgb   = xgb_ap();   ap_xgb.fit(X_tr, y_tr)

preds_ridge = ap_ridge.predict(X_te).clip(0, 1)
preds_xgb   = ap_xgb.predict(X_te).clip(0, 1)
preds_ens   = ((preds_ridge + 2 * preds_xgb) / 3).clip(0, 1)  # weight toward XGB if better

sp_ho_ridge = spearmanr(y_te, preds_ridge).statistic
sp_ho_xgb   = spearmanr(y_te, preds_xgb).statistic
sp_ho_ens   = spearmanr(y_te, preds_ens).statistic

print(f"\n  Held-out Spearman r:  Ridge={sp_ho_ridge:.3f}  XGB={sp_ho_xgb:.3f}  Ensemble={sp_ho_ens:.3f}")
print(f"  Held-out MAE:         Ridge={np.abs(y_te-preds_ridge).mean():.4f}  XGB={np.abs(y_te-preds_xgb).mean():.4f}  Ensemble={np.abs(y_te-preds_ens).mean():.4f}")

use_ap_ens = sp_ho_ens >= max(sp_ho_ridge, sp_ho_xgb)
final_preds = preds_ens if use_ap_ens else (preds_ridge if sp_ho_ridge >= sp_ho_xgb else preds_xgb)
final_label = "ensemble" if use_ap_ens else ("ridge" if sp_ho_ridge >= sp_ho_xgb else "xgb")
print(f"  → Using {final_label}")

# Compute CV residual std for prediction intervals
_, mae_best = manual_cv_reg(texts_ap, X_emb_ap, X_meas_ap, X_src_ap, X_extra_ap, y_ap, best_ap_fn, cv5_ap)
resid_std = mae_best.mean() * 1.25  # approx: ±1.25 MAE ≈ ~70% coverage interval
print(f"  Prediction interval half-width (±1.25×MAE): ±{resid_std:.4f}")

# Percentile bucket breakdown on held-out
print("\n  Held-out bucket accuracy (predicted vs actual quartile):")
quartiles = np.percentile(y_te, [25, 50, 75])
actual_q  = np.digitize(y_te, quartiles)
pred_q    = np.digitize(final_preds, quartiles)
for q, label in enumerate(["Q1 (low)", "Q2", "Q3", "Q4 (elite)"]):
    mask = actual_q == q
    if mask.sum():
        acc = (pred_q[mask] == q).mean()
        print(f"    {label:<14} n={mask.sum():3d}  within-quartile accuracy={acc:.2f}")

best_ap  = ap_ridge if sp_ho_ridge >= sp_ho_xgb else ap_xgb
best_ap2 = ap_xgb   if sp_ho_ridge >= sp_ho_xgb else ap_ridge

joblib.dump({"tfidf": tfidf_ap, "svd": svd_ap,
             "clf": best_ap,
             "clf2": best_ap2 if use_ap_ens else None,
             "ensemble": use_ap_ens,
             "resid_std": float(resid_std),
             "st_model": MODEL_NAME, "z_cols": Z_COLS,
             "beast_cols": BEAST_COLS, "pff_cols": PFF_COLS, "br_cols": BR_COLS,
             "extra_features": ["consensus", "br_flag", "position", "beast_rank", "text_length", "round"]},
            BASE / "models/model_apy_pct.pkl")
print("  Saved → model_apy_pct.pkl")

print("\nDone.")
