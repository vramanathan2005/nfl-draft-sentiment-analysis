"""
Beast+PFF stacked models.

sc_tier uses binary label (made_it vs didnt) and recovers 2017/2018/2021
rows where Beast text is unavailable by falling back to PFF text.
An explicit text_source feature (0=Beast, 1=PFF) lets the model account
for vocabulary differences between the two sources.

Features:
  - TF-IDF (SVD 150d) on unified scouting text
  - Sentence embeddings (384d) on unified scouting text
  - Z-scored measurables (11d), 0-imputed
  - text_source (1d binary)
  - beast_grade_round (1d, 1-8 where 8=undrafted projection, 0-imputed)

Tasks:
  sc_tier      — 2017-2021 drafts, Beast OR PFF text, ~1,373 rows, 3-class
  draft_value  — 2019-2024 drafts with Beast text, 1343 rows
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
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE        = Path("/Users/varunramanathan/Downloads/sentiment-analysis")
MODEL_NAME  = "all-MiniLM-L6-v2"
BEAST_COLS  = ["beast_summary", "beast_strengths", "beast_weaknesses"]
PFF_COLS    = ["pff_overview", "pff_pros", "pff_cons", "pff_bottom_line", "pff_extra"]
Z_COLS      = ["z_ht_in","z_wt_lbs","z_arm_in","z_hand_in","z_wing_in",
               "z_dash40","z_vj_in","z_bj_in","z_shuttle","z_cone3","z_bench"]
TFIDF_PARAMS = dict(ngram_range=(1, 2), max_features=15_000,
                    sublinear_tf=True, min_df=2)
SVD_DIM = 150


def parse_beast_grade(val):
    if pd.isna(val):
        return 0.0
    s = str(val).lower()
    if any(x in s for x in ["undrafted", "priority free", "udfa", "free agent"]):
        return 8.0
    m = re.search(r"(\d+)(?:st|nd|rd|th)", s)
    return float(m.group(1)) if m else 0.0


def beast_grade_feature(df):
    return df["beast_grade"].map(parse_beast_grade).values.reshape(-1, 1)


def beast_text(df):
    return df[BEAST_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()


def pff_text(df):
    return df[PFF_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()


def unified_text(df):
    """Beast text if available, else PFF text."""
    bt = beast_text(df)
    pt = pff_text(df)
    return bt.where(bt.str.len() > 0, pt)


def text_source_flag(df):
    """0 = Beast text used, 1 = PFF text used (Beast was missing)."""
    bt = beast_text(df)
    return (bt.str.len() == 0).astype(float).values.reshape(-1, 1)


def get_meas(df):
    return df[Z_COLS].fillna(0.0).values


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def manual_cv(texts, X_emb, X_meas, X_src, X_grade, y, clf_fn, cv, scoring):
    scores = []
    for train_idx, val_idx in cv.split(texts, y):
        tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        svd   = TruncatedSVD(n_components=SVD_DIM, random_state=42)
        Xtr = np.hstack([svd.fit_transform(tfidf.fit_transform(texts.iloc[train_idx])),
                         X_emb[train_idx], X_meas[train_idx], X_src[train_idx], X_grade[train_idx]])
        Xval = np.hstack([svd.transform(tfidf.transform(texts.iloc[val_idx])),
                          X_emb[val_idx], X_meas[val_idx], X_src[val_idx], X_grade[val_idx]])
        clf = clf_fn()
        clf.fit(Xtr, y[train_idx])
        if scoring == "f1_macro":
            scores.append(f1_score(y[val_idx], clf.predict(Xval), average="macro"))
        elif scoring == "roc_auc":
            scores.append(roc_auc_score(y[val_idx], clf.predict_proba(Xval)[:, 1]))
    return np.array(scores)


def rpt(name, scores, metric):
    print(f"  {name:<50} {metric}: {scores.mean():.3f} ±{scores.std():.3f}")


# ─────────────────────────────────────────────────────────────────────────────
print("Loading data and model...")
df = pd.read_csv(BASE / "all_prospects.csv")
st = SentenceTransformer(MODEL_NAME)

has_beast = df["beast_summary"].notna()
has_pff   = df["pff_overview"].notna()


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: sc_binary  (binary, Beast OR PFF text, 2017-2021)
# ─────────────────────────────────────────────────────────────────────────────

section("TASK 1a: sc_binary  (Beast+PFF, 2017-2021, binary: made_it vs didnt)")

sc_df = df[df["sc_binary"].isin(["made_it", "didnt"])
           & (has_beast | has_pff)].copy().reset_index(drop=True)

texts_sc   = unified_text(sc_df)
X_src_sc   = text_source_flag(sc_df)
X_grade_sc = beast_grade_feature(sc_df)
X_emb_sc   = st.encode(texts_sc.tolist(), batch_size=64,
                        show_progress_bar=False, normalize_embeddings=True)
X_meas_sc  = get_meas(sc_df)

le_bin = LabelEncoder()
y_sc_bin = le_bin.fit_transform(sc_df["sc_binary"])

print(f"  Dataset: {len(sc_df)} rows | {dict(sc_df['sc_binary'].value_counts())}")
print(f"  Beast rows: {(X_src_sc[:,0]==0).sum()} | PFF-only rows: {(X_src_sc[:,0]==1).sum()}")

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def xgb_bin():
    return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="logloss", verbosity=0, random_state=42)
def lr_bin():
    return LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
def dummy_clf():
    return DummyClassifier(strategy="stratified")

zeros0 = np.zeros((len(sc_df), 0))

print("\n  Cross-validated ROC-AUC (5-fold):")
s = manual_cv(texts_sc, X_emb_sc, zeros0, zeros0, zeros0, y_sc_bin, dummy_clf, cv5, "roc_auc")
rpt("Dummy", s, "roc_auc")
s_bin_xgb = manual_cv(texts_sc, X_emb_sc, X_meas_sc, X_src_sc, zeros0, y_sc_bin, xgb_bin, cv5, "roc_auc")
rpt("no grade  (XGBoost)", s_bin_xgb, "roc_auc")
s_bin_grade = manual_cv(texts_sc, X_emb_sc, X_meas_sc, X_src_sc, X_grade_sc, y_sc_bin, xgb_bin, cv5, "roc_auc")
rpt("+ beast_grade  (XGBoost)", s_bin_grade, "roc_auc")
s_bin_lr = manual_cv(texts_sc, X_emb_sc, X_meas_sc, X_src_sc, X_grade_sc, y_sc_bin, lr_bin, cv5, "roc_auc")
rpt("+ beast_grade  (LogReg)", s_bin_lr, "roc_auc")

print("\n  Cross-validated macro-F1 (5-fold):")
s = manual_cv(texts_sc, X_emb_sc, X_meas_sc, X_src_sc, zeros0, y_sc_bin, xgb_bin, cv5, "f1_macro")
rpt("no grade  (XGBoost)", s, "f1_macro")
s = manual_cv(texts_sc, X_emb_sc, X_meas_sc, X_src_sc, X_grade_sc, y_sc_bin, xgb_bin, cv5, "f1_macro")
rpt("+ beast_grade  (XGBoost)", s, "f1_macro")
s = manual_cv(texts_sc, X_emb_sc, X_meas_sc, X_src_sc, X_grade_sc, y_sc_bin, lr_bin, cv5, "f1_macro")
rpt("+ beast_grade  (LogReg)", s, "f1_macro")

bin_auc = max(s_bin_grade.mean(), s_bin_lr.mean())

# ─────────────────────────────────────────────────────────────────────────────
section("TASK 1b: sc_tier  (Beast+PFF, 2017-2021, 3-class)")

sc3_df = df[df["sc_tier"].isin(["real_contract","practice_only","out_of_league"])
            & (has_beast | has_pff)].copy().reset_index(drop=True)

texts_sc3   = unified_text(sc3_df)
X_src_sc3   = text_source_flag(sc3_df)
X_grade_sc3 = beast_grade_feature(sc3_df)
X_emb_sc3   = st.encode(texts_sc3.tolist(), batch_size=64,
                         show_progress_bar=False, normalize_embeddings=True)
X_meas_sc3  = get_meas(sc3_df)

le3 = LabelEncoder()
y_sc3 = le3.fit_transform(sc3_df["sc_tier"])

print(f"  Dataset: {len(sc3_df)} rows | {dict(sc3_df['sc_tier'].value_counts())}")
print(f"  Beast rows: {(X_src_sc3[:,0]==0).sum()} | PFF-only rows: {(X_src_sc3[:,0]==1).sum()}")

def xgb_multi():
    return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="mlogloss", verbosity=0, random_state=42)
def lr_multi():
    return LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)

zeros3 = np.zeros((len(sc3_df), 0))

print("\n  Cross-validated macro-F1 (5-fold):")
s = manual_cv(texts_sc3, X_emb_sc3, zeros3, zeros3, zeros3, y_sc3, dummy_clf, cv5, "f1_macro")
rpt("Dummy", s, "f1_macro")
s_3cls_xgb = manual_cv(texts_sc3, X_emb_sc3, X_meas_sc3, X_src_sc3, zeros3, y_sc3, xgb_multi, cv5, "f1_macro")
rpt("no grade  (XGBoost)", s_3cls_xgb, "f1_macro")
s_3cls_grade = manual_cv(texts_sc3, X_emb_sc3, X_meas_sc3, X_src_sc3, X_grade_sc3, y_sc3, xgb_multi, cv5, "f1_macro")
rpt("+ beast_grade  (XGBoost)", s_3cls_grade, "f1_macro")
s_3cls_lr = manual_cv(texts_sc3, X_emb_sc3, X_meas_sc3, X_src_sc3, X_grade_sc3, y_sc3, lr_multi, cv5, "f1_macro")
rpt("+ beast_grade  (LogReg)", s_3cls_lr, "f1_macro")

three_f1 = max(s_3cls_grade.mean(), s_3cls_lr.mean())
use_grade_sc = s_3cls_grade.mean() >= s_3cls_xgb.mean()

# ── Pick best model and save ──────────────────────────────────────────────────
USE_3CLASS = three_f1 >= 0.46
print(f"\n  Binary best ROC-AUC: {bin_auc:.3f}  |  3-class best F1: {three_f1:.3f}")
print(f"  → Using {'3-class' if USE_3CLASS else 'binary'} model")

if USE_3CLASS:
    best_clf_fn = xgb_multi if s_3cls_grade.mean() >= s_3cls_lr.mean() else lr_multi
    use_texts, use_emb, use_meas, use_src = texts_sc3, X_emb_sc3, X_meas_sc3, X_src_sc3
    use_grade = X_grade_sc3
    use_y, use_le = y_sc3, le3
else:
    best_clf_fn = xgb_bin if s_bin_grade.mean() >= s_bin_lr.mean() else lr_bin
    use_texts, use_emb, use_meas, use_src = texts_sc, X_emb_sc, X_meas_sc, X_src_sc
    use_grade = X_grade_sc
    use_y, use_le = y_sc_bin, le_bin

print("\n  Held-out classification report:")
tfidf_sc = TfidfVectorizer(**TFIDF_PARAMS)
svd_sc   = TruncatedSVD(n_components=SVD_DIM, random_state=42)
X_sc_all = np.hstack([svd_sc.fit_transform(tfidf_sc.fit_transform(use_texts)),
                       use_emb, use_meas, use_src, use_grade])
X_tr, X_te, y_tr, y_te = train_test_split(
    X_sc_all, use_y, test_size=0.2, stratify=use_y, random_state=42)
best_sc = best_clf_fn()
best_sc.fit(X_tr, y_tr)
if not USE_3CLASS:
    print(f"  Held-out AUC: {roc_auc_score(y_te, best_sc.predict_proba(X_te)[:,1]):.3f}")
print(classification_report(y_te, best_sc.predict(X_te), target_names=use_le.classes_))

joblib.dump({"tfidf": tfidf_sc, "svd": svd_sc, "clf": best_sc,
             "le": use_le, "st_model": MODEL_NAME, "z_cols": Z_COLS,
             "beast_cols": BEAST_COLS, "pff_cols": PFF_COLS,
             "binary": not USE_3CLASS},
            BASE / "model_sc_tier.pkl")
print("  Saved → model_sc_tier.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: draft_value binary  (2019-2024, Beast text + beast_grade)
# ─────────────────────────────────────────────────────────────────────────────

section("TASK 2: draft_tier 3-class  (Beast text, 2019-2024, slide/consensus/reach)")

dv_df = df[has_beast & df["draft_tier"].isin(["slide","consensus","reach"])
           & df["round"].notna()].copy().reset_index(drop=True)

texts_dv   = beast_text(dv_df)
X_src_dv   = np.zeros((len(dv_df), 1))
X_grade_dv = beast_grade_feature(dv_df)
X_emb_dv   = st.encode(texts_dv.tolist(), batch_size=64,
                        show_progress_bar=False, normalize_embeddings=True)
X_meas_dv  = get_meas(dv_df)

le_dv  = LabelEncoder()
y_dv   = le_dv.fit_transform(dv_df["draft_tier"])

print(f"  Dataset: {len(dv_df)} rows | {dict(dv_df['draft_tier'].value_counts())}")

cv5_dv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def xgb_dv():
    return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="mlogloss", verbosity=0, random_state=42)
def lr_dv():
    return LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)

zeros_dv = np.zeros((len(dv_df), 0))

print("\n  Cross-validated macro-F1 (5-fold):")
s = manual_cv(texts_dv, X_emb_dv, zeros_dv, zeros_dv, zeros_dv,
              y_dv, dummy_clf, cv5_dv, "f1_macro")
rpt("Dummy", s, "f1_macro")
s_dv_base = manual_cv(texts_dv, X_emb_dv, X_meas_dv, X_src_dv, zeros_dv,
                       y_dv, xgb_dv, cv5_dv, "f1_macro")
rpt("no grade  (XGBoost)", s_dv_base, "f1_macro")
s_dv_grade = manual_cv(texts_dv, X_emb_dv, X_meas_dv, X_src_dv, X_grade_dv,
                        y_dv, xgb_dv, cv5_dv, "f1_macro")
rpt("+ beast_grade  (XGBoost)", s_dv_grade, "f1_macro")
s_dv_lr = manual_cv(texts_dv, X_emb_dv, X_meas_dv, X_src_dv, X_grade_dv,
                     y_dv, lr_dv, cv5_dv, "f1_macro")
rpt("+ beast_grade  (LogReg)", s_dv_lr, "f1_macro")

best_dv_fn = xgb_dv if s_dv_grade.mean() >= s_dv_lr.mean() else lr_dv

tfidf_dv = TfidfVectorizer(**TFIDF_PARAMS)
svd_dv   = TruncatedSVD(n_components=SVD_DIM, random_state=42)
X_dv_all = np.hstack([svd_dv.fit_transform(tfidf_dv.fit_transform(texts_dv)),
                       X_emb_dv, X_meas_dv, X_src_dv, X_grade_dv])
X_tr, X_te, y_tr, y_te = train_test_split(
    X_dv_all, y_dv, test_size=0.2, stratify=y_dv, random_state=42)
best_dv = best_dv_fn()
best_dv.fit(X_tr, y_tr)
print(f"\n  Held-out classification report:")
print(classification_report(y_te, best_dv.predict(X_te), target_names=le_dv.classes_))

joblib.dump({"tfidf": tfidf_dv, "svd": svd_dv, "clf": best_dv,
             "le": le_dv, "st_model": MODEL_NAME, "z_cols": Z_COLS,
             "beast_cols": BEAST_COLS, "use_grade": True},
            BASE / "model_draft_value.pkl")
print("  Saved → model_draft_value.pkl")

print("\nDone.")
