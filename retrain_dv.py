"""Retrain only the draft_value model (no consensus feature)."""

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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from features import (
    BEAST_COLS, BR_COLS, Z_COLS,
    unified_text, has_br_flag, get_meas,
)

warnings.filterwarnings("ignore")

BASE       = Path("/Users/varunramanathan/Downloads/sentiment-analysis")
MODEL_NAME = "all-mpnet-base-v2"
TFIDF_PARAMS = dict(ngram_range=(1, 2), max_features=15_000, sublinear_tf=True, min_df=2)
SVD_DIM    = 150

print("Loading data...")
df = pd.read_csv(BASE / "data/processed/all_prospects.csv")
st = SentenceTransformer(MODEL_NAME)

has_beast = df["beast_summary"].notna()

dv_df = df[has_beast & df["draft_tier"].isin(["slide","consensus","riser"])
           & df["round"].notna()].copy().reset_index(drop=True)

texts_dv  = unified_text(dv_df)
X_src_dv  = np.zeros((len(dv_df), 1))
X_br_dv   = has_br_flag(dv_df)
X_extra_dv = X_br_dv  # no consensus — purely text/measurables driven
X_emb_dv  = st.encode(texts_dv.tolist(), batch_size=64,
                       show_progress_bar=True, normalize_embeddings=True)
X_meas_dv = get_meas(dv_df)

le_dv = LabelEncoder()
y_dv  = le_dv.fit_transform(dv_df["draft_tier"])

print(f"Dataset: {len(dv_df)} rows | {dict(dv_df['draft_tier'].value_counts())}")
print(f"Classes: {list(le_dv.classes_)}")

cv5_dv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

def xgb_dv(): return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    eval_metric="mlogloss", verbosity=0, random_state=42)
def lr_dv():  return LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
def dummy():  return DummyClassifier(strategy="stratified")

zeros = np.zeros((len(dv_df), 0))

print("\nCross-validated macro-F1 (5-fold):")
s = manual_cv(texts_dv, X_emb_dv, X_meas_dv, X_src_dv, zeros, y_dv, dummy, cv5_dv)
print(f"  Dummy     f1={s.mean():.3f}")
s_xgb = manual_cv(texts_dv, X_emb_dv, X_meas_dv, X_src_dv, X_extra_dv, y_dv, xgb_dv, cv5_dv)
print(f"  XGBoost   f1={s_xgb.mean():.3f} ±{s_xgb.std():.3f}")
s_lr  = manual_cv(texts_dv, X_emb_dv, X_meas_dv, X_src_dv, X_extra_dv, y_dv, lr_dv,  cv5_dv)
print(f"  LogReg    f1={s_lr.mean():.3f} ±{s_lr.std():.3f}")

best_fn = xgb_dv if s_xgb.mean() >= s_lr.mean() else lr_dv

dv_tr_mask = dv_df["draft_year"].between(2019, 2023)
dv_te_mask = dv_df["draft_year"].between(2024, 2025)
print(f"\nYear split — train: 2019-2023 ({dv_tr_mask.sum()}) | test: 2024-2025 ({dv_te_mask.sum()})")

tfidf_dv = TfidfVectorizer(**TFIDF_PARAMS)
svd_dv   = TruncatedSVD(n_components=SVD_DIM, random_state=42)
X_dv_all = np.hstack([svd_dv.fit_transform(tfidf_dv.fit_transform(texts_dv)),
                       X_emb_dv, X_meas_dv, X_src_dv, X_extra_dv])
X_tr, X_te = X_dv_all[dv_tr_mask.values], X_dv_all[dv_te_mask.values]
y_tr, y_te = y_dv[dv_tr_mask.values],     y_dv[dv_te_mask.values]

best_dv = best_fn()
best_dv.fit(X_tr, y_tr)
print("\nHeld-out classification report (test: 2024-2025):")
print(classification_report(y_te, best_dv.predict(X_te), target_names=le_dv.classes_))

print("\nPrediction distribution on test set:")
preds = le_dv.inverse_transform(best_dv.predict(X_te))
import pandas as pd
print(pd.Series(preds).value_counts())

joblib.dump({"tfidf": tfidf_dv, "svd": svd_dv, "clf": best_dv,
             "le": le_dv, "st_model": MODEL_NAME, "z_cols": Z_COLS,
             "beast_cols": BEAST_COLS, "br_cols": BR_COLS},
            BASE / "models/model_draft_value.pkl")
print("\nSaved → model_draft_value.pkl")
