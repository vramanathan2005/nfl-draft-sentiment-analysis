"""
Batch explanation export — one row per 2026 prospect.

Outputs explain_2026.csv with top words and feature contributions
for all three models (sc_tier, draft_tier, apy_pct).

Word contribution math (linear models only):
    effective_word_weights = coef[class] @ SVD.components_   # shape (vocab,)
    contribution(word)     = effective_word_weights[word_idx] × tfidf_weight

Feature slices (must match stacked_models.py build order):
    SC / APY  (954d): tfidf_svd[150] | emb[768] | meas[11] | src[1] | cons[1] | br[1] | pos[12] | rank[1] | tlen[1] | round[8]
    DV        (932d): tfidf_svd[150] | emb[768] | meas[11] | src[1] | cons[1] | br[1]
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

BASE      = Path("/Users/varunramanathan/Downloads/sentiment-analysis")
TOP_N     = 5   # positive words per model
BOTTOM_N  = 3   # negative words per model

BEAST_COLS = ["beast_summary", "beast_strengths", "beast_weaknesses"]
PFF_COLS   = ["pff_overview", "pff_pros", "pff_cons", "pff_bottom_line", "pff_extra"]
BR_COLS    = ["br_positives", "br_negatives"]
Z_COLS     = ["z_ht_in","z_wt_lbs","z_arm_in","z_hand_in","z_wing_in",
              "z_dash40","z_vj_in","z_bj_in","z_shuttle","z_cone3","z_bench"]
MEAS_LABELS = ["height","weight","arm","hand","wingspan",
               "40yd","vert_jump","broad_jump","shuttle","3cone","bench"]

POS_GROUPS  = {"QB":["QB"],"RB":["RB","FB"],"WR":["WR"],"TE":["TE"],
               "OL":["OT","IOL","C","G","T"],"EDGE":["EDGE","OLB","DE"],
               "DL":["DL","DT","NT"],"LB":["LB","ILB","MLB"],
               "CB":["CB"],"S":["S","FS","SS"],"SPEC":["K","P","LS"]}
POS_TO_GROUP = {p: g for g, ps in POS_GROUPS.items() for p in ps}
POS_ORDER    = sorted(POS_GROUPS.keys()) + ["OTHER"]


# ── text helpers ──────────────────────────────────────────────────────────────

def make_texts(df):
    bt  = df[BEAST_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()
    pt  = df[PFF_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()
    brt = df[BR_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()
    base = bt.where(bt.str.len() > 0, pt)
    return (base + " " + brt).str.strip().where(brt.str.len() > 0, base)


# ── feature builders ─────────────────────────────────────────────────────────

def build_sc_features(df, texts, emb, sc_bundle):
    svd_vec   = sc_bundle["svd"].transform(sc_bundle["tfidf"].transform(texts))
    meas      = df[Z_COLS].fillna(0.0).values
    src       = (df[BEAST_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip().str.len() == 0).astype(float).values.reshape(-1, 1)
    brt_len   = df[BR_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip().str.len()
    br_flag   = (brt_len > 0).astype(float).values.reshape(-1, 1)

    # consensus z-score per year
    cons_z = np.zeros(len(df))
    for yr, grp in df.groupby("draft_year"):
        vals = grp["consensus"].dropna()
        if len(vals) < 2: continue
        mu, sigma = vals.mean(), vals.std()
        cons_z[grp.index] = df.loc[grp.index, "consensus"].fillna(mu).map(lambda v: (v - mu) / (sigma + 1e-8))
    cons_z = cons_z.reshape(-1, 1)

    # position one-hot
    pos_grp = df["Position"].str.upper().map(POS_TO_GROUP).fillna("OTHER")
    enc_pos = OneHotEncoder(categories=[POS_ORDER], sparse_output=False, handle_unknown="ignore")
    pos_ohe = enc_pos.fit_transform(pos_grp.values.reshape(-1, 1))

    # beast_rank z-score per year
    rank_z = np.zeros(len(df))
    df2 = df.reset_index(drop=True)
    for yr, grp in df2.groupby("draft_year"):
        vals = grp["beast_rank"].dropna()
        if len(vals) < 2: continue
        mu, sigma = vals.mean(), vals.std()
        rank_z[grp.index] = df2.loc[grp.index, "beast_rank"].fillna(mu).map(lambda v: (v - mu) / (sigma + 1e-8))
    rank_z = rank_z.reshape(-1, 1)

    # text length
    log_lens = np.log1p(texts.str.len().values.astype(float))
    tlen = ((log_lens - log_lens.mean()) / (log_lens.std() + 1e-8)).reshape(-1, 1)

    # round one-hot + imputed flag
    rounds   = df["round"].copy()
    missing  = rounds.isna()
    if missing.any():
        cons = df.loc[missing, "consensus"].fillna(250)
        rounds.loc[missing] = (cons // 32 + 1).clip(upper=7)
    rounds = rounds.fillna(4).astype(int).clip(1, 7)
    enc_rnd = OneHotEncoder(categories=[list(range(1, 8))], sparse_output=False, handle_unknown="ignore")
    rnd_ohe = enc_rnd.fit_transform(rounds.values.reshape(-1, 1))
    rnd_flag = missing.astype(float).values.reshape(-1, 1)

    X_extra = np.hstack([cons_z, br_flag, pos_ohe, rank_z, tlen, rnd_ohe, rnd_flag])
    X       = np.hstack([svd_vec, emb, meas, src, X_extra])
    return X


def build_dv_features(df, texts, emb, dv_bundle):
    svd_vec  = dv_bundle["svd"].transform(dv_bundle["tfidf"].transform(texts))
    meas     = df[Z_COLS].fillna(0.0).values
    src      = np.zeros((len(df), 1))   # draft model trained on Beast only
    brt_len  = df[BR_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip().str.len()
    br_flag  = (brt_len > 0).astype(float).values.reshape(-1, 1)

    cons_z = np.zeros(len(df))
    for yr, grp in df.groupby("draft_year"):
        vals = grp["consensus"].dropna()
        if len(vals) < 2: continue
        mu, sigma = vals.mean(), vals.std()
        cons_z[grp.index] = df.loc[grp.index, "consensus"].fillna(mu).map(lambda v: (v - mu) / (sigma + 1e-8))
    cons_z = cons_z.reshape(-1, 1)

    X_extra = np.hstack([cons_z, br_flag])
    X       = np.hstack([svd_vec, emb, meas, src, X_extra])
    return X


# ── explanation helpers ───────────────────────────────────────────────────────

def word_contribs(tfidf_bundle, coef_vec, tfidf_row):
    """Back-project SVD → vocab, return list of (word, contribution) sorted by |contrib|."""
    word_weights = coef_vec[:150] @ tfidf_bundle["svd"].components_
    vocab        = {v: k for k, v in tfidf_bundle["tfidf"].vocabulary_.items()}
    cx           = tfidf_row.tocoo()
    return sorted([(vocab[j], word_weights[j] * v) for j, v in zip(cx.col, cx.data)],
                  key=lambda x: -abs(x[1]))


def fmt_words(contribs, n, positive=True):
    """Format top n words as 'word(+0.0123)|word2(+0.0098)|...'"""
    filtered = [(w, c) for w, c in contribs if (c > 0) == positive][:n]
    return " | ".join(f"{w}({c:+.4f})" for w, c in filtered)


def sc_feat_contribs(X_row, coef_vec):
    """Named feature contributions for the SC/APY 954d layout."""
    c = coef_vec
    x = X_row
    out = {
        "text_tfidf_svd": float(c[:150]   @ x[:150]),
        "text_embedding": float(c[150:918] @ x[150:918]),
        "consensus":      float(c[930]     * x[930]),
        "br_flag":        float(c[931]     * x[931]),
        "position":       float(c[932:944] @ x[932:944]),
        "beast_rank":     float(c[944]     * x[944]),
        "text_length":    float(c[945]     * x[945]),
        "round":          float(c[946:954] @ x[946:954]),
    }
    # top measurable
    meas_contribs = {MEAS_LABELS[i]: float(c[918+i] * x[918+i]) for i in range(11)}
    top_m = max(meas_contribs, key=lambda k: abs(meas_contribs[k]))
    out["top_measurable"]       = top_m
    out["top_measurable_contrib"] = meas_contribs[top_m]
    return out


def dv_feat_contribs(X_row, coef_vec):
    """Named feature contributions for the DV 932d layout."""
    c = coef_vec
    x = X_row
    out = {
        "text_tfidf_svd": float(c[:150]   @ x[:150]),
        "text_embedding": float(c[150:918] @ x[150:918]),
        "consensus":      float(c[930]     * x[930]),
        "br_flag":        float(c[931]     * x[931]),
    }
    meas_contribs = {MEAS_LABELS[i]: float(c[918+i] * x[918+i]) for i in range(11)}
    top_m = max(meas_contribs, key=lambda k: abs(meas_contribs[k]))
    out["top_measurable"]        = top_m
    out["top_measurable_contrib"] = meas_contribs[top_m]
    return out


# ── main ──────────────────────────────────────────────────────────────────────

print("Loading data...")
df_all = pd.read_csv(BASE / "data/processed/all_prospects.csv")
p26    = df_all[df_all["draft_year"] == 2026].copy().reset_index(drop=True)
has_text = p26["beast_summary"].notna() | p26["pff_overview"].notna()
p26    = p26[has_text].reset_index(drop=True)
print(f"  {len(p26)} prospects with text")

print("Loading models...")
sc_bundle  = joblib.load(BASE / "models/model_sc_tier.pkl")
dv_bundle  = joblib.load(BASE / "models/model_draft_value.pkl")
apy_bundle = joblib.load(BASE / "models/model_apy_pct.pkl")

print("Encoding texts (batch)...")
texts = make_texts(p26)
st    = SentenceTransformer(sc_bundle["st_model"])
emb   = st.encode(texts.tolist(), batch_size=64,
                   show_progress_bar=True, normalize_embeddings=True)

print("Building feature matrices...")
X_sc  = build_sc_features(p26, texts, emb, sc_bundle)
X_dv  = build_dv_features(p26, texts, emb, dv_bundle)
# APY uses same layout as SC
X_apy = X_sc.copy()
X_apy_tfidf = apy_bundle["svd"].transform(apy_bundle["tfidf"].transform(texts))
X_apy[:, :150] = X_apy_tfidf

print("Predicting...")
sc_clf   = sc_bundle["clf"]
sc_le    = sc_bundle["le"]
sc_classes = list(sc_le.classes_)

if sc_bundle.get("ensemble") and sc_bundle.get("clf2") is not None:
    sc_proba = (sc_clf.predict_proba(X_sc) + 2 * sc_bundle["clf2"].predict_proba(X_sc)) / 3
else:
    sc_proba = sc_clf.predict_proba(X_sc)
sc_preds = sc_le.inverse_transform(sc_proba.argmax(axis=1))

dv_clf    = dv_bundle["clf"]
dv_le     = dv_bundle["le"]
dv_classes = list(dv_le.classes_)
dv_proba  = dv_clf.predict_proba(X_dv)
dv_preds  = dv_le.inverse_transform(dv_proba.argmax(axis=1))

apy_clf   = apy_bundle["clf"]
if apy_bundle.get("ensemble") and apy_bundle.get("clf2") is not None:
    apy_preds = np.clip((apy_clf.predict(X_apy) + 2 * apy_bundle["clf2"].predict(X_apy)) / 3, 0, 1)
else:
    apy_preds = np.clip(apy_clf.predict(X_apy), 0, 1)
apy_resid = apy_bundle.get("resid_std", 0.10)

# precompute tfidf rows for word contributions
sc_tfidf_rows  = sc_bundle["tfidf"].transform(texts)
dv_tfidf_rows  = dv_bundle["tfidf"].transform(texts)
apy_tfidf_rows = apy_bundle["tfidf"].transform(texts)

idx_cornerstone   = sc_classes.index("cornerstone")
idx_53man         = sc_classes.index("53_man")
idx_roster_bubble = sc_classes.index("roster_bubble")
idx_out_of_league = sc_classes.index("out_of_league")
idx_slide         = dv_classes.index("slide")
idx_consensus     = dv_classes.index("consensus")
idx_reach         = dv_classes.index("reach")

print("Building explanation rows...")
rows = []
for i in range(len(p26)):
    row = p26.iloc[i]
    sc_pred_cls = sc_preds[i]
    dv_pred_cls = dv_preds[i]

    sc_pred_idx  = sc_classes.index(sc_pred_cls)
    dv_pred_idx  = dv_classes.index(dv_pred_cls)

    # word contributions
    sc_wc  = word_contribs(sc_bundle,  sc_clf.coef_[sc_pred_idx],  sc_tfidf_rows[i])
    dv_wc  = word_contribs(dv_bundle,  dv_clf.coef_[dv_pred_idx],  dv_tfidf_rows[i])
    apy_wc = word_contribs(apy_bundle, apy_clf.coef_,               apy_tfidf_rows[i])

    # feature contributions
    sc_feat  = sc_feat_contribs(X_sc[i],  sc_clf.coef_[sc_pred_idx])
    dv_feat  = dv_feat_contribs(X_dv[i],  dv_clf.coef_[dv_pred_idx])
    apy_feat = sc_feat_contribs(X_apy[i], apy_clf.coef_)

    p_real = float(sc_proba[i, idx_cornerstone] + sc_proba[i, idx_53man])

    rows.append({
        # identity
        "player_name":         row["Player Name"],
        "position":            row["Position"],
        "college":             row.get("College", ""),
        "consensus":           row.get("consensus", ""),
        "text_source":         "pff" if not str(row.get("beast_summary","") or "").strip() else "beast",

        # sc_tier predictions
        "sc_prediction":       sc_pred_cls,
        "p_cornerstone":       round(sc_proba[i, idx_cornerstone] * 100, 1),
        "p_53_man":            round(sc_proba[i, idx_53man]        * 100, 1),
        "p_roster_bubble":     round(sc_proba[i, idx_roster_bubble]* 100, 1),
        "p_out_of_league":     round(sc_proba[i, idx_out_of_league]* 100, 1),

        # sc word explanations
        "sc_top_words":        fmt_words(sc_wc, TOP_N, positive=True),
        "sc_words_against":    fmt_words(sc_wc, BOTTOM_N, positive=False),

        # sc feature contributions
        "sc_feat_consensus":   round(sc_feat["consensus"],           4),
        "sc_feat_round":       round(sc_feat["round"],               4),
        "sc_feat_position":    round(sc_feat["position"],            4),
        "sc_feat_beast_rank":  round(sc_feat["beast_rank"],          4),
        "sc_feat_text_length": round(sc_feat["text_length"],         4),
        "sc_top_measurable":   sc_feat["top_measurable"],
        "sc_top_measurable_contrib": round(sc_feat["top_measurable_contrib"], 4),

        # draft_tier predictions
        "draft_prediction":    dv_pred_cls,
        "p_slide":             round(dv_proba[i, idx_slide]    * 100, 1),
        "p_consensus":         round(dv_proba[i, idx_consensus]* 100, 1),
        "p_reach":             round(dv_proba[i, idx_reach]    * 100, 1),

        # draft word explanations
        "draft_top_words":     fmt_words(dv_wc, TOP_N, positive=True),
        "draft_words_against": fmt_words(dv_wc, BOTTOM_N, positive=False),

        # draft feature contributions
        "draft_feat_consensus": round(dv_feat["consensus"], 4),
        "draft_top_measurable": dv_feat["top_measurable"],
        "draft_top_measurable_contrib": round(dv_feat["top_measurable_contrib"], 4),

        # apy_pct predictions
        "pred_apy_pct":        round(float(apy_preds[i]),                        3),
        "pred_apy_pct_lo":     round(float(np.clip(apy_preds[i] - apy_resid, 0, 1)), 3),
        "pred_apy_pct_hi":     round(float(np.clip(apy_preds[i] + apy_resid, 0, 1)), 3),
        "p_real_contract":     round(p_real * 100, 1),
        "expected_apy_pct":    round(p_real * float(apy_preds[i]), 3),

        # apy word explanations
        "apy_top_words":       fmt_words(apy_wc, TOP_N, positive=True),
        "apy_words_against":   fmt_words(apy_wc, BOTTOM_N, positive=False),

        # apy feature contributions
        "apy_feat_consensus":  round(apy_feat["consensus"],  4),
        "apy_feat_round":      round(apy_feat["round"],      4),
        "apy_feat_beast_rank": round(apy_feat["beast_rank"], 4),
        "apy_top_measurable":  apy_feat["top_measurable"],
        "apy_top_measurable_contrib": round(apy_feat["top_measurable_contrib"], 4),
    })

out = pd.DataFrame(rows)
out_path = BASE / "data/processed/explain_2026.csv"
out.to_csv(out_path, index=False)
print(f"\nWrote {len(out)} rows → {out_path}")
print(out[["player_name","sc_prediction","draft_prediction","pred_apy_pct","sc_top_words"]].head(10).to_string(index=False))
