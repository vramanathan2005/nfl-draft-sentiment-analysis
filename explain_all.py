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

from features import (
    BEAST_COLS, PFF_COLS, BR_COLS, Z_COLS,
    POS_TO_GROUP, POS_ORDER, MEAS_LABELS,
    unified_text, text_source_flag, has_br_flag, get_meas,
    consensus_feature, beast_rank_feature, text_length_feature,
    round_feature, position_feature,
    deduplicate_ngrams, find_source_sentences, _STOPWORD_UNIGRAMS, _STOPWORD_BIGRAMS,
)

warnings.filterwarnings("ignore")

BASE = Path("/Users/varunramanathan/Downloads/sentiment-analysis")


# ── feature builders ─────────────────────────────────────────────────────────

def build_sc_features(df, texts, emb, sc_bundle):
    svd_vec = sc_bundle["svd"].transform(sc_bundle["tfidf"].transform(texts))
    X_extra = np.hstack([consensus_feature(df), has_br_flag(df), position_feature(df),
                         beast_rank_feature(df), text_length_feature(texts), round_feature(df)])
    return np.hstack([svd_vec, emb, get_meas(df), text_source_flag(df), X_extra])


def build_dv_features(df, texts, emb, dv_bundle):
    svd_vec = dv_bundle["svd"].transform(dv_bundle["tfidf"].transform(texts))
    X_extra = has_br_flag(df)  # consensus removed: DV is text/measurables-driven
    return np.hstack([svd_vec, emb, get_meas(df), np.zeros((len(df), 1)), X_extra])


# ── explanation helpers ───────────────────────────────────────────────────────

def word_contribs(tfidf_bundle, coef_vec, tfidf_row):
    """Back-project SVD → vocab, return list of (word, contribution) sorted by |contrib|."""
    word_weights = coef_vec[:150] @ tfidf_bundle["svd"].components_
    vocab        = {v: k for k, v in tfidf_bundle["tfidf"].vocabulary_.items()}
    cx           = tfidf_row.tocoo()
    return sorted([(vocab[j], word_weights[j] * v) for j, v in zip(cx.col, cx.data)],
                  key=lambda x: -abs(x[1]))


def top_words_str(wc, positive=True, n=8):
    """Top N words by contribution for given sign. Returns pipe-separated 'word:score'."""
    def _ok(w):
        tokens = w.lower().split()
        if len(tokens) == 1 and tokens[0] in _STOPWORD_UNIGRAMS:
            return False
        if w.lower() in _STOPWORD_BIGRAMS:
            return False
        return True
    filtered = [(w, c) for w, c in wc if (c > 0) == positive and abs(c) >= 0.001 and _ok(w)][:n]
    if not filtered:
        filtered = [(w, c) for w, c in wc if (c > 0) == positive and _ok(w)][:n]
    return "|".join(f"{w}:{c:.4f}" for w, c in filtered)


def fmt_sentences(sent_list):
    """Convert find_source_sentences output to CSV-friendly string."""
    parts = []
    for sent, phrases in sent_list:
        tagged = ", ".join(ph for ph, _ in phrases)
        parts.append(f'"{sent}" [{tagged}]' if tagged else f'"{sent}"')
    return " || ".join(parts)


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
    """Named feature contributions for the DV 931d layout (no consensus)."""
    c = coef_vec
    x = X_row
    out = {
        "text_tfidf_svd": float(c[:150]   @ x[:150]),
        "text_embedding": float(c[150:918] @ x[150:918]),
        "br_flag":        float(c[930]     * x[930]),
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
texts = unified_text(p26)
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
sc_tfidf_rows = sc_bundle["tfidf"].transform(texts)
dv_tfidf_rows = dv_bundle["tfidf"].transform(texts)

idx_cornerstone   = sc_classes.index("cornerstone")
idx_53man         = sc_classes.index("53_man")
idx_roster_bubble = sc_classes.index("roster_bubble")
idx_out_of_league = sc_classes.index("out_of_league")
idx_slide         = dv_classes.index("slide")
idx_consensus     = dv_classes.index("consensus")
idx_reach         = dv_classes.index("riser")

print("Building explanation rows...")
rows = []
for i in range(len(p26)):
    row = p26.iloc[i]
    sc_pred_cls = sc_preds[i]
    dv_pred_cls = dv_preds[i]

    sc_pred_idx  = sc_classes.index(sc_pred_cls)
    dv_pred_idx  = dv_classes.index(dv_pred_cls)

    # word contributions (deduplicated n-grams)
    sc_wc = deduplicate_ngrams(word_contribs(sc_bundle, sc_clf.coef_[sc_pred_idx], sc_tfidf_rows[i]))
    dv_wc = deduplicate_ngrams(word_contribs(dv_bundle, dv_clf.coef_[dv_pred_idx], dv_tfidf_rows[i]))

    # feature contributions
    sc_feat  = sc_feat_contribs(X_sc[i],  sc_clf.coef_[sc_pred_idx])
    dv_feat  = dv_feat_contribs(X_dv[i],  dv_clf.coef_[dv_pred_idx])
    apy_feat = sc_feat_contribs(X_apy[i], apy_clf.coef_)

    p_real   = float(sc_proba[i, idx_cornerstone] + sc_proba[i, idx_53man])
    text_str = texts.iloc[i]

    sc_pos_sents = find_source_sentences(text_str, sc_wc, n=3, positive=True)
    sc_neg_sents = find_source_sentences(text_str, sc_wc, n=2, positive=False)
    dv_pos_sents = find_source_sentences(text_str, dv_wc, n=3, positive=True)
    dv_neg_sents = find_source_sentences(text_str, dv_wc, n=2, positive=False)

    rows.append({
        # identity
        "player_name":   row["Player Name"],
        "position":      row["Position"],
        "college":       row.get("College", ""),
        "consensus":     row.get("consensus", ""),
        "text_source":   "pff" if not str(row.get("beast_summary","") or "").strip() else "beast",

        # sc_tier predictions
        "sc_prediction":   sc_pred_cls,
        "p_cornerstone":   round(sc_proba[i, idx_cornerstone]  * 100, 1),
        "p_53_man":        round(sc_proba[i, idx_53man]         * 100, 1),
        "p_roster_bubble": round(sc_proba[i, idx_roster_bubble] * 100, 1),
        "p_out_of_league": round(sc_proba[i, idx_out_of_league] * 100, 1),
        "sc_key_sentences":     fmt_sentences(sc_pos_sents),
        "sc_concerns":          fmt_sentences(sc_neg_sents),

        # sc structural features
        "sc_feat_consensus":  round(sc_feat["consensus"],  4),
        "sc_feat_round":      round(sc_feat["round"],      4),
        "sc_feat_position":   round(sc_feat["position"],   4),
        "sc_feat_beast_rank": round(sc_feat["beast_rank"], 4),
        "sc_top_measurable":  sc_feat["top_measurable"],
        "sc_top_measurable_contrib": round(sc_feat["top_measurable_contrib"], 4),

        # draft_tier predictions
        "draft_prediction": dv_pred_cls,
        "p_slide":          round(dv_proba[i, idx_slide]     * 100, 1),
        "p_consensus":      round(dv_proba[i, idx_consensus]  * 100, 1),
        "p_reach":          round(dv_proba[i, idx_reach]      * 100, 1),
        "draft_key_sentences":  fmt_sentences(dv_pos_sents),
        "draft_concerns":       fmt_sentences(dv_neg_sents),

        # draft structural features
        "draft_feat_br":        round(dv_feat["br_flag"], 4),
        "draft_top_measurable": dv_feat["top_measurable"],
        "draft_top_measurable_contrib": round(dv_feat["top_measurable_contrib"], 4),

        # apy_pct predictions
        "pred_apy_pct":     round(float(apy_preds[i]),                             3),
        "pred_apy_pct_lo":  round(float(np.clip(apy_preds[i] - apy_resid, 0, 1)),  3),
        "pred_apy_pct_hi":  round(float(np.clip(apy_preds[i] + apy_resid, 0, 1)),  3),
        "p_real_contract":  round(p_real * 100, 1),
        "expected_apy_pct": round(p_real * float(apy_preds[i]), 3),

        # apy structural features
        "apy_feat_consensus":  round(apy_feat["consensus"],  4),
        "apy_feat_round":      round(apy_feat["round"],      4),
        "apy_feat_beast_rank": round(apy_feat["beast_rank"], 4),
        "apy_top_measurable":  apy_feat["top_measurable"],
        "apy_top_measurable_contrib": round(apy_feat["top_measurable_contrib"], 4),

        # word contributions (top words by sign)
        "sc_pos_words":    top_words_str(sc_wc, positive=True,  n=10),
        "sc_neg_words":    top_words_str(sc_wc, positive=False, n=6),
        "draft_pos_words": top_words_str(dv_wc, positive=True,  n=10),
        "draft_neg_words": top_words_str(dv_wc, positive=False, n=6),
    })

out = pd.DataFrame(rows)
out_path = BASE / "data/processed/explain_2026.csv"
out.to_csv(out_path, index=False)
print(f"\nWrote {len(out)} rows → {out_path}")
print(out[["player_name","sc_prediction","draft_prediction","pred_apy_pct","sc_key_sentences"]].head(5).to_string(index=False))
