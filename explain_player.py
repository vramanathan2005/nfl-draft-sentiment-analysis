"""
Explain a player's model predictions.

Usage:
    python explain_player.py "Sonny Styles"
    python explain_player.py "Jeremiyah Love" --model apy
    python explain_player.py "Spencer Fano" --model all

Output per player:
  1. Prediction probabilities
  2. Scouting sentences most responsible for the prediction
  3. Structural drivers: consensus rank, round, beast rank, measurables vs position peers
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder

from features import (
    BEAST_COLS, BR_COLS, Z_COLS,
    POS_TO_GROUP, POS_ORDER,
    unified_text_row, deduplicate_ngrams, find_source_sentences,
)

warnings.filterwarnings("ignore")

BASE = Path("/Users/varunramanathan/Downloads/sentiment-analysis")

RAW_MEAS_COLS  = ["ht_in","wt_lbs","arm_in","hand_in","wing_in",
                   "dash40","vj_in","bj_in","shuttle","cone3","bench"]
MEAS_LABELS    = ["height","weight","arm","hand","wingspan",
                  "40yd","vert jump","broad jump","shuttle","3-cone","bench"]


# ── feature building ──────────────────────────────────────────────────────────

def build_features(row, df_all, bundle):
    text = unified_text_row(row)

    tfidf_vec = bundle["tfidf"].transform([text])
    svd_vec   = bundle["svd"].transform(tfidf_vec)
    emb       = SentenceTransformer(bundle["st_model"]).encode(
                    [text], normalize_embeddings=True)

    meas = np.array([0.0 if pd.isna(row.get(c)) else float(row.get(c))
                     for c in Z_COLS]).reshape(1, -1)

    bt      = " ".join(str(row.get(c, "") or "") for c in BEAST_COLS).strip()
    brt     = " ".join(str(row.get(c, "") or "") for c in BR_COLS).strip()
    src     = np.array([[0.0 if bt else 1.0]])
    br_flag = np.array([[1.0 if brt else 0.0]])

    year    = int(row["draft_year"])
    yr_cons = df_all[df_all["draft_year"] == year]["consensus"].dropna()
    cons_mu = yr_cons.mean() if len(yr_cons) else 250.0
    cons_sg = yr_cons.std()  if len(yr_cons) > 1 else 1.0
    cons_val = float(row.get("consensus") or cons_mu)
    cons_z   = np.array([[(cons_val - cons_mu) / (cons_sg + 1e-8)]])

    pos_grp = POS_TO_GROUP.get(str(row.get("Position", "")).upper(), "OTHER")
    enc_pos = OneHotEncoder(categories=[POS_ORDER], sparse_output=False, handle_unknown="ignore")
    pos_ohe = enc_pos.fit_transform([[pos_grp]])

    yr_rank = df_all[df_all["draft_year"] == year]["beast_rank"].dropna()
    if len(yr_rank) > 1:
        rk_mu, rk_sg = yr_rank.mean(), yr_rank.std()
        rk_val = float(row.get("beast_rank") or rk_mu)
        rk_z   = np.array([[(rk_val - rk_mu) / (rk_sg + 1e-8)]])
    else:
        rk_z = np.array([[0.0]])

    tlen = np.array([[(np.log1p(len(text)) - 7.0) / 1.0]])

    raw_round = row.get("round")
    imputed   = pd.isna(raw_round) or raw_round is None
    if imputed:
        rnd_val = int(min(int(float(row.get("consensus") or 250) // 32) + 1, 7))
    else:
        rnd_val = int(min(int(raw_round), 7))
    enc_rnd = OneHotEncoder(categories=[list(range(1, 8))], sparse_output=False, handle_unknown="ignore")
    rnd_ohe = enc_rnd.fit_transform([[rnd_val]])
    rnd_flag = np.array([[1.0 if imputed else 0.0]])

    X_extra = np.hstack([cons_z, br_flag, pos_ohe, rk_z, tlen, rnd_ohe, rnd_flag])
    X       = np.hstack([svd_vec, emb, meas, src, X_extra])

    meta = {
        "text":           text,
        "tfidf_vec":      tfidf_vec,
        "pos_grp":        pos_grp,
        "src":            "pff" if src[0, 0] else "beast",
        "round":          rnd_val,
        "round_imputed":  bool(imputed),
        "consensus_val":  None if pd.isna(row.get("consensus")) else int(row["consensus"]),
        "beast_rank_val": None if pd.isna(row.get("beast_rank")) else int(row["beast_rank"]),
        # z-scores for each measurable (for display)
        "meas_z": {MEAS_LABELS[i]: float(meas[0, i]) for i in range(11)},
    }
    return X, meta


# ── contribution math ─────────────────────────────────────────────────────────

def word_contributions(meta, bundle, coef_vec):
    word_weights = coef_vec[:150] @ bundle["svd"].components_
    vocab        = {v: k for k, v in bundle["tfidf"].vocabulary_.items()}
    cx           = meta["tfidf_vec"].tocoo()
    contribs     = sorted(
        [(vocab[j], word_weights[j] * v) for j, v in zip(cx.col, cx.data)],
        key=lambda x: -abs(x[1])
    )
    return deduplicate_ngrams(contribs)


def feature_contributions(X, coef_vec):
    x, c = X[0], coef_vec
    return {
        "consensus":  float(c[930] * x[930]),
        "round":      float(c[946:954] @ x[946:954]),
        "beast_rank": float(c[944] * x[944]),
        "position":   float(c[932:944] @ x[932:944]),
        "measurables": {MEAS_LABELS[i]: float(c[918+i] * x[918+i]) for i in range(11)},
    }


# ── display helpers ───────────────────────────────────────────────────────────

def _arrow(v):
    if   v >  0.3: return "↑↑"
    elif v >  0.05: return "↑"
    elif v > -0.05: return ""
    elif v > -0.3:  return "↓"
    else:           return "↓↓"


def _fmt_meas(label, raw_val):
    if pd.isna(raw_val): return None
    v = float(raw_val)
    if label == "height":
        ft, inch = int(v // 12), v % 12
        return f"{ft}'{inch:.1f}\""
    if label == "weight":    return f"{v:.0f} lbs"
    if label in ("40yd","shuttle","3-cone"): return f"{v:.2f}s"
    if label == "bench":     return f"{v:.0f} reps"
    return f"{v:.1f}\""


def show_structural_drivers(X, coef_vec, meta, row, label, show_arrows=True):
    feat = feature_contributions(X, coef_vec)

    print(f"\n  Structural drivers → {label}:")
    if meta["consensus_val"] is not None:
        arr = f"  {_arrow(feat['consensus'])}" if show_arrows else ""
        print(f"    {'consensus rank':<18} #{meta['consensus_val']:<6}{arr}")
    rnd_str = f"round {meta['round']}{'*' if meta['round_imputed'] else ''}"
    arr = f"  {_arrow(feat['round'])}" if show_arrows else ""
    print(f"    {'draft round':<18} {rnd_str:<8}{arr}"
          + ("  (* projected from consensus)" if meta['round_imputed'] else ""))
    if meta["beast_rank_val"] is not None:
        arr = f"  {_arrow(feat['beast_rank'])}" if show_arrows else ""
        print(f"    {'beast rank':<18} #{meta['beast_rank_val']:<6}{arr}")
    arr = f"  {_arrow(feat['position'])}" if show_arrows else ""
    print(f"    {'position group':<18} {meta['pos_grp']:<8}{arr}")

    # measurables: show value + z-score only — arrows omitted because model weight
    # direction can be counterintuitive (e.g. longer arms ↓ for LBs) and misleads readers
    meas_c = feat["measurables"]
    notable = {k: v for k, v in meas_c.items() if abs(v) > 0.02}
    if notable:
        print(f"\n  Measurables vs {meta['pos_grp']} peers:")
        raw_map = dict(zip(MEAS_LABELS, RAW_MEAS_COLS))
        for k in sorted(notable, key=lambda x: -abs(notable[x])):
            z    = meta["meas_z"][k]
            rval = _fmt_meas(k, row.get(raw_map[k]))
            if rval is None:
                continue
            above = "above avg" if z >= 0 else "below avg"
            print(f"    {k:<15} {rval:<12} {z:+.1f}σ {above}")


# ── model explain functions ───────────────────────────────────────────────────

def explain_sc(player_row, df_all, bundle, verbose=True):
    X, meta = build_features(player_row, df_all, bundle)
    clf     = bundle["clf"]
    le      = bundle["le"]

    proba = (clf.predict_proba(X) + 2 * bundle["clf2"].predict_proba(X)) / 3 \
            if bundle.get("ensemble") and bundle.get("clf2") is not None \
            else clf.predict_proba(X)

    classes  = list(le.classes_)
    pred_idx = proba[0].argmax()
    pred_cls = classes[pred_idx]
    coef     = clf.coef_[pred_idx]

    if not verbose:
        return pred_cls, proba[0], meta

    print(f"\n{'─'*65}")
    print(f"  Contract tier prediction")
    print(f"{'─'*65}")
    print(f"  Prediction : {pred_cls}  ({proba[0, pred_idx]*100:.1f}%)")
    for i, cls in enumerate(classes):
        print(f"  {cls:<14} {proba[0,i]*100:>5.1f}%  {'█' * int(proba[0,i] * 40)}")

    contribs       = word_contributions(meta, bundle, coef)
    pos_sentences  = find_source_sentences(meta["text"], contribs, n=3, positive=True)
    neg_sentences  = find_source_sentences(meta["text"], contribs, n=2, positive=False)

    if pos_sentences:
        print(f"\n  Key scouting language → {pred_cls}:")
        for sent, phrases in pos_sentences:
            print(f"    \"{sent}\"")
            for ph, _ in phrases:
                print(f"      → {ph}")

    if neg_sentences:
        print(f"\n  Concerns / risk factors:")
        for sent, phrases in neg_sentences:
            print(f"    \"{sent}\"")
            for ph, _ in phrases:
                print(f"      → {ph}")

    show_structural_drivers(X, coef, meta, player_row, pred_cls)
    print(f"\n  Text source: {meta['src']}")


def explain_apy(player_row, df_all, bundle, verbose=True):
    X, meta = build_features(player_row, df_all, bundle)
    clf     = bundle["clf"]

    pred = float(np.clip(
        (clf.predict(X) + 2 * bundle["clf2"].predict(X)) / 3
        if bundle.get("ensemble") and bundle.get("clf2") is not None
        else clf.predict(X), 0, 1)[0])

    resid = bundle.get("resid_std", 0.10)

    if not verbose:
        return pred, meta

    print(f"\n{'─'*65}")
    print(f"  APY percentile prediction")
    print(f"{'─'*65}")
    print(f"  Predicted APY percentile : {pred:.3f}  [{max(0,pred-resid):.3f} – {min(1,pred+resid):.3f}]")
    print(f"  (1.0 = highest-paid player in NFL on signing date)")

    show_structural_drivers(X, clf.coef_, meta, player_row, "APY percentile", show_arrows=False)
    print(f"\n  Text source: {meta['src']}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("player", help="Player name (partial match ok)")
    parser.add_argument("--model", default="all", choices=["sc", "apy", "all"])
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    df   = pd.read_csv(BASE / "data/processed/all_prospects.csv")
    mask = (df["Player Name"].str.lower().str.contains(args.player.lower()) &
            (df["draft_year"] == args.year))
    hits = df[mask]

    if len(hits) == 0:
        hits = df[df["Player Name"].str.lower().str.contains(args.player.lower())]
        if len(hits) == 0:
            print(f"No player matching '{args.player}' found.")
            sys.exit(1)

    if len(hits) > 1:
        print("Multiple matches — using first:")
        for _, r in hits.iterrows():
            print(f"  {r['Player Name']}  ({r['Position']}, {int(r['draft_year'])})")
        print()

    row = hits.iloc[0]
    print(f"\n{'═'*65}")
    print(f"  {row['Player Name']}  |  {row['Position']}  |  {row.get('College','?')}")
    print(f"  Consensus: #{int(row['consensus']) if pd.notna(row.get('consensus')) else '?'}"
          f"   Draft year: {int(row['draft_year'])}")
    print(f"{'═'*65}")

    sc_bundle  = joblib.load(BASE / "models/model_sc_tier.pkl")
    apy_bundle = joblib.load(BASE / "models/model_apy_pct.pkl")

    if args.model in ("sc", "all"):
        explain_sc(row, df, sc_bundle)
    if args.model in ("apy", "all"):
        explain_apy(row, df, apy_bundle)

    print()


if __name__ == "__main__":
    main()
