"""
Explain a player's model predictions.

Usage:
    python explain_player.py "Sonny Styles"
    python explain_player.py "Jeremiyah Love" --model apy
    python explain_player.py "Spencer Fano" --model all

For linear models (LogReg / Ridge) contributions are exact:
    word_contribution(w, class) = (LR.coef_[class] @ SVD.components_)[:vocab] × tfidf(w, player)
    feature_contribution(k, class) = LR.coef_[class, k] × x[k]

Sections shown per player:
  1. Prediction summary
  2. Top words pushing toward / away from the predicted class
  3. Structural feature contributions (consensus, round, position, measurables, etc.)
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

warnings.filterwarnings("ignore")

BASE = Path("/Users/varunramanathan/Downloads/sentiment-analysis")

# ── feature helpers (must match stacked_models.py / inference_2026.py) ───────

BEAST_COLS = ["beast_summary", "beast_strengths", "beast_weaknesses"]
PFF_COLS   = ["pff_overview", "pff_pros", "pff_cons", "pff_bottom_line", "pff_extra"]
BR_COLS    = ["br_positives", "br_negatives"]

POS_GROUPS = {
    "QB":["QB"], "RB":["RB","FB"], "WR":["WR"], "TE":["TE"],
    "OL":["OT","IOL","C","G","T"], "EDGE":["EDGE","OLB","DE"],
    "DL":["DL","DT","NT"], "LB":["LB","ILB","MLB"],
    "CB":["CB"], "S":["S","FS","SS"], "SPEC":["K","P","LS"],
}
POS_TO_GROUP = {p: g for g, ps in POS_GROUPS.items() for p in ps}
POS_ORDER    = sorted(POS_GROUPS.keys()) + ["OTHER"]
Z_COLS       = ["z_ht_in","z_wt_lbs","z_arm_in","z_hand_in","z_wing_in",
                "z_dash40","z_vj_in","z_bj_in","z_shuttle","z_cone3","z_bench"]
MEAS_LABELS  = ["height","weight","arm","hand","wingspan",
                "40yd","vert jump","broad jump","shuttle","3-cone","bench"]

def unified_text(row):
    bt  = " ".join(str(row.get(c, "") or "") for c in BEAST_COLS).strip()
    pt  = " ".join(str(row.get(c, "") or "") for c in PFF_COLS).strip()
    brt = " ".join(str(row.get(c, "") or "") for c in BR_COLS).strip()
    base = bt if bt else pt
    return (base + " " + brt).strip() if brt else base

def build_features(row, df_all, bundle):
    """Build the full feature vector for a single player row (as a dict/Series)."""
    text = unified_text(row)

    # TF-IDF + SVD
    tfidf_vec = bundle["tfidf"].transform([text])
    svd_vec   = bundle["svd"].transform(tfidf_vec)           # (1, 150)

    # Sentence embedding
    st_model = SentenceTransformer(bundle["st_model"])
    emb      = st_model.encode([text], normalize_embeddings=True)  # (1, 768)

    # Measurables (0-impute NaN, matching stacked_models behavior)
    meas = np.array([0.0 if pd.isna(row.get(c)) else float(row.get(c))
                     for c in Z_COLS]).reshape(1, -1)

    # Source flag (0=beast, 1=pff)
    bt = " ".join(str(row.get(c, "") or "") for c in BEAST_COLS).strip()
    src = np.array([[0.0 if bt else 1.0]])

    # Consensus z-score within year
    year = int(row["draft_year"])
    yr_rows = df_all[df_all["draft_year"] == year]["consensus"].dropna()
    cons_mu, cons_sigma = yr_rows.mean(), yr_rows.std() if len(yr_rows) > 1 else 1.0
    cons_val = float(row.get("consensus") or cons_mu)
    cons_z   = np.array([[(cons_val - cons_mu) / (cons_sigma + 1e-8)]])

    # BR flag
    brt = " ".join(str(row.get(c, "") or "") for c in BR_COLS).strip()
    br_flag = np.array([[1.0 if brt else 0.0]])

    # Position one-hot
    pos_grp = POS_TO_GROUP.get(str(row.get("Position", "")).upper(), "OTHER")
    enc_pos = OneHotEncoder(categories=[POS_ORDER], sparse_output=False, handle_unknown="ignore")
    pos_ohe = enc_pos.fit_transform([[pos_grp]])  # (1, 12)

    # Beast rank z-score within year
    yr_rank = df_all[df_all["draft_year"] == year]["beast_rank"].dropna()
    if len(yr_rank) > 1:
        rk_mu, rk_sigma = yr_rank.mean(), yr_rank.std()
        rk_val = float(row.get("beast_rank") or rk_mu)
        rk_z   = np.array([[(rk_val - rk_mu) / (rk_sigma + 1e-8)]])
    else:
        rk_z = np.array([[0.0]])

    # Text length
    tlen = np.array([[(np.log1p(len(text)) - 7.0) / 1.0]])  # rough standardization

    # Round one-hot + imputed flag
    raw_round = row.get("round")
    imputed   = 1.0 if (pd.isna(raw_round) or raw_round is None) else 0.0
    if imputed:
        cons_for_round = float(row.get("consensus") or 250)
        rnd_val = int(min(int(cons_for_round // 32) + 1, 7))
    else:
        rnd_val = int(min(int(raw_round), 7))
    enc_rnd  = OneHotEncoder(categories=[list(range(1, 8))], sparse_output=False, handle_unknown="ignore")
    rnd_ohe  = enc_rnd.fit_transform([[rnd_val]])  # (1, 7)
    rnd_flag = np.array([[imputed]])

    X_extra = np.hstack([cons_z, br_flag, pos_ohe, rk_z, tlen, rnd_ohe, rnd_flag])
    X       = np.hstack([svd_vec, emb, meas, src, X_extra])

    meta = {
        "text": text, "tfidf_vec": tfidf_vec, "svd_vec": svd_vec,
        "cons_z": float(cons_z[0, 0]), "br_flag": float(br_flag[0, 0]),
        "pos_grp": pos_grp, "rk_z": float(rk_z[0, 0]),
        "tlen": float(np.log1p(len(text))),
        "round": rnd_val, "round_imputed": bool(imputed),
        "src": "pff" if src[0, 0] else "beast",
    }
    return X, meta


def word_contributions(meta, bundle, coef_vec):
    """
    Back-project from SVD → vocab space.
    contribution(word) = (coef_vec[:150] @ SVD.components_)[vocab_idx] × tfidf_weight
    Returns sorted list of (word, contribution) for words present in the player's text.
    """
    # Effective word weights for this coefficient vector
    word_weights = coef_vec[:150] @ bundle["svd"].components_  # shape (vocab_size,)

    # Player's raw TF-IDF weights (sparse row → dict of nonzero terms)
    tfidf_row  = meta["tfidf_vec"]
    vocab      = {v: k for k, v in bundle["tfidf"].vocabulary_.items()}
    cx         = tfidf_row.tocoo()
    contribs   = [(vocab[j], word_weights[j] * v) for j, v in zip(cx.col, cx.data)]
    return sorted(contribs, key=lambda x: -abs(x[1]))


def feature_contributions(X, bundle_coef, meta):
    """
    Split the feature vector into named groups and sum contributions.
    Returns dict: group_name → scalar contribution.
    """
    x = X[0]
    c = bundle_coef

    groups = {}
    groups["text (TF-IDF+SVD)"]  = float(c[:150] @ x[:150])
    groups["text (embedding)"]   = float(c[150:918] @ x[150:918])
    groups["measurables"]        = {MEAS_LABELS[i]: float(c[918+i] * x[918+i])
                                    for i in range(11)}
    groups["source_flag"]        = float(c[929] * x[929])
    # extra block starts at 930
    groups["consensus"]          = float(c[930] * x[930])
    groups["BR_flag"]            = float(c[931] * x[931])
    pos_contrib = float(c[932:944] @ x[932:944])
    groups["position"]           = pos_contrib
    groups["beast_rank"]         = float(c[944] * x[944])
    groups["text_length"]        = float(c[945] * x[945])
    groups["round"]              = float(c[946:954] @ x[946:954])
    return groups


def fmt_pct(v, decimals=1):
    return f"{v*100:.{decimals}f}%"

def bar(v, width=20):
    filled = int(abs(v) / 0.05 * width) if abs(v) < 0.05 * width else width
    filled = min(filled, width)
    sign   = "+" if v >= 0 else "-"
    return f"[{sign * filled}{' ' * (width - filled)}]"


# ── main ──────────────────────────────────────────────────────────────────────

def explain_sc(player_row, df_all, bundle, verbose=True):
    X, meta = build_features(player_row, df_all, bundle)
    clf     = bundle["clf"]
    le      = bundle["le"]

    if bundle.get("ensemble") and bundle.get("clf2") is not None:
        proba = (clf.predict_proba(X) + 2 * bundle["clf2"].predict_proba(X)) / 3
    else:
        proba = clf.predict_proba(X)

    classes   = list(le.classes_)
    pred_idx  = proba[0].argmax()
    pred_cls  = classes[pred_idx]
    coef_pred = clf.coef_[pred_idx]   # weights for predicted class

    if not verbose:
        return pred_cls, proba[0], meta

    print(f"\n{'─'*65}")
    print(f"  sc_tier prediction")
    print(f"{'─'*65}")
    print(f"  Prediction : {pred_cls}  ({fmt_pct(proba[0, pred_idx])})")
    for i, cls in enumerate(classes):
        bar_str = "█" * int(proba[0, i] * 40)
        print(f"  {cls:<12} {fmt_pct(proba[0,i]):>6}  {bar_str}")

    # Word contributions toward predicted class
    contribs = word_contributions(meta, bundle, coef_pred)
    print(f"\n  Top words → {pred_cls}:")
    pos_words = [(w, c) for w, c in contribs if c > 0][:12]
    neg_words = [(w, c) for w, c in contribs if c < 0][:8]
    for w, c in pos_words:
        print(f"    {'+' if c>0 else ''}{c:+.4f}  {w}")
    if neg_words:
        print(f"\n  Words pulling AWAY from {pred_cls}:")
        for w, c in neg_words:
            print(f"    {c:+.4f}  {w}")

    # Structural features
    feat = feature_contributions(X, coef_pred, meta)
    print(f"\n  Structural feature contributions → {pred_cls}:")
    scalar_feats = {k: v for k, v in feat.items() if isinstance(v, float)}
    for name, val in sorted(scalar_feats.items(), key=lambda x: -abs(x[1])):
        if name.startswith("text"):
            continue
        print(f"    {name:<20} {val:+.4f}")

    meas = feat["measurables"]
    notable_meas = {k: v for k, v in meas.items() if abs(v) > 0.0005}
    if notable_meas:
        print(f"\n  Notable measurables → {pred_cls}:")
        for k, v in sorted(notable_meas.items(), key=lambda x: -abs(x[1])):
            print(f"    {k:<15} {v:+.4f}")

    print(f"\n  Text source: {meta['src']} | Position group: {meta['pos_grp']} | "
          f"Round: {meta['round']}{'*' if meta['round_imputed'] else ''}")


def explain_apy(player_row, df_all, bundle, verbose=True):
    X, meta = build_features(player_row, df_all, bundle)
    clf     = bundle["clf"]

    if bundle.get("ensemble") and bundle.get("clf2") is not None:
        pred = float(np.clip((clf.predict(X) + 2 * bundle["clf2"].predict(X)) / 3, 0, 1)[0])
    else:
        pred = float(np.clip(clf.predict(X), 0, 1)[0])

    resid = bundle.get("resid_std", 0.10)
    coef  = clf.coef_   # shape (n_features,) for Ridge

    if not verbose:
        return pred, meta

    print(f"\n{'─'*65}")
    print(f"  apy_pct prediction")
    print(f"{'─'*65}")
    print(f"  Predicted APY percentile: {pred:.3f}  [{max(0,pred-resid):.3f} – {min(1,pred+resid):.3f}]")
    print(f"  (1.0 = highest-paid active player on signing date)")

    contribs = word_contributions(meta, bundle, coef)
    print(f"\n  Top words → higher APY percentile:")
    pos_words = [(w, c) for w, c in contribs if c > 0][:12]
    neg_words = [(w, c) for w, c in contribs if c < 0][:8]
    for w, c in pos_words:
        print(f"    {c:+.4f}  {w}")
    if neg_words:
        print(f"\n  Words pulling APY percentile DOWN:")
        for w, c in neg_words:
            print(f"    {c:+.4f}  {w}")

    feat = feature_contributions(X, coef, meta)
    print(f"\n  Structural feature contributions → APY percentile:")
    scalar_feats = {k: v for k, v in feat.items() if isinstance(v, float)}
    for name, val in sorted(scalar_feats.items(), key=lambda x: -abs(x[1])):
        if name.startswith("text"):
            continue
        print(f"    {name:<20} {val:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("player", help="Player name (partial match ok)")
    parser.add_argument("--model", default="all", choices=["sc", "apy", "all"],
                        help="Which model to explain (default: all)")
    parser.add_argument("--year", type=int, default=2026,
                        help="Draft year to filter on (default: 2026)")
    args = parser.parse_args()

    df   = pd.read_csv(BASE / "data/processed/all_prospects.csv")
    mask = (df["Player Name"].str.lower().str.contains(args.player.lower()) &
            (df["draft_year"] == args.year))
    hits = df[mask]

    if len(hits) == 0:
        # Broaden search across all years
        mask2 = df["Player Name"].str.lower().str.contains(args.player.lower())
        hits  = df[mask2]
        if len(hits) == 0:
            print(f"No player matching '{args.player}' found.")
            sys.exit(1)

    if len(hits) > 1:
        print(f"Multiple matches — using first:")
        for _, r in hits.iterrows():
            print(f"  {r['Player Name']}  ({r['Position']}, {int(r['draft_year'])})")
        print()

    row = hits.iloc[0]
    print(f"\n{'═'*65}")
    print(f"  Player : {row['Player Name']}")
    print(f"  Pos    : {row['Position']}   Draft year: {int(row['draft_year'])}")
    print(f"  Cons.  : {row.get('consensus', 'N/A')}   College: {row.get('College', 'N/A')}")
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
