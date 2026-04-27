"""
Shared feature engineering for the NFL draft NLP pipeline.
Imported by stacked_models.py, inference_2026.py, explain_player.py, explain_all.py.
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# ── Column constants ──────────────────────────────────────────────────────────

BEAST_COLS  = ["beast_summary", "beast_strengths", "beast_weaknesses"]
PFF_COLS    = ["pff_overview", "pff_pros", "pff_cons", "pff_bottom_line", "pff_extra"]
BR_COLS     = ["br_positives", "br_negatives"]
Z_COLS      = ["z_ht_in","z_wt_lbs","z_arm_in","z_hand_in","z_wing_in",
               "z_dash40","z_vj_in","z_bj_in","z_shuttle","z_cone3","z_bench"]
MEAS_LABELS = ["height","weight","arm","hand","wingspan",
               "40yd","vert_jump","broad_jump","shuttle","3cone","bench"]

# ── Position groups ───────────────────────────────────────────────────────────

POS_GROUPS = {
    "QB":["QB"], "RB":["RB","FB"], "WR":["WR"], "TE":["TE"],
    "OL":["OT","IOL","C","G","T"], "EDGE":["EDGE","OLB","DE"],
    "DL":["DL","DT","NT"], "LB":["LB","ILB","MLB"],
    "CB":["CB"], "S":["S","FS","SS"], "SPEC":["K","P","LS"],
}
POS_TO_GROUP = {p: g for g, ps in POS_GROUPS.items() for p in ps}
POS_ORDER    = sorted(POS_GROUPS.keys()) + ["OTHER"]

# ── Text assembly ─────────────────────────────────────────────────────────────

def unified_text(df: pd.DataFrame) -> pd.Series:
    """Combine Beast/PFF/BR columns into one text per row — all available sources concatenated."""
    bt  = df[BEAST_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()
    pt  = df[PFF_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()
    brt = df[BR_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()
    parts = bt.where(bt.str.len() > 0, "")
    parts = (parts + " " + pt).str.strip()
    parts = (parts + " " + brt).str.strip()
    # Fallback: if still empty somehow, shouldn't happen but guard anyway
    return parts.where(parts.str.len() > 0, bt.where(bt.str.len() > 0, pt))


def unified_text_row(row) -> str:
    """Single-row version (dict or Series) for per-player inference."""
    bt  = " ".join(str(row.get(c, "") or "") for c in BEAST_COLS).strip()
    pt  = " ".join(str(row.get(c, "") or "") for c in PFF_COLS).strip()
    brt = " ".join(str(row.get(c, "") or "") for c in BR_COLS).strip()
    return " ".join(part for part in [bt, pt, brt] if part).strip()


def text_source_flag(df: pd.DataFrame) -> np.ndarray:
    """1 if Beast text is absent (PFF-sourced), 0 if Beast. Shape (n, 1)."""
    bt = df[BEAST_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()
    return (bt.str.len() == 0).astype(float).values.reshape(-1, 1)


def has_br_flag(df: pd.DataFrame) -> np.ndarray:
    """1 if Bleacher Report text is present. Shape (n, 1)."""
    brt = df[BR_COLS].fillna("").apply(lambda r: " ".join(r), axis=1).str.strip()
    return (brt.str.len() > 0).astype(float).values.reshape(-1, 1)


def get_meas(df: pd.DataFrame) -> np.ndarray:
    """Z-scored measurables, NaN → 0. Shape (n, 11)."""
    return df[Z_COLS].fillna(0.0).values


# ── Numeric features ──────────────────────────────────────────────────────────

def consensus_feature(df: pd.DataFrame) -> np.ndarray:
    """Per-year z-score of consensus rank, 0-imputed for missing. Shape (n, 1)."""
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


def beast_rank_feature(df: pd.DataFrame) -> np.ndarray:
    """Per-year z-score of beast_rank, 0-imputed for missing. Shape (n, 1)."""
    out = np.zeros(len(df))
    df2 = df.reset_index(drop=True)
    for _, grp in df2.groupby("draft_year"):
        vals = grp["beast_rank"].dropna()
        if len(vals) < 2:
            continue
        mu, sigma = vals.mean(), vals.std()
        if sigma < 1e-6:
            continue
        out[grp.index] = df2.loc[grp.index, "beast_rank"].fillna(mu).map(
            lambda v: (v - mu) / sigma
        )
    return out.reshape(-1, 1)


def text_length_feature(texts: pd.Series) -> np.ndarray:
    """Log text length, standardized to zero mean / unit variance. Shape (n, 1)."""
    lengths = np.log1p(texts.str.len().values.astype(float))
    mu, sigma = lengths.mean(), lengths.std()
    if sigma < 1e-6:
        return np.zeros((len(texts), 1))
    return ((lengths - mu) / sigma).reshape(-1, 1)


def round_feature(df: pd.DataFrame) -> np.ndarray:
    """One-hot draft rounds 1-7 + imputed flag (imputed from consensus when round unknown). Shape (n, 8)."""
    rounds  = df["round"].copy()
    missing = rounds.isna()
    if missing.any():
        cons = df.loc[missing, "consensus"].fillna(250)
        rounds.loc[missing] = (cons // 32 + 1).clip(upper=7)
    rounds = rounds.fillna(4).astype(int).clip(1, 7)
    enc = OneHotEncoder(categories=[list(range(1, 8))], sparse_output=False, handle_unknown="ignore")
    ohe = enc.fit_transform(rounds.values.reshape(-1, 1))
    return np.hstack([ohe, missing.astype(float).values.reshape(-1, 1)])


def position_feature(df: pd.DataFrame) -> np.ndarray:
    """One-hot position group. Shape (n, 12)."""
    groups = df["Position"].str.upper().map(POS_TO_GROUP).fillna("OTHER")
    enc = OneHotEncoder(categories=[POS_ORDER], sparse_output=False, handle_unknown="ignore")
    return enc.fit_transform(groups.values.reshape(-1, 1))


# ── Explanation helpers ───────────────────────────────────────────────────────

_STOPWORD_UNIGRAMS = {
    "a","an","the","and","or","but","in","of","to","for","on","at","by","as",
    "is","are","was","were","be","been","has","had","have","it","its","he","his",
    "him","they","them","their","we","with","from","into","up","so","if","no",
    "not","can","do","does","did","all","some","any","this","that","which",
    "when","where","while","who","what","how","then","just","very","more","like",
    "also","than","yet","too","even","both","each","few","own","same","will",
    "would","could","should","may","might","much","many","most","over","out",
    "off","use","uses","used","per","one","two","three","four","five","six",
    "such","other","after","before","about","upon","only","new","there","here",
}

_STOPWORD_BIGRAMS = {
    # generic prepositions / articles
    "to be","in the","of the","on the","and on","as an","to the","he is",
    "it is","at the","for the","from the","with the","by the","into the",
    "on his","in his","of his","to his","and the","in a","of a","to a",
    "him the","back seasons","in back","named to","became the","player in",
    # generic time / statistical language Beast uses for all top prospects
    "over the","at least","the last","of just","one of","the past","past two",
    "two seasons","in this","this class","to win","give his",
    # generic connectives / fragments
    "is an","but he","runner and","athlete with","allow him","allows him",
    "ability to","is to","he can","though he","him to","the way","in the way",
    "when playing","or plays","and tight","or carry","when he",
    # high-frequency generic phrases found in output
    "where he","his best","percent of","he has","he was","off the","up in",
    "from his","his way","how to","the ball","top of","with his","because of",
    "his first","in run","his college","so he","to make","has been","in man",
    "of just","in back","his second","on both","to find","to get","to use",
    "and has","and is","and he","he also","he will","who has","who is",
    "they are","there is","it was","there are","to do","to play","to help",
}


def deduplicate_ngrams(contribs: list) -> list:
    """
    Remove n-grams that overlap with already-selected longer/equal phrases.
    Uses both string containment and word-token overlap so fragments like
    'against tight' are dropped when 'tight ends' is already selected.
    """
    selected, phrases = [], []
    for word, contrib in contribs:
        w_tokens = set(word.lower().split())
        skip = False
        for ph in phrases:
            ph_tokens = set(ph.lower().split())
            if word in ph or ph in word:
                skip = True; break
            if w_tokens & ph_tokens and len(w_tokens) <= len(ph_tokens):
                skip = True; break
        if not skip:
            selected.append((word, contrib))
            phrases.append(word)
    return selected


_NEGATIVE_MARKERS = {
    "lack", "lacks", "lacking", "limited", "limitation", "below", "struggle",
    "struggles", "struggling", "average", "weak", "weakness", "weaknesses",
    "poor", "needs", "need to", "must", "concern", "concerns", "issue", "issues",
    "can't", "cannot", "doesn't", "doesn't have", "does not", "not", "without",
    "though", "however", "but", "despite", "although", "worrisome", "inconsistent",
    "unreliable", "disappointing", "question mark", "raw", "underdeveloped",
    "overmatched", "misses", "missed", "miss", "loses", "lost", "slow", "late",
    "short", "small", "thin", "narrow", "stiff", "tight", "rigid", "pad level",
    "overpowered", "outmatched", "rarely", "seldom", "never", "only", "just",
    "improve", "improvement", "development", "developmental", "project",
}


def find_source_sentences(text: str, contribs: list, n: int = 3,
                           positive: bool = True, top_per_sent: int = 3,
                           min_contrib: float = 0.005) -> list:
    """
    Return up to n sentences ranked by total absolute phrase contribution.
    Filters to multi-word phrases (≥2 tokens), excluding stopword bigrams,
    and drops phrases below min_contrib to remove TF-IDF noise.
    Beast reports use ● as bullet separators; also splits on . and newlines.
    Returns list of (sentence, [(phrase, score), ...]).
    For concern sentences (positive=False), requires at least one negative
    language marker so pure-positive sentences aren't mislabeled as concerns.
    """
    raw = re.split(r'[●•|\n]|(?<=[a-z])\.\s+', text)
    sentences = [s.strip() for s in raw if len(s.strip()) > 20]

    sign_filtered = [(w, c) for w, c in contribs if (c > 0) == positive and abs(c) >= min_contrib]
    if not sign_filtered:
        sign_filtered = [(w, c) for w, c in contribs if (c > 0) == positive]
    multi_word    = [(w, c) for w, c in sign_filtered
                     if " " in w and w.lower() not in _STOPWORD_BIGRAMS]
    phrases = multi_word if multi_word else [(w, c) for w, c in sign_filtered if " " in w]

    sentence_scores, sentence_phrases = {}, {}
    for phrase, contrib in phrases:
        for sent in sentences:
            if phrase.lower() in sent.lower():
                sentence_scores[sent] = sentence_scores.get(sent, 0) + abs(contrib)
                sentence_phrases.setdefault(sent, []).append((phrase, contrib))
                break

    ranked = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    out = []
    for s in ranked:
        if len(out) >= n:
            break
        if not positive:
            words = set(re.sub(r"[^\w\s]", " ", s.lower()).split())
            if not (words & _NEGATIVE_MARKERS or
                    any(m in s.lower() for m in _NEGATIVE_MARKERS if " " in m)):
                continue
        top = sorted(sentence_phrases.get(s, []), key=lambda x: -abs(x[1]))[:top_per_sent]
        out.append((s, top))
    return out
