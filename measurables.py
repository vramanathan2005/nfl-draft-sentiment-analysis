"""
Parse Beast combine/pro-day measurables into clean floats and add to all_prospects.csv.

Formats encountered:
  height  : "6033"  → 6 ft, 03 in, 3/8 in  (NFL combine encoding: FIIEE where F=ft, II=in, E=eighths)
  fractions: "33 1/2", "09 5/8"  → decimal inches
  broad jump: "09'04\""  → feet'inches"  → total inches
  floats  : "4.69", "7.03"  → already decimal
  missing : "-", "", NaN   → NaN

Strategy:
  Use combine value when available; fall back to pro-day value when combine is "-".
  Outputs: ht_in, wt_lbs, arm_in, hand_in, wing_in, dash40, vj_in, bj_in, shuttle, cone3, bench
"""

import re
import numpy as np
import pandas as pd
from fractions import Fraction

# ── parsers ───────────────────────────────────────────────────────────────────

def parse_missing(val) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return True
    return str(val).strip() in {"", "-", "N/A", "na", "NA"}


def parse_fraction_inches(val) -> float | None:
    """'33 1/2' → 33.5,  '09 5/8' → 9.625"""
    if parse_missing(val):
        return None
    s = str(val).strip()
    try:
        parts = s.split()
        if len(parts) == 2:
            return float(parts[0]) + float(Fraction(parts[1]))
        return float(s)
    except Exception:
        return None


def parse_height(val) -> float | None:
    """'6033' → 75.375 inches  (6 ft, 3 in, 3/8 in)"""
    if parse_missing(val):
        return None
    s = str(val).strip()
    if len(s) == 4 and s.isdigit():
        feet   = int(s[0])
        inches = int(s[1:3])
        eighths = int(s[3])
        return feet * 12 + inches + eighths / 8
    # fallback: try plain float
    try:
        return float(s)
    except Exception:
        return None


def parse_broad(val) -> float | None:
    """'09\\'04"' → 112 inches"""
    if parse_missing(val):
        return None
    s = str(val).strip()
    m = re.match(r"(\d+)['\u2019](\d+)", s)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    try:
        return float(s)
    except Exception:
        return None


def parse_float(val) -> float | None:
    if parse_missing(val):
        return None
    try:
        return float(str(val).strip())
    except Exception:
        return None


# ── combine + pro-day merge ───────────────────────────────────────────────────

def best(c_val, pd_val, parser):
    """Use combine value; fall back to pro-day if combine is missing."""
    v = parser(c_val)
    if v is None:
        v = parser(pd_val)
    return v


def parse_beast(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ht_in"]   = [best(c, p, parse_height)           for c, p in zip(df["c_ht"],   df["pd_ht"])]
    out["wt_lbs"]  = [best(c, p, parse_float)            for c, p in zip(df["c_wt"],   df["pd_wt"])]
    out["arm_in"]  = [best(c, p, parse_fraction_inches)  for c, p in zip(df["c_arm"],  df["pd_arm"])]
    out["hand_in"] = [best(c, p, parse_fraction_inches)  for c, p in zip(df["c_hand"], df["pd_hand"])]
    out["wing_in"] = [best(c, p, parse_fraction_inches)  for c, p in zip(df["c_wing"], df["pd_wing"])]
    out["dash40"]  = [best(c, p, parse_float)            for c, p in zip(df["c_40"],   df["pd_40"])]
    out["vj_in"]   = [best(c, p, parse_fraction_inches)  for c, p in zip(df["c_vj"],   df["pd_vj"])]
    out["bj_in"]   = [best(c, p, parse_broad)            for c, p in zip(df["c_bj"],   df["pd_bj"])]
    out["shuttle"] = [best(c, p, parse_float)            for c, p in zip(df["c_ss"],   df["pd_ss"])]
    out["cone3"]   = [best(c, p, parse_float)            for c, p in zip(df["c_3c"],   df["pd_3c"])]
    out["bench"]   = [best(c, p, parse_float)            for c, p in zip(df["c_bp"],   df["pd_bp"])]
    return out


# ── position-relative z-scores ────────────────────────────────────────────────

POS_GROUPS = {
    "QB":   ["QB"],
    "RB":   ["RB", "FB"],
    "WR":   ["WR"],
    "TE":   ["TE"],
    "OL":   ["OT", "IOL", "C", "G", "T"],
    "EDGE": ["EDGE", "OLB", "DE"],
    "DL":   ["DL", "DT", "NT"],
    "LB":   ["LB", "ILB", "MLB"],
    "CB":   ["CB"],
    "S":    ["S", "FS", "SS"],
    "SPEC": ["K", "P", "LS"],
}
POS_TO_GROUP = {pos: grp for grp, positions in POS_GROUPS.items() for pos in positions}

MEAS_COLS = ["ht_in", "wt_lbs", "arm_in", "hand_in", "wing_in",
             "dash40", "vj_in", "bj_in", "shuttle", "cone3", "bench"]


def add_position_zscores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_pos_group"] = df["Position"].str.upper().map(POS_TO_GROUP).fillna("OTHER")

    for col in MEAS_COLS:
        z_col = f"z_{col}"
        df[z_col] = np.nan
        for grp, sub in df.groupby("_pos_group"):
            vals = sub[col].dropna()
            if len(vals) < 5:
                continue
            mu, sigma = vals.mean(), vals.std()
            if sigma < 1e-6:
                continue
            df.loc[sub.index, z_col] = (df.loc[sub.index, col] - mu) / sigma

    df.drop(columns=["_pos_group"], inplace=True)
    return df


# ── main: parse beast + append to all_prospects.csv ──────────────────────────

if __name__ == "__main__":
    beast = pd.read_csv("beast_prospects.csv")
    meas  = parse_beast(beast)

    print("=== Raw measurables fill rates ===")
    for col in MEAS_COLS:
        n = meas[col].notna().sum()
        print(f"  {col:<10}: {n:4d}/{len(meas)} ({n/len(meas):.0%})")

    print("\nSample parsed rows:")
    print(meas.join(beast[["draft_year","name","position"]]).dropna(subset=["dash40"]).head(5)[
        ["name","position","ht_in","wt_lbs","dash40","vj_in","bj_in","cone3"]
    ].to_string())

    # Reload all_prospects and join on (draft_year, name) via beast's index
    ap = pd.read_csv("all_prospects.csv")

    # Build a lookup: (draft_year, name) → measurables row index in beast
    SUFFIXES = {"JR", "SR", "II", "III", "IV", "V"}
    def clean(name):
        s = (name or "").upper()
        s = re.sub(r"[^A-Z0-9 ]+", " ", s)
        return " ".join(t for t in s.split() if t and t not in SUFFIXES)

    beast_key = beast.apply(lambda r: (int(r["draft_year"]), clean(str(r["name"]))), axis=1)
    key_to_idx = dict(zip(beast_key, beast.index))

    ap_keys = ap.apply(lambda r: (int(r["draft_year"]), clean(str(r["Player Name"]))), axis=1)
    meas_indices = ap_keys.map(key_to_idx)

    # Attach raw measurables
    for col in MEAS_COLS:
        ap[col] = meas_indices.map(lambda i: meas.at[i, col] if pd.notna(i) else np.nan)

    # Add position z-scores
    ap = add_position_zscores(ap)

    ap.to_csv("all_prospects.csv", index=False)

    print("\n=== Fill rates in all_prospects.csv ===")
    for col in MEAS_COLS:
        n = ap[col].notna().sum()
        print(f"  {col:<10}: {n:4d}/{len(ap)} ({n/len(ap):.0%})")

    z_cols = [f"z_{c}" for c in MEAS_COLS]
    print("\n=== Z-score fill rates ===")
    for col in z_cols:
        n = ap[col].notna().sum()
        print(f"  {col:<12}: {n:4d}/{len(ap)} ({n/len(ap):.0%})")

    print(f"\nWrote → all_prospects.csv  ({len(ap.columns)} columns total)")
