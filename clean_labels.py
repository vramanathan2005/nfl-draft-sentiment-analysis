"""
Clean and encode both NLP outcome labels, appending them to all_prospects.csv.

Label 1: draft_value  (below_consensus, cleaned to numeric)
  - numeric rows: used as-is (negative = fell in draft, positive = rose above consensus)
  - "undrafted - proj drafted": player scouts expected to be drafted but wasn't
      → encode as max_pick - consensus (how far they fell past the last pick)
  - "undrafted - proj undrafted": scouts didn't expect them drafted and they weren't
      → encode as 0 (consensus was correct, no surprise)

Label 2: sc_tier  (three-class NFL success label, restricted to 2017-2021 drafts
                   where careers have had enough time to mature)
  - "real_contract": got a non-practice second contract (UFA/SFA/Extension/etc.)
      + sc_rank_on_date_APY is the continuous version for regression
  - "practice_only": only ever signed practice squad second contracts
  - "out_of_league": no second contract at all
  - 2022+ drafts: labeled "too_early" — excluded from training
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("/Users/varunramanathan/Downloads/sentiment-analysis")

df = pd.read_csv(BASE / "data/processed/all_prospects.csv")

# ── Label 1: draft_value ──────────────────────────────────────────────────────

REAL_CONTRACT_TYPES = {"UFA", "SFA", "Extension", "Franchise", "RFA", "ERFA", "Transition"}

def encode_draft_value(row):
    bc = str(row["below_consensus"]).strip()
    if bc.lstrip("-").isnumeric():
        return int(bc)
    elif bc == "undrafted - proj drafted":
        # fell past the last pick — consensus was their floor, max_pick is where draft ended
        return int(row["max_pick"]) - int(row["consensus"])
    elif bc == "undrafted - proj undrafted":
        return 0
    return np.nan

df["draft_value"] = df.apply(encode_draft_value, axis=1)

# ── Label 2: sc_tier + sc_rank (continuous, for regression) ──────────────────

MATURE_YEARS = range(2017, 2022)  # 2022+ careers not yet mature enough

def encode_sc_tier(row):
    if row["draft_year"] not in MATURE_YEARS:
        return "too_early"
    if pd.isna(row["sc_APY"]):
        return "out_of_league"
    if row["sc_contract_type"] in REAL_CONTRACT_TYPES:
        return "real_contract"
    return "practice_only"

df["sc_tier"] = df.apply(encode_sc_tier, axis=1)

# For regression: sc_rank only meaningful for real_contract rows
df["sc_rank"] = df.apply(
    lambda r: r["sc_rank_on_date_APY"] if r["sc_tier"] == "real_contract" else np.nan,
    axis=1,
)

# Label 3: draft_tier — 3-class version of draft_value (±20 pick threshold)
def encode_draft_tier(dv):
    if pd.isna(dv):
        return np.nan
    if dv < -20:
        return "slide"
    if dv > 20:
        return "reach"
    return "consensus"

df["draft_tier"] = df["draft_value"].map(encode_draft_tier)

# Label 4: sc_contract_tier — 4-class career outcome using NFL front-office lingo
# APY thresholds computed per position group (p33/p67 of real_contract players)
# so "cornerstone" means top-third earner relative to position peers.
# Positions with <25 real_contract players fall back to global thresholds.

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
POS_TO_GROUP = {p: g for g, ps in POS_GROUPS.items() for p in ps}
MIN_GROUP_SIZE = 25  # fall back to global if fewer real_contract rows

rc = df[df["sc_tier"] == "real_contract"].copy()
rc["_pos_group"] = rc["Position"].str.upper().map(POS_TO_GROUP).fillna("OTHER")

global_p33 = rc["sc_APY"].quantile(0.33)
global_p67 = rc["sc_APY"].quantile(0.67)

pos_thresholds = {}
for grp, sub in rc.groupby("_pos_group"):
    apy = sub["sc_APY"].dropna()
    if len(apy) >= MIN_GROUP_SIZE:
        pos_thresholds[grp] = (apy.quantile(0.33), apy.quantile(0.67))

print("Position-specific APY thresholds (roster_filler | starter | cornerstone):")
for grp, (lo, hi) in sorted(pos_thresholds.items()):
    print(f"  {grp:<6}  <${lo/1e6:.2f}M  |  ${lo/1e6:.2f}M-${hi/1e6:.2f}M  |  >${hi/1e6:.2f}M")
print(f"  (global fallback: <${global_p33/1e6:.2f}M | >${global_p67/1e6:.2f}M)")

df["_pos_group"] = df["Position"].str.upper().map(POS_TO_GROUP).fillna("OTHER")

def encode_contract_tier(row):
    if row["sc_tier"] in ("too_early", None) or pd.isna(row["sc_tier"]):
        return np.nan
    if row["sc_tier"] == "out_of_league":
        return "out_of_league"
    if row["sc_tier"] == "practice_only":
        return "roster_bubble"
    apy = row["sc_APY"]
    if pd.isna(apy):
        return "out_of_league"
    lo, hi = pos_thresholds.get(row["_pos_group"], (global_p33, global_p67))
    if apy <= hi:
        return "53_man"
    return "cornerstone"

df["sc_contract_tier"] = df.apply(encode_contract_tier, axis=1)
df.drop(columns=["_pos_group"], inplace=True)

# ── Summary ───────────────────────────────────────────────────────────────────

print("=== draft_value ===")
print(f"numeric rows    : {df['draft_value'].notna().sum()}")
print(f"  of which = 0  : {(df['draft_value'] == 0).sum()}  (proj undrafted, drafted undrafted)")
print(f"  negative      : {(df['draft_value'] < 0).sum()}  (fell in draft)")
print(f"  positive      : {(df['draft_value'] > 0).sum()}  (rose / undrafted)")
print(f"  NaN           : {df['draft_value'].isna().sum()}")
print()
print(df["draft_value"].describe())

print()
print("=== sc_tier ===")
print(df["sc_tier"].value_counts())
print()
print("Training rows (2017-2021, has text):")
train = df[(df["sc_tier"] != "too_early") & df[["beast_summary","pff_overview","br_positives"]].notna().any(axis=1)]
print(train["sc_tier"].value_counts())
print()
print("sc_rank (real contracts only):")
print(df["sc_rank"].describe())

print()
print("=== draft_tier ===")
print(df["draft_tier"].value_counts())

print()
print("=== sc_contract_tier (4-class) ===")
print(df["sc_contract_tier"].value_counts())

df.to_csv(BASE / "data/processed/all_prospects.csv", index=False)
print("\nWrote → all_prospects.csv")
