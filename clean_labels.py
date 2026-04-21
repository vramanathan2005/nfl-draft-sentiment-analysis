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

import pandas as pd
import numpy as np

df = pd.read_csv("all_prospects.csv")

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

# Label 4: sc_binary — binary version of sc_tier for improved signal at small N
df["sc_binary"] = df["sc_tier"].map({
    "real_contract": "made_it",
    "practice_only": "didnt",
    "out_of_league": "didnt",
})  # too_early → NaN automatically

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
print("=== sc_binary ===")
print(df["sc_binary"].value_counts())

df.to_csv("all_prospects.csv", index=False)
print("\nWrote → all_prospects.csv")
