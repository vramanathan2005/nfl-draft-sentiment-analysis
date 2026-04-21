"""
Build all_prospects.csv with player_consensus as the spine.

Left-joins:
  - Beast prospects  (strengths + weaknesses + summary → beast_text)
  - PFF scouting     (per-year text fields       → pff_text)
  - Bleacher Report  (positives + negatives       → br_text)
  - Second contracts (rank_on_date_APY, APY, etc.)

PFF 2025 excluded — no player name column.
BR 'overall' excluded — CSS-polluted in 2021-2025 scrape.
"""

import re
import pandas as pd
from pathlib import Path

BASE = Path("/Users/varunramanathan/Downloads/sentiment-analysis")
SUFFIXES = {"JR", "SR", "II", "III", "IV", "V"}


def clean_name(name: str) -> str:
    s = (name or "").upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    return " ".join(t for t in s.split() if t and t not in SUFFIXES)


def compact_name(name: str) -> str:
    return clean_name(name).replace(" ", "")


def name_tokens(name: str) -> list[str]:
    return clean_name(name).split()


def concat_text(*parts) -> str | None:
    joined = " | ".join(str(p) for p in parts if pd.notna(p) and str(p).strip())
    return joined if joined else None


# ── loaders ───────────────────────────────────────────────────────────────────

def load_beast() -> pd.DataFrame:
    df = pd.read_csv(BASE / "beast_prospects.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "draft_year":        r["draft_year"],
            "_name":             str(r["name"]),
            "beast_school":      r.get("school"),
            "beast_grade":       r.get("grade"),
            "beast_rank":        r.get("rank"),
            "beast_summary":     r.get("summary")     if pd.notna(r.get("summary"))     else None,
            "beast_strengths":   r.get("strengths")   if pd.notna(r.get("strengths"))   else None,
            "beast_weaknesses":  r.get("weaknesses")  if pd.notna(r.get("weaknesses"))  else None,
        })
    return pd.DataFrame(rows)


def load_pff() -> pd.DataFrame:
    rows = []

    # Normalized columns across all years:
    #   pff_overview   — main bio/narrative (overview, blurb1, blurb, profile)
    #   pff_pros       — strengths/positives
    #   pff_cons       — weaknesses/negatives
    #   pff_bottom_line — bottom-line verdict
    #   pff_extra      — additional fields (blurb2, where_wins, what_role, what_improve)
    # Note: 2020/2021 combine pros+cons into one field → goes into pff_pros, pff_cons=None

    def add(year, name, overview=None, pros=None, cons=None, bottom_line=None, extra=None):
        if not (pd.notna(name) and str(name).strip()):
            return
        rows.append({
            "draft_year":       year,
            "_name":            str(name),
            "pff_overview":     overview     if pd.notna(overview)     else None,
            "pff_pros":         pros         if pd.notna(pros)         else None,
            "pff_cons":         cons         if pd.notna(cons)         else None,
            "pff_bottom_line":  bottom_line  if pd.notna(bottom_line)  else None,
            "pff_extra":        extra        if pd.notna(extra)        else None,
        })

    df = pd.read_csv(BASE / "2017_data_pff.csv")
    for _, r in df.iterrows():
        add(2017, r["full_name"],
            overview=r.get("overview"), pros=r.get("pros"), cons=r.get("cons"))

    df = pd.read_csv(BASE / "2018_data_pff.csv")
    for _, r in df.iterrows():
        add(2018, r["name"],
            overview=r.get("overview"), bottom_line=r.get("bottom_line"))

    df = pd.read_csv(BASE / "2020_data_pff.csv")
    for _, r in df.iterrows():
        # blurb2 is additional narrative; pros_and_cons combines both sides
        add(2020, r["name"],
            overview=r.get("blurb1"), pros=r.get("pros_and_cons"), extra=r.get("blurb2"))

    df = pd.read_csv(BASE / "2021_data_pff.csv")
    for _, r in df.iterrows():
        add(2021, r["full_name"],
            overview=r.get("blurb1"), pros=r.get("pros_a_cons"), extra=r.get("blurb2"))

    df = pd.read_csv(BASE / "2022_data_pff.csv")
    for _, r in df.iterrows():
        add(2022, r["full_name"],
            overview=r.get("blurb"), pros=r.get("pros"), cons=r.get("cons"),
            bottom_line=r.get("bottom_line"),
            extra=concat_text(r.get("where_wins"), r.get("what_role"), r.get("what_improve")))

    df = pd.read_csv(BASE / "2024_data_pff.csv")
    for _, r in df.iterrows():
        name = r["player"] if pd.notna(r.get("player")) else f"{r['first_name']} {r['last_name']}"
        add(2024, name,
            overview=r.get("profile"), pros=r.get("strengths"),
            cons=r.get("weaknesses"), bottom_line=r.get("bottom_line"))

    # 2025: no player name column — skipped

    return pd.DataFrame(rows)


def load_bleacher() -> pd.DataFrame:
    rows = []

    for year in [2021, 2022, 2023, 2024, 2025]:
        df = pd.read_csv(BASE / f"Bleacher{year}.csv")
        for _, r in df.iterrows():
            name = r.get("player_from_title", "")
            if not pd.notna(name) or not str(name).strip():
                continue
            rows.append({
                "draft_year":        year,
                "_name":             str(name),
                # 'overall' excluded — CSS-polluted in scrape output
                "br_positives":      r.get("positives") if pd.notna(r.get("positives")) else None,
                "br_negatives":      r.get("negatives") if pd.notna(r.get("negatives")) else None,
                "br_article_grade":  r.get("article_grade"),
                "br_overall_rank":   r.get("overall_rank"),
                "br_position_rank":  r.get("position_rank"),
                "br_pro_comparison": r.get("pro_comparison"),
            })

    df = pd.read_csv(BASE / "Bleacher2026.csv")
    for _, r in df.iterrows():
        name = r.get("player_from_title", "")
        if not pd.notna(name) or not str(name).strip():
            continue
        rows.append({
            "draft_year":        2026,
            "_name":             str(name),
            # 2026 scrape uses different field names
            "br_positives":      r.get("where_he_wins")        if pd.notna(r.get("where_he_wins"))        else None,
            "br_negatives":      r.get("areas_of_improvement") if pd.notna(r.get("areas_of_improvement")) else None,
            "br_article_grade":  r.get("article_grade"),
            "br_overall_rank":   r.get("overall_rank"),
            "br_position_rank":  r.get("position_rank"),
            "br_pro_comparison": r.get("pro_comparison"),
        })

    return pd.DataFrame(rows)


def load_second_contracts() -> pd.DataFrame:
    df = pd.read_csv(BASE / "ranked_second_contracts.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "draft_year":          r["draft_year"],
            "_name":               str(r["player"]),
            "sc_APY":              r.get("APY"),
            "sc_rank_on_date_APY": r.get("rank_on_date_APY"),
            "sc_contract_type":    r.get("contract_type"),
            "sc_date_signed":      r.get("date_signed"),
            "sc_years":            r.get("Years"),
        })
    return pd.DataFrame(rows)


# ── fuzzy join ────────────────────────────────────────────────────────────────

def build_index(df: pd.DataFrame) -> dict:
    df = df.reset_index(drop=True)
    idx: dict = {}
    for i, row in df.iterrows():
        year = int(row["draft_year"])
        name = str(row["_name"])
        tokens = name_tokens(name)
        if not tokens:
            continue
        first, last = tokens[0], tokens[-1]
        idx.setdefault(("exact",       year, clean_name(name)),   []).append(i)
        idx.setdefault(("compact",     year, compact_name(name)), []).append(i)
        idx.setdefault(("firstlast",   year, first, last),        []).append(i)
        idx.setdefault(("initiallast", year, first[0], last),     []).append(i)
    return idx


def lookup(year: int, name: str, idx: dict, used: set) -> int | None:
    tokens = name_tokens(name)
    if not tokens:
        return None
    first, last = tokens[0], tokens[-1]
    keys = [
        ("exact",       year, clean_name(name)),
        ("compact",     year, compact_name(name)),
        ("firstlast",   year, first, last),
        ("initiallast", year, first[0], last),
    ]
    for key in keys:
        candidates = idx.get(key, [])
        # require uniqueness for weaker tiers to avoid false positives
        if key[0] in {"exact", "compact"}:
            for i in candidates:
                if i not in used:
                    return i
        elif len(candidates) == 1:
            i = candidates[0]
            if i not in used:
                return i
    return None


def left_join(spine: pd.DataFrame, spine_name_col: str, right: pd.DataFrame) -> pd.DataFrame:
    right = right.reset_index(drop=True)
    idx = build_index(right)
    used: set = set()

    value_cols = [c for c in right.columns if c not in {"draft_year", "_name"}]
    joined: dict = {c: [] for c in value_cols}

    for _, row in spine.iterrows():
        mi = lookup(int(row["draft_year"]), str(row[spine_name_col]), idx, used)
        if mi is not None:
            used.add(mi)
            for c in value_cols:
                joined[c].append(right.at[mi, c])
        else:
            for c in value_cols:
                joined[c].append(None)

    return pd.concat([spine.reset_index(drop=True), pd.DataFrame(joined)], axis=1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    spine = pd.read_csv(BASE / "player_consensus.csv", index_col=0)
    beast = load_beast()
    pff   = load_pff()
    br    = load_bleacher()
    sc    = load_second_contracts()

    merged = left_join(spine, "Player Name", beast)
    merged = left_join(merged, "Player Name", pff)
    merged = left_join(merged, "Player Name", br)
    merged = left_join(merged, "Player Name", sc)

    out = BASE / "all_prospects.csv"
    merged.to_csv(out, index=False)

    total    = len(merged)
    beast_n  = merged["beast_summary"].notna().sum()
    pff_n    = merged["pff_overview"].notna().sum()
    br_n     = merged["br_positives"].notna().sum()
    sc_n     = merged["sc_APY"].notna().sum()
    any_text = merged[["beast_summary", "pff_overview", "br_positives"]].notna().any(axis=1).sum()

    print(f"Total rows      : {total}")
    print(f"beast_summary   : {beast_n:4d}  ({beast_n/total:.1%})")
    print(f"pff_overview    : {pff_n:4d}  ({pff_n/total:.1%})")
    print(f"br_positives    : {br_n:4d}  ({br_n/total:.1%})")
    print(f"2nd contract    : {sc_n:4d}  ({sc_n/total:.1%})")
    print(f"any text        : {any_text:4d}  ({any_text/total:.1%})")
    print()

    print(f"{'Year':<6} {'Total':>6} {'Beast':>6} {'PFF':>5} {'BR':>5} {'SC':>5}")
    for yr in sorted(merged["draft_year"].unique()):
        sub = merged[merged["draft_year"] == yr]
        print(f"{yr:<6} {len(sub):>6} "
              f"{sub['beast_summary'].notna().sum():>6} "
              f"{sub['pff_overview'].notna().sum():>5} "
              f"{sub['br_positives'].notna().sum():>5} "
              f"{sub['sc_APY'].notna().sum():>5}")

    print(f"\nWrote → {out}")


if __name__ == "__main__":
    main()
