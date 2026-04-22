"""
Scrape player prospect data and scouting reports from Dane Brugler's
"The Beast" NFL Draft Guide PDFs into a single CSV.

Format variants:
  OLD (2019, 2020, 2022, 2023, 2024):
      "1.   NAME | SCHOOL   6034 | 221 lbs. | rSR.   Hometown   1/10/1996 (age 23.37)   #9"
  NEW (2025):
      "QB1 Cam Ward\nMiami, 5SR"  +  info-box row

Combine column order:
  2019-2023:  HT  WT  ARM   HAND  WING  40  20  10  VJ  BJ  SS  3C  BP
  2024:       HT  WT  HAND  ARM   WING  40  20  10  VJ  BJ  SS  3C  BP
  2025 (new): HT  WT  HAND  ARM   WING  40  20  10  VJ  BJ  SS  3C  (no BP)
"""

import re
import csv
from pathlib import Path
import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PDF_FILES = [
    ("2019", "Dane-Bruglers-2019-NFL-Draft-Guide-Athletic.pdf", ""),
    ("2020", "2020-NFL-DRAFT-FINAL-2.pdf", ""),
    ("2022", "2022_TheBeast_NFL_Draft_Guide-6.pdf", ""),
    ("2023", "TheBeast_NFL_Draft_Guide-7.pdf", ""),
    ("2024", "TheBeast2024_MasterFile-9.pdf", ""),
    ("2025", "2025-Beast-v03.pdf", ""),
    ("2026", "the-beast-2026.pdf", "thebeast2026!"),
]


BASE_DIR = Path("Dane_Brugler_The_Beast")
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "raw" / "beast_prospects.csv"

POSITIONS = [
    "QUARTERBACKS", "RUNNING BACKS", "FULLBACKS", "FULLBACKS/H-BACKS",
    "WIDE RECEIVERS", "TIGHT ENDS",
    "OFFENSIVE TACKLES", "OFFENSIVE GUARDS", "OFFENSIVE CENTERS",
    "GUARDS", "CENTERS",
    "EDGE RUSHERS", "DEFENSIVE LINEMEN", "DEFENSIVE TACKLES", "LINEBACKERS",
    "CORNERBACKS", "SAFETIES", "SPECIALISTS",
]

POSITION_ABBREVS = {
    "QUARTERBACKS": "QB", "RUNNING BACKS": "RB", "FULLBACKS": "FB",
    "FULLBACKS/H-BACKS": "FB", "WIDE RECEIVERS": "WR", "TIGHT ENDS": "TE",
    "OFFENSIVE TACKLES": "OT", "OFFENSIVE GUARDS": "OG",
    "OFFENSIVE CENTERS": "OC", "GUARDS": "OG", "CENTERS": "OC",
    "EDGE RUSHERS": "EDGE", "DEFENSIVE LINEMEN": "DL",
    "DEFENSIVE TACKLES": "DT", "LINEBACKERS": "LB",
    "CORNERBACKS": "CB", "SAFETIES": "S", "SPECIALISTS": "SPEC",
}

POS_ABBREV_2025 = {
    "QB": "QB", "RB": "RB", "FB": "FB", "WR": "WR", "TE": "TE",
    "OT": "OT", "OG": "OG", "OC": "OC", "IOL": "OG",
    "EDGE": "EDGE", "DL": "DL", "DT": "DT", "LB": "LB",
    "CB": "CB", "S": "S", "K": "SPEC", "P": "SPEC", "LS": "SPEC",
}

# Standardized combine column names (always output in this order)
COMBINE_COLS = [
    "c_ht", "c_wt", "c_arm", "c_hand", "c_wing",
    "c_40", "c_20", "c_10",
    "c_vj", "c_bj", "c_ss", "c_3c", "c_bp",
    "c_notes",
    "pd_ht", "pd_wt", "pd_arm", "pd_hand", "pd_wing",
    "pd_40", "pd_20", "pd_10",
    "pd_vj", "pd_bj", "pd_ss", "pd_3c", "pd_bp",
    "pd_notes",
]

CSV_FIELDS = [
    "draft_year", "position", "rank", "name", "school", "year_class",
    "height", "weight", "hometown", "birthday", "age", "jersey",
    "strengths", "weaknesses", "summary", "grade",
] + COMBINE_COLS


# ---------------------------------------------------------------------------
# Combine parser
# ---------------------------------------------------------------------------

# Tokenize measurement values out of a raw text string.
# Order matters: longer/more-specific patterns first.
# BJ values use Unicode typographic quotes: ' (U+2019) and " (U+201D)
_FOOT  = r"['\u2019]"   # ASCII or right-single-quote for feet
_INCH  = r'["\u201d]?'  # ASCII or right-double-quote for inches (optional)
_VAL_TOK = re.compile(
    r'(\([^)]+\))'                         # parenthetical note  → group 1
    r'|(\d+' + _FOOT + r'\s*\d+\s*' + _INCH + r')'  # BJ feet/in → group 2
    r'|(\d+\s+\d+/\d+)'                    # fraction            → group 3
    r'|(\d+\.\d+)'                          # decimal time        → group 4
    r'|([56]\d{3})'                         # height              → group 5
    r'|(\d{1,3})'                           # integer (1-3 digit) → group 6
    r'|(DNP|N/A|—|–|(?<!\w)-(?!\w))'         # missing (dash must not be in a word)
)


def _tokenize_values(raw: str) -> list[str]:
    """Pull all measurement tokens from raw text, preserving order."""
    return [m.group(0).strip() for m in _VAL_TOK.finditer(raw)]


# Per-year column order: list of (dest_column, source_index) mappings.
# source_index is the 0-based position in the tokenized COMBINE/PRO DAY row.
# The last token (if it is a parenthetical note) maps to _notes.

def _col_order(draft_year: str, row_type: str) -> list[str]:
    """Return list of column names corresponding to each positional token."""
    prefix = "c_" if row_type == "combine" else "pd_"
    # 2019-2023: HT WT ARM HAND WING 40 20 10 VJ BJ SS 3C BP
    if draft_year in ("2019", "2020", "2022", "2023"):
        return [
            prefix+"ht", prefix+"wt", prefix+"arm", prefix+"hand", prefix+"wing",
            prefix+"40", prefix+"20", prefix+"10",
            prefix+"vj", prefix+"bj", prefix+"ss", prefix+"3c", prefix+"bp",
        ]
    # 2024: HT WT HAND ARM WING 40 20 10 VJ BJ SS 3C BP
    elif draft_year == "2024":
        return [
            prefix+"ht", prefix+"wt", prefix+"hand", prefix+"arm", prefix+"wing",
            prefix+"40", prefix+"20", prefix+"10",
            prefix+"vj", prefix+"bj", prefix+"ss", prefix+"3c", prefix+"bp",
        ]
    # 2025: HT WT HAND ARM WING 40 20 10 VJ BJ SS 3C  (no BP)
    else:
        return [
            prefix+"ht", prefix+"wt", prefix+"hand", prefix+"arm", prefix+"wing",
            prefix+"40", prefix+"20", prefix+"10",
            prefix+"vj", prefix+"bj", prefix+"ss", prefix+"3c",
        ]


def parse_combine_row(raw: str, draft_year: str, row_type: str) -> dict:
    """
    Parse one COMBINE or PRO DAY row.
    `raw` is the text of that row (after stripping the COMBINE/PRO DAY keyword).
    Returns a dict of combine column values.
    """
    prefix = "c_" if row_type == "combine" else "pd_"
    result = {col: "" for col in [prefix+"ht", prefix+"wt", prefix+"arm", prefix+"hand",
                                   prefix+"wing", prefix+"40", prefix+"20", prefix+"10",
                                   prefix+"vj", prefix+"bj", prefix+"ss", prefix+"3c",
                                   prefix+"bp", prefix+"notes"]}

    tokens = _tokenize_values(raw)
    cols   = _col_order(draft_year, row_type)

    for idx, col in enumerate(cols):
        if idx < len(tokens) and not tokens[idx].startswith("("):
            result[col] = tokens[idx]

    # Notes: everything after the last measurement token.
    # Handles "(no workout – choice)" and "No workout (choice)" styles.
    last_meas_end = 0
    for tok_m in _VAL_TOK.finditer(raw):
        if not tok_m.group(1):   # skip parenthetical tokens
            last_meas_end = tok_m.end()
    notes_raw = " ".join(raw[last_meas_end:].split()).strip()
    if notes_raw:
        # Strip outer parens if fully wrapped, e.g. "(no workout – choice)"
        if notes_raw.startswith("(") and notes_raw.endswith(")"):
            notes_raw = notes_raw[1:-1]
        result[prefix+"notes"] = notes_raw

    return result


def extract_combine(block: str, draft_year: str) -> dict:
    """
    Find COMBINE and PRO DAY rows in a player block and parse them.
    Returns a dict with all combine/proday columns.
    """
    empty = {col: "" for col in COMBINE_COLS}

    # Find the COMBINE section (text from COMBINE up to PRO DAY)
    combine_m = re.search(r'\bCOMBINE\b', block)
    proday_m  = re.search(r'\bPRO DAY\b', block)

    if not combine_m:
        return empty

    # Text of the COMBINE row = from after "COMBINE" up to PRO DAY (or STRENGTHS)
    combine_end = proday_m.start() if proday_m else len(block)
    combine_raw = block[combine_m.end():combine_end]

    result = parse_combine_row(combine_raw, draft_year, "combine")

    if proday_m:
        # PRO DAY row ends at STRENGTHS or end of block
        strengths_m = re.search(r'\bSTRENGTHS\b', block[proday_m.end():])
        proday_end  = proday_m.end() + strengths_m.start() if strengths_m else len(block)
        proday_raw  = block[proday_m.end():proday_end]
        result.update(parse_combine_row(proday_raw, draft_year, "proday"))

    return result


# ---------------------------------------------------------------------------
# Scouting text helpers
# ---------------------------------------------------------------------------

STRENGTHS_RE = re.compile(
    r'STRENGTHS[:\s]*(.*?)(?=WEAKNESSES[:\s]|$)', re.DOTALL | re.IGNORECASE
)
WEAKNESSES_RE = re.compile(
    r'WEAKNESSES[:\s]*(.*?)(?=SUMMARY[:\s]|$)', re.DOTALL | re.IGNORECASE
)
SUMMARY_RE = re.compile(
    r'SUMMARY[:\s]*(.*?)(?=\nGRADE:|$)', re.DOTALL | re.IGNORECASE
)
GRADE_RE = re.compile(
    r'(?:^|\n)GRADE:\s*(.+?)(?:\n|$)', re.IGNORECASE
)


def clean(s: str) -> str:
    return " ".join(s.split()).strip() if s else ""


def extract_scouting(block: str) -> dict:
    out = {}
    for key, pattern in [
        ("strengths", STRENGTHS_RE),
        ("weaknesses", WEAKNESSES_RE),
        ("summary", SUMMARY_RE),
    ]:
        m = pattern.search(block)
        out[key] = clean(m.group(1)) if m else ""
    gm = GRADE_RE.search(block)
    out["grade"] = clean(gm.group(1)) if gm else ""
    return out


# ---------------------------------------------------------------------------
# Old format parser  (2019, 2020, 2022, 2023, 2024)
# ---------------------------------------------------------------------------

OLD_HEADER_RE = re.compile(
    r'^(\d+)\.\s{1,6}([A-Z][A-Z\s\'\-\.]+?)\s*\|\s*(.+?)\s+'
    r'(\d{4})\s*\|\s*(\d+)\s+lbs\.\s*\|\s*(\S+)\s+'
    r'(.+?)\s+'
    r'(\d{1,2}/\d{1,2}/\d{4})\s*\(age\s*([\d.]+)\)'
    r'\s*#(\d+)',
    re.MULTILINE
)

POS_HEADING_RE = re.compile(
    r'^\s*(' + "|".join(re.escape(p) for p in POSITIONS) + r')\s*$',
    re.MULTILINE
)


def parse_old_format(pages: list[str], draft_year: str) -> list[dict]:
    full_text = "\n".join(pages)
    records   = []

    hits      = list(OLD_HEADER_RE.finditer(full_text))
    pos_spans = [(m.start(), m.group(1).strip()) for m in POS_HEADING_RE.finditer(full_text)]

    def get_position(offset: int) -> str:
        pos = "UNKNOWN"
        for span_start, pos_name in pos_spans:
            if span_start <= offset:
                pos = pos_name
            else:
                break
        return POSITION_ABBREVS.get(pos, pos)

    for i, m in enumerate(hits):
        start = m.start()
        end   = hits[i + 1].start() if i + 1 < len(hits) else len(full_text)
        block = full_text[start:end]

        position = get_position(start)

        if "TOP-100" in full_text[max(0, start - 200):start].upper():
            continue

        scouting = extract_scouting(block)
        combine  = extract_combine(block, draft_year)

        records.append({
            "draft_year": draft_year,
            "position":   position,
            "rank":       m.group(1),
            "name":       clean(m.group(2)),
            "school":     clean(m.group(3)),
            "year_class": m.group(6).rstrip("."),
            "height":     m.group(4),
            "weight":     m.group(5),
            "hometown":   clean(m.group(7)),
            "birthday":   m.group(8),
            "age":        m.group(9),
            "jersey":     m.group(10),
            "strengths":  scouting["strengths"],
            "weaknesses": scouting["weaknesses"],
            "summary":    scouting["summary"],
            "grade":      scouting["grade"],
            **combine,
        })

    print(f"  [old] {draft_year}: {len(records)} players parsed")
    return records


# ---------------------------------------------------------------------------
# New format parser  (2025)
# ---------------------------------------------------------------------------

NEW_HEADER_RE = re.compile(
    r'^([A-Z]{1,6})(\d+)\s+([A-Z][A-Za-z\s\'\-\.]+?)\n(.+?),\s+(\S+)\s*$',
    re.MULTILINE
)

BIRTHDAY_RE = re.compile(r'([A-Z][a-z]+ \d{1,2}, \d{4})')
AGE_RE      = re.compile(r'\b(\d{2}\.\d{2})\b')
HT_RE       = re.compile(r'\b([56]\d{3})\b')
NUM_RE      = re.compile(r'#(\d+)')


def parse_new_format(pages: list[str], draft_year: str) -> list[dict]:
    full_text = "\n".join(pages)
    records   = []

    hits = list(NEW_HEADER_RE.finditer(full_text))

    for i, m in enumerate(hits):
        start = m.start()
        end   = hits[i + 1].start() if i + 1 < len(hits) else len(full_text)
        block = full_text[start:end]

        pos_abbrev = m.group(1)
        rank       = m.group(2)
        name       = clean(m.group(3))
        school     = clean(m.group(4))
        year_class = m.group(5)
        position   = POS_ABBREV_2025.get(pos_abbrev, pos_abbrev)

        lines_after = block.split("\n")[2:22]
        info_chunk  = " ".join(lines_after)

        bday_m   = BIRTHDAY_RE.search(info_chunk)
        birthday = bday_m.group(1) if bday_m else ""

        age_m = AGE_RE.search(info_chunk)
        age   = age_m.group(1) if age_m else ""

        ht_candidates = HT_RE.findall(info_chunk)
        height = ht_candidates[0] if ht_candidates else ""

        wt_candidates = [
            x for x in re.findall(r'\b(\d{3})\b', info_chunk)
            if 140 <= int(x) <= 420 and x != height[-3:]
        ]
        weight = wt_candidates[0] if wt_candidates else ""

        num_m  = NUM_RE.search(info_chunk)
        jersey = num_m.group(1) if num_m else ""

        hometown = ""
        for line in lines_after:
            line = line.strip()
            if (line
                    and line not in ("HOMETOWN", "HIGH SCHOOL", "BIRTHDAY", "AGE", "HT", "WT", "NUM")
                    and not line.startswith("THE BEAST")
                    and not line.startswith("BACK TO")
                    and len(line) > 2):
                hometown = clean(line.split("  ")[0])
                break

        scouting = extract_scouting(block)
        combine  = extract_combine(block, draft_year)

        records.append({
            "draft_year": draft_year,
            "position":   position,
            "rank":       rank,
            "name":       name,
            "school":     school,
            "year_class": year_class,
            "height":     height,
            "weight":     weight,
            "hometown":   hometown,
            "birthday":   birthday,
            "age":        age,
            "jersey":     jersey,
            "strengths":  scouting["strengths"],
            "weaknesses": scouting["weaknesses"],
            "summary":    scouting["summary"],
            "grade":      scouting["grade"],
            **combine,
        })

    print(f"  [new] {draft_year}: {len(records)} players parsed")
    return records


# ---------------------------------------------------------------------------
# 2026 format parser
# ---------------------------------------------------------------------------

# Header line: "QB1 Fernando Mendoza Indiana"
# followed immediately by "GRADE" on the next line (info-box label).
NEW_2026_HEADER_RE = re.compile(
    r'^[ \t]*([A-Z]{1,6})(\d+)\s+([A-Z].+?)\s*\n(?=GRADE\s*\n)',
    re.MULTILINE
)

# Info-box: 8 label lines then 8 value lines.
INFOBOX_2026_RE = re.compile(
    r'GRADE\s*\nOVR\. RANK\s*\nYEAR\s*\nBIRTHDAY\s*\nAGE\s*\nHT\s*\nWT\s*\nJERSEY\s*\n'
    r'(.+?)\n(.+?)\n(.+?)\n(.+?)\n(.+?)\n(.+?)\n(.+?)\n(.+?)\n',
    re.DOTALL
)


def parse_2026_format(pages: list[str], draft_year: str) -> list[dict]:
    full_text = "\n".join(pages)
    records   = []

    hits = list(NEW_2026_HEADER_RE.finditer(full_text))

    for i, m in enumerate(hits):
        start = m.start()
        end   = hits[i + 1].start() if i + 1 < len(hits) else len(full_text)
        block = full_text[start:end]

        pos_abbrev = m.group(1)
        pos_rank   = m.group(2)
        name_school = m.group(3).strip()

        # First two words are the player name; remainder is the school.
        parts  = name_school.split()
        name   = " ".join(parts[:2]) if len(parts) >= 2 else name_school
        school = " ".join(parts[2:]) if len(parts) > 2 else ""

        position = POS_ABBREV_2025.get(pos_abbrev, pos_abbrev)

        # Parse info-box values.
        grade = year_class = birthday = age = height = weight = jersey = ""
        ib_m = INFOBOX_2026_RE.search(block)
        if ib_m:
            grade      = clean(ib_m.group(1))
            # group(2) is OVR. RANK — not stored separately
            year_class = clean(ib_m.group(3))
            birthday   = clean(ib_m.group(4))
            age        = clean(ib_m.group(5))
            height     = clean(ib_m.group(6))
            weight     = re.sub(r'\s*lbs\.?', '', clean(ib_m.group(7)))
            jersey     = re.sub(r'No\.\s*', '', clean(ib_m.group(8)))

        scouting = extract_scouting(block)
        combine  = extract_combine(block, draft_year)

        records.append({
            "draft_year": draft_year,
            "position":   position,
            "rank":       pos_rank,
            "name":       name,
            "school":     school,
            "year_class": year_class,
            "height":     height,
            "weight":     weight,
            "hometown":   "",
            "birthday":   birthday,
            "age":        age,
            "jersey":     jersey,
            "strengths":  scouting["strengths"],
            "weaknesses": scouting["weaknesses"],
            "summary":    scouting["summary"],
            "grade":      grade or scouting["grade"],
            **combine,
        })

    print(f"  [2026] {draft_year}: {len(records)} players parsed")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_records = []

    for draft_year, filename, password in PDF_FILES:
        pdf_path = BASE_DIR / filename
        if not pdf_path.exists():
            print(f"  SKIP (not found): {filename}")
            continue

        print(f"Processing {draft_year}: {filename}")
        pages = extract_pages(pdf_path, password)

        if draft_year == "2026":
            records = parse_2026_format(pages, draft_year)
        elif draft_year == "2025":
            records = parse_new_format(pages, draft_year)
        else:
            records = parse_old_format(pages, draft_year)

        all_records.extend(records)

    print(f"\nTotal records: {len(all_records)}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"Written to: {OUTPUT_CSV}")


def extract_pages(pdf_path: Path, password: str = "") -> list[str]:
    doc = fitz.open(str(pdf_path))
    if password:
        doc.authenticate(password)
    return [doc[i].get_text() for i in range(len(doc))]


if __name__ == "__main__":
    main()
