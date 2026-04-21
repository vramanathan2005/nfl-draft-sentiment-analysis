"""
Scrape NFL.com draft prospect + combine data for 2014-2025.

Data sources (api.nfl.com — requires Bearer JWT captured from browser session):
  /football/v2/combine/profiles?limit=1000&year={year}
      → bio, measurements, scouting (overview/strengths/weaknesses), scores
  /football/v2/draft/picks/report?limit=1000&year={year}
      → draft round/pick/overall + team
  /football/v2/teams?limit=32 (or /v1/teams)
      → team abbreviation lookup

Strategy:
  1. Launch headless Chromium via Playwright, load any NFL.com page, capture JWT.
  2. Use requests + that JWT to call the APIs for each year.
  3. Refresh the JWT if a 401 is returned.
  4. Write one CSV row per prospect.
"""

import csv
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_CSV  = Path("nfl_prospects_nflcom.csv")
YEARS       = list(range(2014, 2026))   # 2014 → 2025 inclusive
API_BASE    = "https://api.nfl.com"
REQUEST_DELAY = 0.5   # seconds between API calls

CSV_FIELDS = [
    # Identity
    "year", "person_id", "esb_id", "first_name", "last_name", "display_name",
    "hometown", "college", "college_class", "position", "position_group",
    # Measurements
    "height_in", "weight_lbs", "arm_length", "hand_size",
    # Combine / Pro Day times
    "forty_yard_dash", "ten_yard_split",
    "twenty_yard_shuttle", "sixty_yard_shuttle",
    "three_cone", "bench_press", "broad_jump", "vertical_jump",
    # Grades & projections
    "grade", "draft_grade", "draft_projection",
    "athleticism_score", "production_score", "size_score",
    # Scouting text
    "nfl_comparison", "overview", "strengths", "weaknesses",
    "sources_tell_us", "bio",
    "profile_author", "combine_attendance",
    # Draft result
    "draft_round", "draft_pick", "draft_overall", "draft_team_id", "draft_team_abbr",
    # Headshot URL
    "headshot_url",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return " ".join(self.fed).strip()


def strip_html(html: str) -> str:
    """Strip HTML tags, convert <li> items to bullet text."""
    if not html:
        return ""
    html = re.sub(r"<li[^>]*>", "• ", html)
    html = re.sub(r"</li>", " ", html)
    s = _MLStripper()
    s.feed(html)
    return " ".join(s.get_data().split())


# ---------------------------------------------------------------------------
# Auth: capture JWT from browser
# ---------------------------------------------------------------------------

def get_jwt() -> str:
    """Load any NFL.com page with Playwright and steal the Bearer JWT."""
    print("  Launching headless browser to capture JWT...")
    token = ""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_page()

        def on_request(req):
            nonlocal token
            if "api.nfl.com" in req.url:
                auth = req.headers.get("authorization", "")
                if auth.startswith("Bearer "):
                    token = auth

        page.on("request", on_request)
        try:
            page.goto(
                "https://www.nfl.com/draft/tracker/prospects/all-positions/all-colleges/all-statuses/2025?page=1",
                wait_until="load",
                timeout=30_000,
            )
            page.wait_for_timeout(6_000)
        except Exception as e:
            print(f"    (page load note: {e})")
        browser.close()

    if not token:
        raise RuntimeError("Could not capture JWT from NFL.com — check network/VPN.")
    print(f"  JWT captured (first 40 chars): {token[:40]}…")
    return token


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers.update({
    "Accept":     "application/json, text/plain, */*",
    "Referer":    "https://www.nfl.com/",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
})

_current_jwt = ""


def _set_jwt(token: str):
    global _current_jwt
    _current_jwt = token
    SESSION.headers["Authorization"] = token


def api_get(path: str, params: dict = None, retry: bool = True) -> dict | list:
    """GET from api.nfl.com, refreshing JWT once on 401."""
    url = API_BASE + path
    r   = SESSION.get(url, params=params, timeout=20)
    if r.status_code == 401 and retry:
        print("  401 — refreshing JWT...")
        _set_jwt(get_jwt())
        return api_get(path, params, retry=False)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Team lookup  (best-effort — returns {} if endpoint unavailable)
# ---------------------------------------------------------------------------

def build_team_lookup() -> dict:
    """Return {teamId: abbreviation} dict (team IDs are stable across seasons)."""
    try:
        data  = api_get("/experience/v1/teams", {"season": 2025})
        teams = data.get("teams", [])
        return {t["id"]: t.get("abbreviation", "") for t in teams if "id" in t}
    except Exception as e:
        print(f"  Team lookup failed ({e}), storing raw teamId instead.")
        return {}


# ---------------------------------------------------------------------------
# Per-year scrapers
# ---------------------------------------------------------------------------

def fetch_combine_profiles(year: int) -> dict:
    """Return {personId: profile_dict}."""
    data     = api_get("/football/v2/combine/profiles", {"limit": 1000, "year": year})
    profiles = data.get("combineProfiles", [])
    return {p["person"]["id"]: p for p in profiles}


def fetch_draft_picks(year: int) -> dict:
    """Return {personId: pick_dict}."""
    try:
        data  = api_get("/football/v2/draft/picks/report", {"limit": 1000, "year": year})
        picks = data.get("picks", [])
        return {pk["personId"]: pk for pk in picks if "personId" in pk}
    except Exception as e:
        print(f"    Draft picks unavailable for {year}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def build_row(profile: dict, pick: dict | None, team_lookup: dict, year: int) -> dict:
    person = profile.get("person", {})

    row = {
        # Identity
        "year":            year,
        "person_id":       person.get("id", ""),
        "esb_id":          person.get("esbId", ""),
        "first_name":      person.get("firstName", ""),
        "last_name":       person.get("lastName", ""),
        "display_name":    person.get("displayName", ""),
        "hometown":        person.get("hometown", ""),
        "college":         ", ".join(person.get("collegeNames", [])),
        "college_class":   profile.get("collegeClass", ""),
        "position":        profile.get("position", ""),
        "position_group":  profile.get("positionGroup", ""),

        # Measurements
        "height_in":   profile.get("height"),      # inches (float)
        "weight_lbs":  profile.get("weight"),
        "arm_length":  profile.get("armLength"),
        "hand_size":   profile.get("handSize"),

        # Combine / Pro Day
        "forty_yard_dash":      profile.get("fortyYardDash"),
        "ten_yard_split":       profile.get("tenYardSplit"),
        "twenty_yard_shuttle":  profile.get("twentyYardShuttle"),
        "sixty_yard_shuttle":   profile.get("sixtyYardShuttle"),
        "three_cone":           profile.get("threeConeDrill"),
        "bench_press":          profile.get("benchPress"),
        "broad_jump":           profile.get("broadJump"),
        "vertical_jump":        profile.get("verticalJump"),

        # Grades
        "grade":              profile.get("grade"),
        "draft_grade":        profile.get("draftGrade"),
        "draft_projection":   profile.get("draftProjection"),
        "athleticism_score":  profile.get("athleticismScore"),
        "production_score":   profile.get("productionScore"),
        "size_score":         profile.get("sizeScore"),

        # Scouting text (strip HTML)
        "nfl_comparison":   strip_html(profile.get("nflComparison", "")),
        "overview":         strip_html(profile.get("overview", "")),
        "strengths":        strip_html(profile.get("strengths", "")),
        "weaknesses":       strip_html(profile.get("weaknesses", "")),
        "sources_tell_us":  strip_html(profile.get("sourcesTellUs", "")),
        "bio":              strip_html(profile.get("bio", "")),
        "profile_author":   profile.get("profileAuthor", ""),
        "combine_attendance": profile.get("combineAttendance", ""),

        # Draft result
        "draft_round":    "",
        "draft_pick":     "",
        "draft_overall":  "",
        "draft_team_id":  "",
        "draft_team_abbr": "",

        # Headshot
        "headshot_url": (profile.get("headshot") or "").replace("{formatInstructions}/", ""),
    }

    if pick:
        team_id = pick.get("teamId", "")
        row.update({
            "draft_round":    pick.get("draftRound", ""),
            "draft_pick":     pick.get("draftPosition", ""),
            "draft_overall":  pick.get("draftNumberOverall", ""),
            "draft_team_id":  team_id,
            "draft_team_abbr": team_lookup.get(team_id, ""),
        })

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Step 1: Capturing JWT from NFL.com...")
    _set_jwt(get_jwt())

    print("\nStep 2: Building team lookup...")
    team_lookup = build_team_lookup()
    print(f"  Teams loaded: {len(team_lookup)}")

    all_rows = []

    for year in YEARS:
        print(f"\nYear {year}:")

        print(f"  Fetching combine profiles...")
        profiles = fetch_combine_profiles(year)
        print(f"  → {len(profiles)} profiles")

        time.sleep(REQUEST_DELAY)

        print(f"  Fetching draft picks...")
        picks = fetch_draft_picks(year)
        print(f"  → {len(picks)} picks")

        time.sleep(REQUEST_DELAY)

        for person_id, profile in profiles.items():
            pick = picks.get(person_id)
            row  = build_row(profile, pick, team_lookup, year)
            all_rows.append(row)

    print(f"\nTotal rows: {len(all_rows)}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
