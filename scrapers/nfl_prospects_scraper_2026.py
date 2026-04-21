"""
Scrape NFL.com draft prospect + combine data for the 2026 draft cycle.

Data sources (api.nfl.com — requires Bearer JWT captured from browser session):
  /football/v2/combine/profiles?limit=1000&year=2026
      -> bio, measurements, scouting (overview/strengths/weaknesses), scores
  /football/v2/draft/picks/report?limit=1000&year=2026
      -> draft round/pick/overall + team
  /experience/v1/teams?season=2026
      -> team abbreviation lookup

Strategy:
  1. Launch headless Chromium via Playwright, load an NFL.com draft page, capture JWT.
  2. Use requests + that JWT to call the APIs for 2026.
  3. Refresh the JWT if a 401 is returned.
  4. Write one CSV row per prospect.
"""

from __future__ import annotations

import csv
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright


OUTPUT_CSV = Path("nfl_prospects_nflcom_2026.csv")
YEAR = 2026
API_BASE = "https://api.nfl.com"
REQUEST_DELAY = 0.5

CSV_FIELDS = [
    "year",
    "person_id",
    "esb_id",
    "first_name",
    "last_name",
    "display_name",
    "hometown",
    "college",
    "college_class",
    "position",
    "position_group",
    "height_in",
    "weight_lbs",
    "arm_length",
    "hand_size",
    "forty_yard_dash",
    "ten_yard_split",
    "twenty_yard_shuttle",
    "sixty_yard_shuttle",
    "three_cone",
    "bench_press",
    "broad_jump",
    "vertical_jump",
    "grade",
    "draft_grade",
    "draft_projection",
    "athleticism_score",
    "production_score",
    "size_score",
    "nfl_comparison",
    "overview",
    "strengths",
    "weaknesses",
    "sources_tell_us",
    "bio",
    "profile_author",
    "combine_attendance",
    "draft_round",
    "draft_pick",
    "draft_overall",
    "draft_team_id",
    "draft_team_abbr",
    "headshot_url",
]


class _MLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.fed: list[str] = []

    def handle_data(self, d: str) -> None:
        self.fed.append(d)

    def get_data(self) -> str:
        return " ".join(self.fed).strip()


def strip_html(html: str) -> str:
    if not html:
        return ""
    html = re.sub(r"<li[^>]*>", "• ", html)
    html = re.sub(r"</li>", " ", html)
    stripper = _MLStripper()
    stripper.feed(html)
    return " ".join(stripper.get_data().split())


def get_jwt() -> str:
    print("  Launching headless browser to capture JWT...")
    token = ""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        def on_request(req) -> None:
            nonlocal token
            if "api.nfl.com" not in req.url:
                return
            auth = req.headers.get("authorization", "")
            if auth.startswith("Bearer "):
                token = auth

        page.on("request", on_request)
        try:
            page.goto(
                f"https://www.nfl.com/draft/tracker/prospects/all-positions/all-colleges/all-statuses/{YEAR}?page=1",
                wait_until="load",
                timeout=30_000,
            )
            page.wait_for_timeout(6_000)
        except Exception as exc:
            print(f"    (page load note: {exc})")
        browser.close()

    if not token:
        raise RuntimeError(
            "Could not capture JWT from NFL.com. The 2026 draft page or token flow may not be live yet."
        )
    print(f"  JWT captured (first 40 chars): {token[:40]}…")
    return token


SESSION = requests.Session()
SESSION.headers.update(
    {
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nfl.com/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }
)

_current_jwt = ""


def _set_jwt(token: str) -> None:
    global _current_jwt
    _current_jwt = token
    SESSION.headers["Authorization"] = token


def api_get(path: str, params: dict | None = None, retry: bool = True) -> dict | list:
    url = API_BASE + path
    response = SESSION.get(url, params=params, timeout=20)
    if response.status_code == 401 and retry:
        print("  401 — refreshing JWT...")
        _set_jwt(get_jwt())
        return api_get(path, params, retry=False)
    response.raise_for_status()
    return response.json()


def build_team_lookup() -> dict:
    try:
        data = api_get("/experience/v1/teams", {"season": YEAR})
        teams = data.get("teams", [])
        return {team["id"]: team.get("abbreviation", "") for team in teams if "id" in team}
    except Exception as exc:
        print(f"  Team lookup failed ({exc}), storing raw teamId instead.")
        return {}


def fetch_combine_profiles() -> dict:
    data = api_get("/football/v2/combine/profiles", {"limit": 1000, "year": YEAR})
    profiles = data.get("combineProfiles", [])
    return {profile["person"]["id"]: profile for profile in profiles}


def fetch_draft_picks() -> dict:
    try:
        data = api_get("/football/v2/draft/picks/report", {"limit": 1000, "year": YEAR})
        picks = data.get("picks", [])
        return {pick["personId"]: pick for pick in picks if "personId" in pick}
    except Exception as exc:
        print(f"  Draft picks unavailable for {YEAR}: {exc}")
        return {}


def build_row(profile: dict, pick: dict | None, team_lookup: dict) -> dict:
    person = profile.get("person", {})

    row = {
        "year": YEAR,
        "person_id": person.get("id", ""),
        "esb_id": person.get("esbId", ""),
        "first_name": person.get("firstName", ""),
        "last_name": person.get("lastName", ""),
        "display_name": person.get("displayName", ""),
        "hometown": person.get("hometown", ""),
        "college": ", ".join(person.get("collegeNames", [])),
        "college_class": profile.get("collegeClass", ""),
        "position": profile.get("position", ""),
        "position_group": profile.get("positionGroup", ""),
        "height_in": profile.get("height"),
        "weight_lbs": profile.get("weight"),
        "arm_length": profile.get("armLength"),
        "hand_size": profile.get("handSize"),
        "forty_yard_dash": profile.get("fortyYardDash"),
        "ten_yard_split": profile.get("tenYardSplit"),
        "twenty_yard_shuttle": profile.get("twentyYardShuttle"),
        "sixty_yard_shuttle": profile.get("sixtyYardShuttle"),
        "three_cone": profile.get("threeConeDrill"),
        "bench_press": profile.get("benchPress"),
        "broad_jump": profile.get("broadJump"),
        "vertical_jump": profile.get("verticalJump"),
        "grade": profile.get("grade"),
        "draft_grade": profile.get("draftGrade"),
        "draft_projection": profile.get("draftProjection"),
        "athleticism_score": profile.get("athleticismScore"),
        "production_score": profile.get("productionScore"),
        "size_score": profile.get("sizeScore"),
        "nfl_comparison": strip_html(profile.get("nflComparison", "")),
        "overview": strip_html(profile.get("overview", "")),
        "strengths": strip_html(profile.get("strengths", "")),
        "weaknesses": strip_html(profile.get("weaknesses", "")),
        "sources_tell_us": strip_html(profile.get("sourcesTellUs", "")),
        "bio": strip_html(profile.get("bio", "")),
        "profile_author": profile.get("profileAuthor", ""),
        "combine_attendance": profile.get("combineAttendance", ""),
        "draft_round": "",
        "draft_pick": "",
        "draft_overall": "",
        "draft_team_id": "",
        "draft_team_abbr": "",
        "headshot_url": (profile.get("headshot") or "").replace("{formatInstructions}/", ""),
    }

    if pick:
        team_id = pick.get("teamId", "")
        row.update(
            {
                "draft_round": pick.get("draftRound", ""),
                "draft_pick": pick.get("draftPosition", ""),
                "draft_overall": pick.get("draftNumberOverall", ""),
                "draft_team_id": team_id,
                "draft_team_abbr": team_lookup.get(team_id, ""),
            }
        )

    return row


def main() -> None:
    print(f"Step 1: Capturing JWT from NFL.com for {YEAR}...")
    _set_jwt(get_jwt())

    print("\nStep 2: Building team lookup...")
    team_lookup = build_team_lookup()
    print(f"  Teams loaded: {len(team_lookup)}")

    print(f"\nStep 3: Fetching combine profiles for {YEAR}...")
    profiles = fetch_combine_profiles()
    print(f"  → {len(profiles)} profiles")

    time.sleep(REQUEST_DELAY)

    print(f"\nStep 4: Fetching draft picks for {YEAR}...")
    picks = fetch_draft_picks()
    print(f"  → {len(picks)} picks")

    rows = []
    for person_id, profile in profiles.items():
        rows.append(build_row(profile, picks.get(person_id), team_lookup))

    print(f"\nTotal rows: {len(rows)}")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
