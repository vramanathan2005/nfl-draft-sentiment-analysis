"""
Scrape NFL.com draft pick data for 2014-2025.

Data source (api.nfl.com — requires Bearer JWT captured from browser session):
  /football/v2/draft/picks/report?limit=1000&year={year}
      -> draft round/pick/overall + team + player identity

Strategy:
  1. Launch headless Chromium via Playwright, load an NFL.com draft page, capture JWT.
  2. Use requests + that JWT to call the draft picks API for each year.
  3. Refresh the JWT if a 401 is returned.
  4. Write one CSV row per draft pick.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright


OUTPUT_CSV = Path("nfl_draft_picks_2014_2025.csv")
YEARS = list(range(2014, 2026))
API_BASE = "https://api.nfl.com"
REQUEST_DELAY = 0.5

CSV_FIELDS = [
    "year",
    "person_id",
    "esb_id",
    "first_name",
    "last_name",
    "display_name",
    "college",
    "position",
    "draft_round",
    "draft_pick",
    "draft_overall",
    "team_id",
    "team_abbr",
]


def get_jwt() -> str:
    """Load an NFL.com draft page with Playwright and capture the Bearer JWT."""
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
                "https://www.nfl.com/draft/tracker/picks/all-trades/2025?page=1",
                wait_until="load",
                timeout=30_000,
            )
            page.wait_for_timeout(6_000)
        except Exception as exc:
            print(f"    (page load note: {exc})")
        browser.close()

    if not token:
        raise RuntimeError("Could not capture JWT from NFL.com — check network/VPN.")
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
    """GET from api.nfl.com, refreshing JWT once on 401."""
    url = API_BASE + path
    response = SESSION.get(url, params=params, timeout=20)
    if response.status_code == 401 and retry:
        print("  401 — refreshing JWT...")
        _set_jwt(get_jwt())
        return api_get(path, params, retry=False)
    response.raise_for_status()
    return response.json()


def build_team_lookup() -> dict:
    """Return {teamId: abbreviation} for draft rows."""
    try:
        data = api_get("/experience/v1/teams", {"season": 2025})
        teams = data.get("teams", [])
        return {team["id"]: team.get("abbreviation", "") for team in teams if "id" in team}
    except Exception as exc:
        print(f"  Team lookup failed ({exc}), storing raw teamId instead.")
        return {}


def fetch_draft_picks(year: int) -> list[dict]:
    """Return raw draft pick rows for a given year."""
    data = api_get("/football/v2/draft/picks/report", {"limit": 1000, "year": year})
    return data.get("picks", [])


def build_row(year: int, pick: dict, team_lookup: dict) -> dict:
    player = pick.get("player", {}) or {}
    team_id = pick.get("teamId", "")
    return {
        "year": year,
        "person_id": pick.get("personId", ""),
        "esb_id": player.get("esbId", ""),
        "first_name": player.get("firstName", ""),
        "last_name": player.get("lastName", ""),
        "display_name": player.get("displayName", ""),
        "college": ", ".join(player.get("collegeNames", [])) if player.get("collegeNames") else "",
        "position": player.get("position", ""),
        "draft_round": pick.get("draftRound", ""),
        "draft_pick": pick.get("draftPosition", ""),
        "draft_overall": pick.get("draftNumberOverall", ""),
        "team_id": team_id,
        "team_abbr": team_lookup.get(team_id, ""),
    }


def main() -> None:
    print("Step 1: Capturing JWT from NFL.com...")
    _set_jwt(get_jwt())

    print("\nStep 2: Building team lookup...")
    team_lookup = build_team_lookup()
    print(f"  Teams loaded: {len(team_lookup)}")

    all_rows = []

    for year in YEARS:
        print(f"\nYear {year}:")
        picks = fetch_draft_picks(year)
        print(f"  → {len(picks)} picks")
        all_rows.extend(build_row(year, pick, team_lookup) for pick in picks)
        time.sleep(REQUEST_DELAY)

    print(f"\nTotal rows: {len(all_rows)}")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
