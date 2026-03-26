#!/usr/bin/env python3
"""
NBA Odds Scraper -- Browser-based fallback for when APIs fail.
==============================================================

Uses browser_scraper.py to scrape odds from web pages that don't have
public APIs (or whose APIs are blocked). This is a fallback layer on
top of the existing fetch-odds.py API-based approach.

Sources:
  1. DraftKings NBA sportsbook page (HTML scrape)
  2. ESPN NBA scoreboard (game data + basic lines)

Output format matches fetch-odds.py exactly:
{
    "source": "draftkings-scrape",
    "timestamp": "2026-03-26T15:30:00Z",
    "games": [
        {
            "home_team": "Boston Celtics",
            "away_team": "Los Angeles Lakers",
            "start_time": "...",
            "source": "draftkings-scrape",
            "markets": {
                "spread": [{"name": "BOS", "american": "-110", "handicap": "-6.5"}, ...],
                "moneyline": [{"name": "BOS", "american": "-250"}, ...],
                "total": [{"name": "Over", "american": "-110", "handicap": "228.5"}, ...]
            }
        }
    ]
}

Usage:
    python3 ops/scrape_odds.py                   # Scrape and print
    python3 ops/scrape_odds.py --save             # Scrape and save to data/odds/
    python3 ops/scrape_odds.py --source espn      # Specific source
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ODDS_DIR = DATA_DIR / "odds"
ODDS_DIR.mkdir(parents=True, exist_ok=True)

# Import the browser scraper (same directory)
sys.path.insert(0, str(BASE_DIR))
from ops.browser_scraper import scrape_sync

# Team name mapping (reuse from fetch-odds.py)
TEAM_ABBREV = {
    "Oklahoma City Thunder": "OKC", "Boston Celtics": "BOS", "Cleveland Cavaliers": "CLE",
    "New York Knicks": "NYK", "Denver Nuggets": "DEN", "Milwaukee Bucks": "MIL",
    "Phoenix Suns": "PHX", "Golden State Warriors": "GSW", "Los Angeles Lakers": "LAL",
    "Dallas Mavericks": "DAL", "Memphis Grizzlies": "MEM", "Sacramento Kings": "SAC",
    "Houston Rockets": "HOU", "Minnesota Timberwolves": "MIN", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Miami Heat": "MIA", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Atlanta Hawks": "ATL", "Chicago Bulls": "CHI",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Detroit Pistons": "DET",
    "Brooklyn Nets": "BKN", "New Orleans Pelicans": "NOP", "Charlotte Hornets": "CHA",
    "Portland Trail Blazers": "POR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# Reverse mapping: abbreviation -> full name
ABBREV_TO_TEAM = {v: k for k, v in TEAM_ABBREV.items()}

# Common short names seen on sportsbook pages
SHORT_NAME_MAP = {
    "Thunder": "Oklahoma City Thunder", "Celtics": "Boston Celtics",
    "Cavaliers": "Cleveland Cavaliers", "Knicks": "New York Knicks",
    "Nuggets": "Denver Nuggets", "Bucks": "Milwaukee Bucks",
    "Suns": "Phoenix Suns", "Warriors": "Golden State Warriors",
    "Lakers": "Los Angeles Lakers", "Mavericks": "Dallas Mavericks",
    "Grizzlies": "Memphis Grizzlies", "Kings": "Sacramento Kings",
    "Rockets": "Houston Rockets", "Timberwolves": "Minnesota Timberwolves",
    "Pacers": "Indiana Pacers", "Clippers": "Los Angeles Clippers",
    "Heat": "Miami Heat", "Magic": "Orlando Magic",
    "76ers": "Philadelphia 76ers", "Sixers": "Philadelphia 76ers",
    "Hawks": "Atlanta Hawks", "Bulls": "Chicago Bulls",
    "Spurs": "San Antonio Spurs", "Raptors": "Toronto Raptors",
    "Pistons": "Detroit Pistons", "Nets": "Brooklyn Nets",
    "Pelicans": "New Orleans Pelicans", "Hornets": "Charlotte Hornets",
    "Trail Blazers": "Portland Trail Blazers", "Blazers": "Portland Trail Blazers",
    "Jazz": "Utah Jazz", "Wizards": "Washington Wizards",
}


def _resolve_team_name(name: str) -> str:
    """Resolve a team name to its full canonical form."""
    name = name.strip()
    # Already a full name
    if name in TEAM_ABBREV:
        return name
    # Abbreviation
    if name.upper() in ABBREV_TO_TEAM:
        return ABBREV_TO_TEAM[name.upper()]
    # Short name (e.g., "Lakers", "76ers")
    for short, full in SHORT_NAME_MAP.items():
        if short.lower() in name.lower():
            return full
    return name


def _parse_american_odds(text: str) -> str:
    """Clean and normalize american odds string."""
    text = text.strip()
    # Normalize Unicode minus to ASCII
    text = text.replace("\u2212", "-").replace("\u2013", "-")
    # Remove any non-numeric chars except +/-
    text = re.sub(r"[^\d\+\-]", "", text)
    if text and text[0].isdigit():
        text = "+" + text
    return text


def scrape_draftkings_odds() -> list[dict]:
    """
    Scrape NBA odds from DraftKings sportsbook page.

    This is a browser-based fallback for when the Nash mobile API fails.
    Scrapes the public sportsbook page and parses odds from the markdown.

    Returns list of games in standard format.
    """
    url = "https://sportsbook.draftkings.com/leagues/basketball/nba"

    # Use CSS selectors to target the odds table structure
    selectors = {
        "event_cards": "div.sportsbook-event-accordion__wrapper",
        "odds_rows": "tr.sportsbook-table__body-row",
        "team_names": "span.event-cell__name-text",
        "odds_cells": "span.sportsbook-odds",
    }

    result = scrape_sync(url, selectors=selectors)

    if not result["success"]:
        print(f"[DK Scrape] Failed: {result['error']}")
        return []

    games = _parse_draftkings_markdown(result["markdown"], result.get("extracted", {}))
    return games


def _parse_draftkings_markdown(markdown: str, extracted: dict) -> list[dict]:
    """
    Parse DraftKings page markdown/extracted data into standard game format.

    The markdown from DK typically contains lines like:
        Team Name
        +150  -3.5  O 220.5
    or similar patterns. This parser attempts to find game blocks.

    This is intentionally flexible -- DK changes their HTML frequently.
    """
    games = []

    # Strategy 1: Try extracted CSS selector data first
    team_names = extracted.get("team_names")
    odds_data = extracted.get("odds_cells")

    if team_names and isinstance(team_names, list) and len(team_names) >= 2:
        # Pair teams (away, home) and associate odds
        for i in range(0, len(team_names) - 1, 2):
            away_raw = team_names[i]
            home_raw = team_names[i + 1]
            away_team = _resolve_team_name(away_raw)
            home_team = _resolve_team_name(home_raw)

            game = {
                "home_team": home_team,
                "away_team": away_team,
                "home_abbr": TEAM_ABBREV.get(home_team, ""),
                "away_abbr": TEAM_ABBREV.get(away_team, ""),
                "start_time": "",
                "source": "draftkings-scrape",
                "markets": {},
            }

            # Try to associate odds if we have them
            if odds_data and isinstance(odds_data, list):
                # DK typically has 6 odds per game: spread(2), total(2), moneyline(2)
                odds_start = i * 3  # 3 odds per team row
                if odds_start + 5 < len(odds_data):
                    away_spread_odds = _parse_american_odds(odds_data[odds_start])
                    away_spread_val = odds_data[odds_start + 1] if odds_start + 1 < len(odds_data) else ""
                    away_ml = _parse_american_odds(odds_data[odds_start + 2]) if odds_start + 2 < len(odds_data) else ""
                    home_spread_odds = _parse_american_odds(odds_data[odds_start + 3]) if odds_start + 3 < len(odds_data) else ""
                    home_spread_val = odds_data[odds_start + 4] if odds_start + 4 < len(odds_data) else ""
                    home_ml = _parse_american_odds(odds_data[odds_start + 5]) if odds_start + 5 < len(odds_data) else ""

                    if away_ml or home_ml:
                        game["markets"]["moneyline"] = [
                            {"name": away_team, "american": away_ml},
                            {"name": home_team, "american": home_ml},
                        ]

            games.append(game)

        if games:
            return games

    # Strategy 2: Parse markdown text with regex patterns
    # Look for game-like patterns: "TeamA vs TeamB" or "TeamA @ TeamB"
    lines = markdown.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Pattern: "TeamA @ TeamB" or "TeamA vs TeamB"
        match = re.search(r'(.+?)\s+[@vs\.]+\s+(.+?)(?:\s*$|\s*\|)', line)
        if match:
            away_raw, home_raw = match.group(1).strip(), match.group(2).strip()
            away_team = _resolve_team_name(away_raw)
            home_team = _resolve_team_name(home_raw)

            game = {
                "home_team": home_team,
                "away_team": away_team,
                "home_abbr": TEAM_ABBREV.get(home_team, ""),
                "away_abbr": TEAM_ABBREV.get(away_team, ""),
                "start_time": "",
                "source": "draftkings-scrape",
                "markets": {},
            }

            # Look ahead for odds on the next few lines
            for j in range(i + 1, min(i + 6, len(lines))):
                odds_line = lines[j].strip()
                # Moneyline pattern: +150 or -200
                ml_matches = re.findall(r'([+-]\d{3,4})', odds_line)
                # Spread pattern: -6.5 or +3.5
                spread_matches = re.findall(r'([+-]\d+\.5)', odds_line)
                # Total pattern: O/U 220.5 or Over 220.5
                total_match = re.search(r'(?:O|Over|U|Under)\s*(\d{3}\.?\d?)', odds_line, re.IGNORECASE)

                if ml_matches and len(ml_matches) >= 2 and "moneyline" not in game["markets"]:
                    game["markets"]["moneyline"] = [
                        {"name": away_team, "american": ml_matches[0]},
                        {"name": home_team, "american": ml_matches[1]},
                    ]
                if spread_matches and "spread" not in game["markets"]:
                    game["markets"]["spread"] = [
                        {"name": away_team, "american": "-110", "handicap": spread_matches[0]},
                    ]
                    if len(spread_matches) >= 2:
                        game["markets"]["spread"].append(
                            {"name": home_team, "american": "-110", "handicap": spread_matches[1]}
                        )
                if total_match and "total" not in game["markets"]:
                    game["markets"]["total"] = [
                        {"name": "Over", "american": "-110", "handicap": total_match.group(1)},
                        {"name": "Under", "american": "-110", "handicap": total_match.group(1)},
                    ]

            games.append(game)
        i += 1

    return games


def scrape_espn_scoreboard() -> list[dict]:
    """
    Scrape ESPN NBA scoreboard for game data.

    ESPN doesn't always show odds, but it provides game schedules and scores
    which can be combined with other odds sources.

    Returns list of games in standard format.
    """
    url = "https://www.espn.com/nba/scoreboard"

    selectors = {
        "game_strips": "section.Scoreboard",
        "team_names": "div.ScoreCell__TeamName",
        "scores": "div.ScoreCell__Score",
    }

    result = scrape_sync(url, selectors=selectors)

    if not result["success"]:
        print(f"[ESPN Scrape] Failed: {result['error']}")
        return []

    games = _parse_espn_markdown(result["markdown"], result.get("extracted", {}))
    return games


def _parse_espn_markdown(markdown: str, extracted: dict) -> list[dict]:
    """Parse ESPN scoreboard markdown into game format."""
    games = []

    # Strategy 1: CSS selector extraction
    team_names = extracted.get("team_names")
    if team_names and isinstance(team_names, list) and len(team_names) >= 2:
        for i in range(0, len(team_names) - 1, 2):
            away_team = _resolve_team_name(team_names[i])
            home_team = _resolve_team_name(team_names[i + 1])

            game = {
                "home_team": home_team,
                "away_team": away_team,
                "home_abbr": TEAM_ABBREV.get(home_team, ""),
                "away_abbr": TEAM_ABBREV.get(away_team, ""),
                "start_time": "",
                "source": "espn-scrape",
                "markets": {},  # ESPN may not have odds
            }
            games.append(game)

        if games:
            return games

    # Strategy 2: Parse markdown for team name patterns
    # ESPN markdown typically has team names in recognizable patterns
    found_teams = []
    for line in markdown.split("\n"):
        line = line.strip()
        for short_name, full_name in SHORT_NAME_MAP.items():
            if short_name in line and full_name not in [t for t in found_teams]:
                found_teams.append(full_name)
                break

    # Pair teams into games (away, home)
    for i in range(0, len(found_teams) - 1, 2):
        game = {
            "home_team": found_teams[i + 1],
            "away_team": found_teams[i],
            "home_abbr": TEAM_ABBREV.get(found_teams[i + 1], ""),
            "away_abbr": TEAM_ABBREV.get(found_teams[i], ""),
            "start_time": "",
            "source": "espn-scrape",
            "markets": {},
        }
        games.append(game)

    return games


def scrape_all() -> tuple[list[dict], str]:
    """
    Scrape odds from all browser-based sources.
    Cascade: DraftKings scrape -> ESPN scrape.

    Returns (games, source_name).
    """
    # 1. DraftKings page scrape
    try:
        games = scrape_draftkings_odds()
        if games:
            # Only count games that have at least one market
            games_with_odds = [g for g in games if g.get("markets")]
            if games_with_odds:
                print(f"[DK Scrape] {len(games_with_odds)} games with odds ({len(games)} total)")
                return games_with_odds, "draftkings-scrape"
            elif games:
                print(f"[DK Scrape] {len(games)} games found but no odds parsed")
    except Exception as e:
        print(f"[DK Scrape] Failed: {e}")

    # 2. ESPN scrape (games only, rarely has odds)
    try:
        games = scrape_espn_scoreboard()
        if games:
            print(f"[ESPN Scrape] {len(games)} games found (schedule only, no odds)")
            return games, "espn-scrape"
    except Exception as e:
        print(f"[ESPN Scrape] Failed: {e}")

    return [], "none"


def save_snapshot(games: list[dict], source: str) -> str:
    """Save odds snapshot to data/odds/ in the same format as fetch-odds.py."""
    ts = datetime.now(timezone.utc)
    snapshot = {
        "source": source,
        "timestamp": ts.isoformat(),
        "n_games": len(games),
        "games": games,
    }
    fname = f"scrape-{ts.strftime('%Y%m%d-%H%M')}.json"
    (ODDS_DIR / fname).write_text(json.dumps(snapshot, indent=2))
    print(f"[Save] {fname} ({len(games)} games)")
    return fname


def print_odds(games: list[dict]) -> None:
    """Pretty print scraped odds (same format as fetch-odds.py)."""
    print(f"\n{'='*80}")
    print(f"  NBA SCRAPED ODDS -- {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*80}")
    print(f"{'Game':<45} {'Spread':>10} {'ML Home':>10} {'O/U':>10}")
    print(f"{'-'*80}")
    for g in games:
        desc = f"{g.get('away_team', '?')[:20]} @ {g.get('home_team', '?')[:20]}"
        spread = ""
        ml = ""
        total = ""
        mkts = g.get("markets", {})
        if mkts.get("spread"):
            for oc in mkts["spread"]:
                handicap = oc.get("handicap", "")
                if handicap and float(handicap) < 0:
                    spread = f"{handicap} ({oc.get('american', '')})"
                    break
        if mkts.get("moneyline"):
            for oc in mkts["moneyline"]:
                if g.get("home_team", "") in oc.get("name", ""):
                    ml = oc.get("american", "")
                    break
            if not ml and len(mkts["moneyline"]) >= 2:
                ml = mkts["moneyline"][1].get("american", "")
        if mkts.get("total"):
            for oc in mkts["total"]:
                if oc.get("handicap"):
                    total = str(oc["handicap"])
                    break
        print(f"{desc:<45} {spread:>10} {ml:>10} {total:>10}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NBA Odds Scraper (browser-based)")
    parser.add_argument("--save", action="store_true", help="Save snapshot to data/odds/")
    parser.add_argument("--source", choices=["draftkings", "espn", "all"], default="all",
                        help="Which source to scrape (default: all)")
    args = parser.parse_args()

    if args.source == "draftkings":
        games = scrape_draftkings_odds()
        source = "draftkings-scrape"
    elif args.source == "espn":
        games = scrape_espn_scoreboard()
        source = "espn-scrape"
    else:
        games, source = scrape_all()

    if games:
        print_odds(games)
        if args.save:
            save_snapshot(games, source)
    else:
        print("No odds data scraped from any source.")
