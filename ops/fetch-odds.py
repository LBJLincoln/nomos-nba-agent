#!/usr/bin/env python3
"""
NBA Odds Fetcher — Multi-source (Bovada primary, The Odds API fallback)
========================================================================
Fetches live NBA odds and saves to data/odds/ as JSON snapshots.

Sources:
  1. Bovada public API (FREE, no key needed)
  2. The Odds API (needs ODDS_API_KEY with credits)

Usage:
    python3 ops/fetch-odds.py              # One-shot fetch
    python3 ops/fetch-odds.py --loop 300   # Fetch every 5 minutes

Output format (standardized across sources):
{
    "source": "bovada",
    "timestamp": "2026-03-21T15:30:00Z",
    "games": [
        {
            "home_team": "Boston Celtics",
            "away_team": "Los Angeles Lakers",
            "start_time": 1774134000000,
            "markets": {
                "spread": [{"name": "BOS", "american": "-110", "handicap": "-6.5"}, ...],
                "moneyline": [{"name": "BOS", "american": "-250", "decimal": "1.4"}, ...],
                "total": [{"name": "Over", "american": "-110", "handicap": "228.5"}, ...]
            }
        }
    ]
}
"""

import os, sys, json, time
from pathlib import Path
from datetime import datetime, timezone
import urllib.request

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ODDS_DIR = DATA_DIR / "odds"
ODDS_DIR.mkdir(parents=True, exist_ok=True)

# Load env
_env = BASE_DIR / ".env.local"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            line = line.replace("export ", "")
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip("'\""))

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

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


def fetch_bovada():
    """Fetch NBA odds from Bovada public API (FREE)."""
    url = "https://www.bovada.lv/services/sports/event/coupon/events/A/description/basketball/nba?marketFilterId=def&lang=en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
    resp = urllib.request.urlopen(req, timeout=15)
    raw = json.loads(resp.read().decode())

    games = []
    for group in raw:
        for ev in group.get("events", []):
            game = {
                "id": ev.get("id"),
                "description": ev.get("description", ""),
                "start_time": ev.get("startTime"),
                "source": "bovada",
                "markets": {},
            }
            parts = ev.get("description", "").split(" @ ")
            if len(parts) == 2:
                game["away_team"] = parts[0].strip()
                game["home_team"] = parts[1].strip()
                game["away_abbr"] = TEAM_ABBREV.get(game["away_team"], game["away_team"][:3].upper())
                game["home_abbr"] = TEAM_ABBREV.get(game["home_team"], game["home_team"][:3].upper())

            for dg in ev.get("displayGroups", []):
                for mkt in dg.get("markets", []):
                    mtype = mkt.get("description", "").lower()
                    outcomes = []
                    for oc in mkt.get("outcomes", []):
                        outcomes.append({
                            "name": oc.get("description", ""),
                            "american": oc.get("price", {}).get("american", ""),
                            "decimal": oc.get("price", {}).get("decimal", ""),
                            "handicap": oc.get("price", {}).get("handicap", ""),
                        })
                    if "spread" in mtype:
                        game["markets"]["spread"] = outcomes
                    elif "moneyline" in mtype or "money" in mtype:
                        game["markets"]["moneyline"] = outcomes
                    elif "total" in mtype:
                        game["markets"]["total"] = outcomes
            games.append(game)
    return games


def fetch_odds_api():
    """Fetch from The Odds API (needs credits)."""
    if not ODDS_API_KEY:
        return []
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h,spreads,totals&oddsFormat=american"
    resp = urllib.request.urlopen(url, timeout=30)
    raw = json.loads(resp.read().decode())
    if isinstance(raw, dict) and "error_code" in raw:
        print(f"[Odds API] Error: {raw.get('message', 'unknown')}")
        return []
    # Normalize to our format
    games = []
    for g in raw:
        game = {
            "home_team": g.get("home_team", ""),
            "away_team": g.get("away_team", ""),
            "start_time": g.get("commence_time", ""),
            "source": "the-odds-api",
            "markets": {},
        }
        for bk in g.get("bookmakers", [])[:1]:  # Take first book
            for mkt in bk.get("markets", []):
                key = mkt.get("key", "")
                outcomes = [{"name": o.get("name", ""), "american": str(o.get("price", "")),
                             "handicap": str(o.get("point", ""))} for o in mkt.get("outcomes", [])]
                if key == "spreads":
                    game["markets"]["spread"] = outcomes
                elif key == "h2h":
                    game["markets"]["moneyline"] = outcomes
                elif key == "totals":
                    game["markets"]["total"] = outcomes
        games.append(game)
    return games


def fetch_all():
    """Fetch from all sources, Bovada first."""
    # Try Bovada first (free, reliable)
    try:
        games = fetch_bovada()
        if games:
            print(f"[Bovada] {len(games)} games fetched")
            return games, "bovada"
    except Exception as e:
        print(f"[Bovada] Failed: {e}")

    # Fallback to The Odds API
    try:
        games = fetch_odds_api()
        if games:
            print(f"[Odds API] {len(games)} games fetched")
            return games, "odds-api"
    except Exception as e:
        print(f"[Odds API] Failed: {e}")

    return [], "none"


def save_snapshot(games, source):
    """Save odds snapshot to data/odds/."""
    ts = datetime.now(timezone.utc)
    snapshot = {
        "source": source,
        "timestamp": ts.isoformat(),
        "n_games": len(games),
        "games": games,
    }
    fname = f"snapshot-{ts.strftime('%Y%m%d-%H%M')}.json"
    (ODDS_DIR / fname).write_text(json.dumps(snapshot, indent=2))
    # Also save as latest
    (ODDS_DIR / "latest.json").write_text(json.dumps(snapshot, indent=2))
    print(f"[Save] {fname} ({len(games)} games)")
    return fname


def print_odds(games):
    """Pretty print odds table."""
    print(f"\n{'='*80}")
    print(f"  NBA ODDS — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
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
                if float(oc.get("handicap", 0) or 0) < 0:
                    spread = f"{oc['handicap']} ({oc['american']})"
                    break
        if mkts.get("moneyline"):
            for oc in mkts["moneyline"]:
                if g.get("home_team", "") in oc.get("name", ""):
                    ml = oc["american"]
                    break
            if not ml and len(mkts["moneyline"]) >= 2:
                ml = mkts["moneyline"][1]["american"]
        if mkts.get("total"):
            for oc in mkts["total"]:
                if oc.get("handicap"):
                    total = str(oc["handicap"])
                    break
        print(f"{desc:<45} {spread:>10} {ml:>10} {total:>10}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NBA Odds Fetcher")
    parser.add_argument("--loop", type=int, default=0, help="Loop interval in seconds (0=one-shot)")
    args = parser.parse_args()

    if args.loop > 0:
        print(f"Starting odds loop (every {args.loop}s)...")
        while True:
            try:
                games, source = fetch_all()
                if games:
                    save_snapshot(games, source)
                    print_odds(games)
            except Exception as e:
                print(f"[ERROR] {e}")
            time.sleep(args.loop)
    else:
        games, source = fetch_all()
        if games:
            save_snapshot(games, source)
            print_odds(games)
        else:
            print("No odds data available from any source.")
