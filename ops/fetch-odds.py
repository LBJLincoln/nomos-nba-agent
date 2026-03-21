#!/usr/bin/env python3
"""
NBA Odds Fetcher — Multi-source (Bovada → DraftKings → TheRundown → odds-api.io → The Odds API)
================================================================================================
Fetches live NBA odds and saves to data/odds/ as JSON snapshots.

Sources:
  1. Bovada public API       (FREE, no key needed)
  2. DraftKings public API   (FREE, no key needed — Nash mobile API)
  3. TheRundown              (FREE tier via RapidAPI key — 20k points/day free)
                              Set THERUNDOWN_API_KEY in .env.local
  4. odds-api.io             (FREE tier — 100 req/hr, register at odds-api.io)
                              Set ODDS_API_IO_KEY in .env.local
  5. The Odds API            (needs ODDS_API_KEY with credits)

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
THERUNDOWN_API_KEY = os.environ.get("THERUNDOWN_API_KEY", "")  # Free tier at therundown.io
ODDS_API_IO_KEY = os.environ.get("ODDS_API_IO_KEY", "")        # Free tier at odds-api.io

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


def fetch_draftkings():
    """Fetch NBA odds from DraftKings public Nash API (FREE, no key needed).

    Uses DraftKings' undocumented mobile sportsbook API (sportsbook-nash.draftkings.com).
    The .json suffix with a mobile User-Agent bypasses the JS challenge on the main
    sportsbook.draftkings.com domain. NBA league ID = 42648.
    """
    url = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1/leagues/42648.json"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "DraftKings/15.2 iOS/17.0",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
    )
    resp = urllib.request.urlopen(req, timeout=15)
    raw = json.loads(resp.read().decode())

    # Index selections by marketId
    sel_by_market = {}
    for sel in raw.get("selections", []):
        mid = sel.get("marketId", "")
        sel_by_market.setdefault(mid, []).append(sel)

    # Index events by id
    events = {e["id"]: e for e in raw.get("events", [])}

    # Group markets by eventId
    mkt_by_event = {}
    for mkt in raw.get("markets", []):
        eid = mkt.get("eventId", "")
        mkt_by_event.setdefault(eid, []).append(mkt)

    # DK uses short team names like "WAS Wizards"; map to full names where known
    _DK_TEAM_MAP = {
        "WAS Wizards": "Washington Wizards", "OKC Thunder": "Oklahoma City Thunder",
        "LA Lakers": "Los Angeles Lakers",   "LA Clippers": "Los Angeles Clippers",
        "GS Warriors": "Golden State Warriors", "NO Pelicans": "New Orleans Pelicans",
        "CLE Cavaliers": "Cleveland Cavaliers", "MEM Grizzlies": "Memphis Grizzlies",
        "SA Spurs": "San Antonio Spurs",     "MIA Heat": "Miami Heat",
        "PHI 76ers": "Philadelphia 76ers",   "MIL Bucks": "Milwaukee Bucks",
        "PHO Suns": "Phoenix Suns",          "CHA Hornets": "Charlotte Hornets",
        "IND Pacers": "Indiana Pacers",      "ORL Magic": "Orlando Magic",
        "ATL Hawks": "Atlanta Hawks",        "DET Pistons": "Detroit Pistons",
        "BKN Nets": "Brooklyn Nets",         "NYK Knicks": "New York Knicks",
        "BOS Celtics": "Boston Celtics",     "TOR Raptors": "Toronto Raptors",
        "DEN Nuggets": "Denver Nuggets",     "MIN Timberwolves": "Minnesota Timberwolves",
        "POR Trail Blazers": "Portland Trail Blazers", "UTA Jazz": "Utah Jazz",
        "SAC Kings": "Sacramento Kings",     "HOU Rockets": "Houston Rockets",
        "DAL Mavericks": "Dallas Mavericks", "CHI Bulls": "Chicago Bulls",
    }

    games = []
    for eid, ev in events.items():
        parts = ev.get("participants", [])
        home_p = next((p for p in parts if p.get("venueRole") == "Home"), {})
        away_p = next((p for p in parts if p.get("venueRole") == "Away"), {})
        _home_raw = home_p.get("name", "")
        _away_raw = away_p.get("name", "")
        home_team = _DK_TEAM_MAP.get(_home_raw, _home_raw)
        away_team = _DK_TEAM_MAP.get(_away_raw, _away_raw)

        game = {
            "home_team": home_team,
            "away_team": away_team,
            "home_abbr": home_p.get("metadata", {}).get("shortName", ""),
            "away_abbr": away_p.get("metadata", {}).get("shortName", ""),
            "start_time": ev.get("startEventDate", ""),
            "source": "draftkings",
            "markets": {},
        }

        for mkt in mkt_by_event.get(eid, []):
            mkt_id = mkt.get("id", "")
            mkt_name = mkt.get("name", "").lower()
            sels = sel_by_market.get(mkt_id, [])
            outcomes = []
            for sel in sels:
                display_odds = sel.get("displayOdds", {})
                # DK uses Unicode minus (U+2212) — normalize to ASCII hyphen
                american = display_odds.get("american", "").replace("\u2212", "-")
                outcomes.append({
                    "name": sel.get("label", ""),
                    "american": american,
                    "decimal": str(sel.get("trueOdds", "")),
                    "handicap": str(sel.get("points", "")) if sel.get("points") != "" else "",
                })
            if "moneyline" in mkt_name:
                game["markets"]["moneyline"] = outcomes
            elif "spread" in mkt_name:
                game["markets"]["spread"] = outcomes
            elif "total" in mkt_name:
                game["markets"]["total"] = outcomes

        games.append(game)
    return games


def fetch_therundown():
    """Fetch NBA odds from TheRundown via RapidAPI (FREE tier — 20k data points/day).

    Requires THERUNDOWN_API_KEY in .env.local (free registration at therundown.io).
    NBA sport ID = 4. Bookmaker ID 123 = Pinnacle (main sharp line).
    Free tier docs: https://rapidapi.com/therundown/api/therundown
    """
    if not THERUNDOWN_API_KEY:
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = f"https://therundown-therundown-v1.p.rapidapi.com/sports/4/events/{today}?offset=0&include=scores"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "x-rapidapi-host": "therundown-therundown-v1.p.rapidapi.com",
            "x-rapidapi-key": THERUNDOWN_API_KEY,
        },
    )
    resp = urllib.request.urlopen(req, timeout=20)
    raw = json.loads(resp.read().decode())

    games = []
    for ev in raw.get("events", []):
        teams = ev.get("teams", {})
        away_info = teams.get("away", {})
        home_info = teams.get("home", {})
        game = {
            "home_team": home_info.get("name", ""),
            "away_team": away_info.get("name", ""),
            "home_abbr": home_info.get("abbreviation", ""),
            "away_abbr": away_info.get("abbreviation", ""),
            "start_time": ev.get("event_date", ""),
            "source": "therundown",
            "markets": {},
        }
        # Lines are keyed by bookmaker_id. Take the first available book.
        lines = ev.get("lines", {})
        book_id = next(iter(lines), None)
        if book_id:
            line = lines[book_id]
            ml = line.get("moneyline", {})
            sp = line.get("spread", {})
            tot = line.get("total", {})
            if ml.get("moneyline_away") and ml.get("moneyline_home"):
                game["markets"]["moneyline"] = [
                    {"name": away_info.get("name", "Away"), "american": str(int(ml["moneyline_away"]))},
                    {"name": home_info.get("name", "Home"), "american": str(int(ml["moneyline_home"]))},
                ]
            if sp.get("point_spread_away") is not None:
                game["markets"]["spread"] = [
                    {"name": away_info.get("name", "Away"),
                     "american": str(int(sp.get("spread_away", -110))),
                     "handicap": str(sp["point_spread_away"])},
                    {"name": home_info.get("name", "Home"),
                     "american": str(int(sp.get("spread_home", -110))),
                     "handicap": str(sp.get("point_spread_home", -sp["point_spread_away"]))},
                ]
            if tot.get("total_over") is not None:
                game["markets"]["total"] = [
                    {"name": "Over",
                     "american": str(int(tot.get("total_over_total", -110))),
                     "handicap": str(tot["total_over"])},
                    {"name": "Under",
                     "american": str(int(tot.get("total_under_total", -110))),
                     "handicap": str(tot.get("total_under", tot["total_over"]))},
                ]
        games.append(game)
    return games


def fetch_odds_api_io():
    """Fetch NBA odds from odds-api.io (FREE tier — 100 req/hr, no credit card needed).

    Requires ODDS_API_IO_KEY in .env.local (free registration at odds-api.io).
    Free tier: 100 requests/hour, 2 bookmakers.
    Docs: https://odds-api.io/docs
    """
    if not ODDS_API_IO_KEY:
        return []

    url = (
        "https://api.odds-api.io/v3/odds"
        f"?sport=basketball&league=nba&apiKey={ODDS_API_IO_KEY}"
        "&markets=moneyline,spread,total&oddsFormat=american"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )
    resp = urllib.request.urlopen(req, timeout=20)
    raw = json.loads(resp.read().decode())

    # odds-api.io returns a list of event objects
    if isinstance(raw, dict) and "error" in raw:
        print(f"[odds-api.io] Error: {raw.get('error', 'unknown')}")
        return []

    games = []
    for ev in (raw if isinstance(raw, list) else raw.get("data", [])):
        game = {
            "home_team": ev.get("home_team", ev.get("homeTeam", "")),
            "away_team": ev.get("away_team", ev.get("awayTeam", "")),
            "start_time": ev.get("commence_time", ev.get("startTime", "")),
            "source": "odds-api-io",
            "markets": {},
        }
        # Normalize odds from first available bookmaker
        bookmakers = ev.get("bookmakers", ev.get("books", []))
        for bk in bookmakers[:1]:
            for mkt in bk.get("markets", bk.get("odds", [])):
                key = mkt.get("key", mkt.get("type", "")).lower()
                outcomes = []
                for o in mkt.get("outcomes", []):
                    outcomes.append({
                        "name": o.get("name", o.get("label", "")),
                        "american": str(o.get("price", o.get("american", ""))),
                        "handicap": str(o.get("point", o.get("handicap", ""))),
                    })
                if key in ("spreads", "spread"):
                    game["markets"]["spread"] = outcomes
                elif key in ("h2h", "moneyline"):
                    game["markets"]["moneyline"] = outcomes
                elif key in ("totals", "total"):
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
    """Fetch from all sources, cascading Bovada → DraftKings → TheRundown → odds-api.io → The Odds API."""
    # 1. Try Bovada first (free, reliable)
    try:
        games = fetch_bovada()
        if games:
            print(f"[Bovada] {len(games)} games fetched")
            return games, "bovada"
    except Exception as e:
        print(f"[Bovada] Failed: {e}")

    # 2. DraftKings public Nash API (free, no key)
    try:
        games = fetch_draftkings()
        if games:
            print(f"[DraftKings] {len(games)} games fetched")
            return games, "draftkings"
    except Exception as e:
        print(f"[DraftKings] Failed: {e}")

    # 3. TheRundown (free tier, needs THERUNDOWN_API_KEY)
    try:
        games = fetch_therundown()
        if games:
            print(f"[TheRundown] {len(games)} games fetched")
            return games, "therundown"
    except Exception as e:
        print(f"[TheRundown] Failed: {e}")

    # 4. odds-api.io (free tier, needs ODDS_API_IO_KEY)
    try:
        games = fetch_odds_api_io()
        if games:
            print(f"[odds-api.io] {len(games)} games fetched")
            return games, "odds-api-io"
    except Exception as e:
        print(f"[odds-api.io] Failed: {e}")

    # 5. Fallback to The Odds API (paid)
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
