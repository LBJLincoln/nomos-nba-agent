#!/usr/bin/env python3
"""
scrape_season_odds.py
=====================
Download complete 2025-26 NBA season moneyline odds.

Sources (tried in priority order):
1. Kaggle dataset: caseydurfee/mgm-grand-nba-betting-data
   - Coverage: Oct 2025 - Feb 12, 2026
   - Format: American moneylines, BetMGM
   - Download: via kaggle CLI or direct API

2. SportsBettingReview.com (SBR) scraper
   - Coverage: any date (historical archive)
   - Format: American moneylines (opening + closing), multiple books
   - URL: https://www.sportsbookreview.com/betting-odds/nba-basketball/money-line/?date=YYYY-MM-DD

3. Local odds snapshots (The Odds API)
   - Coverage: Mar 15-17, 2026 only (API quota exhausted after that)
   - Path: /home/termius/nomos-nba-agent/data/odds-*.json
   - Format: decimal moneylines, Pinnacle/FanDuel

Output: data/historical-odds/nba_2025-26_odds.csv
Columns: date, home_team, away_team, moneyline_home, moneyline_away, book, source

Usage:
    python3 scripts/scrape_season_odds.py            # full season
    python3 scripts/scrape_season_odds.py --from-date 2026-02-13  # partial
    python3 scripts/scrape_season_odds.py --source sbr  # SBR only

Notes:
    - Rate-limited to 2s between SBR requests to avoid ban
    - MGM dataset requires kaggle CLI configured (kaggle.json in ~/.kaggle)
    - The Odds API historical endpoint requires paid plan (not used here)
    - Closing lines from SBR are the last posted odds before game tip-off
"""

import os
import sys
import csv
import json
import time
import glob
import re
import argparse
import subprocess
import urllib.request
import urllib.error
from datetime import datetime, timedelta, date

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "historical-odds")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "nba_2025-26_odds.csv")

GAMES_FILE = os.path.join(BASE_DIR, "data", "historical", "games-2025-26.json")
MGM_LOCAL = os.path.join("/tmp", "mgm-nba", "all_odds.csv")

CSV_FIELDS = ["date", "home_team", "away_team", "moneyline_home", "moneyline_away",
              "spread_home", "total", "book", "source"]

# ─── Team name normalization ──────────────────────────────────────────────────

# SBR uses short/city names; normalize to full team names
SBR_NAME_MAP = {
    "Atlanta": "Atlanta Hawks",
    "Boston": "Boston Celtics",
    "Brooklyn": "Brooklyn Nets",
    "Charlotte": "Charlotte Hornets",
    "Chicago": "Chicago Bulls",
    "Cleveland": "Cleveland Cavaliers",
    "Dallas": "Dallas Mavericks",
    "Denver": "Denver Nuggets",
    "Detroit": "Detroit Pistons",
    "Golden State": "Golden State Warriors",
    "Golden St.": "Golden State Warriors",
    "Houston": "Houston Rockets",
    "Indiana": "Indiana Pacers",
    "LA Clippers": "Los Angeles Clippers",
    "L.A. Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
    "L.A. Lakers": "Los Angeles Lakers",
    "Los Angeles": "Los Angeles Lakers",  # fallback
    "Memphis": "Memphis Grizzlies",
    "Miami": "Miami Heat",
    "Milwaukee": "Milwaukee Bucks",
    "Minnesota": "Minnesota Timberwolves",
    "New Orleans": "New Orleans Pelicans",
    "New York": "New York Knicks",
    "Oklahoma City": "Oklahoma City Thunder",
    "Orlando": "Orlando Magic",
    "Philadelphia": "Philadelphia 76ers",
    "Phoenix": "Phoenix Suns",
    "Portland": "Portland Trail Blazers",
    "Sacramento": "Sacramento Kings",
    "San Antonio": "San Antonio Spurs",
    "Toronto": "Toronto Raptors",
    "Utah": "Utah Jazz",
    "Washington": "Washington Wizards",
    # Alternate forms SBR uses
    "Utah Jazz": "Utah Jazz",
    "Memphis Grizzlies": "Memphis Grizzlies",
    "Indiana Pacers": "Indiana Pacers",
    "Brooklyn Nets": "Brooklyn Nets",
    "San Antonio Spurs": "San Antonio Spurs",
}

MGM_NAME_MAP = {
    # Short city names used by MGM dataset
    "Atlanta": "Atlanta Hawks",
    "Boston": "Boston Celtics",
    "Brooklyn": "Brooklyn Nets",
    "Charlotte": "Charlotte Hornets",
    "Chicago": "Chicago Bulls",
    "Cleveland": "Cleveland Cavaliers",
    "Dallas": "Dallas Mavericks",
    "Denver": "Denver Nuggets",
    "Detroit": "Detroit Pistons",
    "Houston": "Houston Rockets",
    "Indiana": "Indiana Pacers",
    "Memphis": "Memphis Grizzlies",
    "Miami": "Miami Heat",
    "Milwaukee": "Milwaukee Bucks",
    "Minnesota": "Minnesota Timberwolves",
    "New Orleans": "New Orleans Pelicans",
    "New York": "New York Knicks",
    "Oklahoma City": "Oklahoma City Thunder",
    "Orlando": "Orlando Magic",
    "Philadelphia": "Philadelphia 76ers",
    "Phoenix": "Phoenix Suns",
    "Portland": "Portland Trail Blazers",
    "Sacramento": "Sacramento Kings",
    "San Antonio": "San Antonio Spurs",
    "Toronto": "Toronto Raptors",
    "Utah": "Utah Jazz",
    "Washington": "Washington Wizards",
    # MGM-specific short names
    "LA Lakers": "Los Angeles Lakers",
    "LA Clippers": "Los Angeles Clippers",
    "Golden State": "Golden State Warriors",
}


def normalize_team(name, name_map=None):
    if name_map is None:
        name_map = SBR_NAME_MAP
    return name_map.get(name, name)


# ─── Source 1: MGM Kaggle dataset ────────────────────────────────────────────

def download_mgm_kaggle():
    """Download MGM Grand NBA betting dataset from Kaggle."""
    os.makedirs("/tmp/mgm-nba", exist_ok=True)
    if os.path.exists(MGM_LOCAL):
        print("[MGM] Already downloaded at", MGM_LOCAL)
        return True

    print("[MGM] Downloading caseydurfee/mgm-grand-nba-betting-data from Kaggle...")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download",
             "caseydurfee/mgm-grand-nba-betting-data",
             "--path", "/tmp/mgm-nba/",
             "--unzip"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and os.path.exists(MGM_LOCAL):
            print("[MGM] Download successful.")
            return True
        else:
            print(f"[MGM] Download failed: {result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"[MGM] Error: {e}")
        return False


def extract_mgm_season(min_date="2025-10-01", max_date="2026-02-12"):
    """Extract 2025-26 season games from MGM dataset."""
    if not os.path.exists(MGM_LOCAL):
        if not download_mgm_kaggle():
            return []

    rows = []
    with open(MGM_LOCAL) as f:
        reader = csv.DictReader(f)
        for r in reader:
            gd = r.get("game_date", "")
            game_date = gd[:10] if gd else ""
            if not (min_date <= game_date <= max_date):
                continue

            home = normalize_team(r.get("home_team", "").strip(), MGM_NAME_MAP)
            away = normalize_team(r.get("away_team", "").strip(), MGM_NAME_MAP)

            ml_h = r.get("money_home_odds", "")
            ml_a = r.get("money_away_odds", "")
            spread = r.get("spread_home_points", "")
            total = r.get("total_over_points", "")

            rows.append({
                "date": game_date,
                "home_team": home,
                "away_team": away,
                "moneyline_home": ml_h,
                "moneyline_away": ml_a,
                "spread_home": spread,
                "total": total,
                "book": "betmgm",
                "source": "mgm_kaggle",
            })

    print(f"[MGM] Extracted {len(rows)} games ({min_date} to {max_date})")
    return rows


# ─── Source 2: SportsBettingReview scraper ───────────────────────────────────

SBR_BASE = "https://www.sportsbookreview.com/betting-odds/nba-basketball/money-line/"
PREFERRED_BOOKS = ["betmgm", "fanduel", "draftkings", "caesars", "bet365", "fanatics"]


def sbr_fetch_date(game_date, retries=3):
    """
    Fetch NBA moneylines from SBR for a given date (YYYY-MM-DD).
    Returns list of dicts with date, home_team, away_team, moneyline_home, moneyline_away.
    """
    url = f"{SBR_BASE}?date={game_date}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = (attempt + 1) * 10
                print(f"  [SBR] Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [SBR] HTTP {e.code} for {game_date}: {e}")
                return []
        except Exception as e:
            print(f"  [SBR] Error for {game_date} (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                return []

    # Extract embedded NEXT_DATA JSON
    matches = re.findall(r'id="__NEXT_DATA__"[^>]*>([^<]+)<', content)
    if not matches:
        # Sometimes whitespace varies
        matches = re.findall(r'__NEXT_DATA__[^>]*>([^<]+)<', content)
    if not matches:
        print(f"  [SBR] No NEXT_DATA found for {game_date}")
        return []

    try:
        d = json.loads(matches[0])
    except json.JSONDecodeError as e:
        print(f"  [SBR] JSON parse error for {game_date}: {e}")
        return []

    tables = d.get("props", {}).get("pageProps", {}).get("oddsTables", [])
    if not tables:
        return []

    rows = tables[0].get("oddsTableModel", {}).get("gameRows", [])
    results = []

    for row in rows:
        gv = row.get("gameView", {})
        away_full = gv.get("awayTeam", {}).get("fullName", "")
        home_full = gv.get("homeTeam", {}).get("fullName", "")
        start = gv.get("startDate", "")

        # Use the game_date passed in (SBR dates can be off by +1 timezone)
        game_dt = game_date

        # Find best closing moneyline across preferred books
        ml_home = ml_away = None
        book_used = None

        odds_views = row.get("oddsViews", [])

        # Try preferred books first
        for pbook in PREFERRED_BOOKS:
            for ov in odds_views:
                if not isinstance(ov, dict):
                    continue
                if ov.get("sportsbook", "").lower() != pbook:
                    continue
                cl = ov.get("currentLine", {})
                if not cl:
                    continue
                mh = cl.get("homeOdds")
                ma = cl.get("awayOdds")
                if mh is not None and ma is not None and mh != 0 and ma != 0:
                    ml_home = mh
                    ml_away = ma
                    book_used = pbook
                    break
            if ml_home is not None:
                break

        # Fallback: first book with valid data
        if ml_home is None:
            for ov in odds_views:
                if not isinstance(ov, dict):
                    continue
                cl = ov.get("currentLine", {})
                if not cl:
                    continue
                mh = cl.get("homeOdds")
                ma = cl.get("awayOdds")
                if mh is not None and ma is not None and mh != 0 and ma != 0:
                    ml_home = mh
                    ml_away = ma
                    book_used = ov.get("sportsbook", "unknown")
                    break

        if ml_home is None:
            continue

        results.append({
            "date": game_dt,
            "home_team": normalize_team(home_full),
            "away_team": normalize_team(away_full),
            "moneyline_home": ml_home,
            "moneyline_away": ml_away,
            "spread_home": "",
            "total": "",
            "book": book_used or "sbr_consensus",
            "source": "sbr_scrape",
        })

    return results


def sbr_scrape_date_range(start_date, end_date, delay=2.5):
    """
    Scrape SBR for all dates between start_date and end_date (inclusive).
    delay: seconds between requests to avoid rate limiting.
    """
    all_rows = []
    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    total_days = (end - current).days + 1
    processed = 0

    while current <= end:
        ds = current.strftime("%Y-%m-%d")
        print(f"  [SBR] Fetching {ds} ({processed+1}/{total_days})...", end=" ")
        rows = sbr_fetch_date(ds)
        if rows:
            all_rows.extend(rows)
            print(f"{len(rows)} games")
        else:
            print("0 games (no NBA games or error)")
        processed += 1
        current += timedelta(days=1)
        if current <= end:
            time.sleep(delay)

    return all_rows


# ─── Source 3: Local odds snapshots ──────────────────────────────────────────

def extract_local_snapshots():
    """
    Extract moneylines from locally stored The Odds API snapshots.
    These cover Mar 15-17, 2026 (API quota exhausted after that).
    """
    # Collect snapshot files
    snapshot_files = []
    base = os.path.join(BASE_DIR, "data")
    snapshot_files += glob.glob(os.path.join(base, "odds-*.json"))

    if not snapshot_files:
        print("[Snapshots] No snapshot files found.")
        return []

    snapshot_files = sorted(snapshot_files)

    def parse_snap_dt(fpath):
        fname = os.path.basename(fpath)
        m = re.search(r"(\d{8})-(\d{4})", fname)
        if m:
            try:
                return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
            except ValueError:
                pass
        return None

    games = {}  # key -> latest entry before game time

    for fpath in snapshot_files:
        snap_dt = parse_snap_dt(fpath)
        if snap_dt is None:
            continue

        try:
            with open(fpath) as f:
                raw = json.load(f)
        except Exception:
            continue

        data = raw.get("data", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])

        for g in data:
            if not isinstance(g, dict):
                continue
            home = g.get("home_team", "")
            away = g.get("away_team", "")
            ctime = g.get("commence_time", "")
            if not home or not away or not ctime:
                continue
            # commence_time can be int (epoch ms) in some formats
            if isinstance(ctime, int):
                from datetime import timezone
                ctime = datetime.fromtimestamp(ctime / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            ctime = str(ctime)

            game_key = (home, away, ctime[:10])
            preferred = ["pinnacle", "fanduel", "draftkings", "betmgm", "bovada"]

            best_ml = None
            for pbook in preferred:
                for book in g.get("bookmakers", []):
                    if book.get("key") != pbook:
                        continue
                    for mkt in book.get("markets", []):
                        if mkt["key"] != "h2h":
                            continue
                        prices = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                        mh = prices.get(home)
                        ma = prices.get(away)
                        if mh and ma and 1.005 < float(mh) < 25 and 1.005 < float(ma) < 25:
                            best_ml = {"home": float(mh), "away": float(ma), "book": pbook}
                            break
                    if best_ml:
                        break
                if best_ml:
                    break

            if not best_ml:
                for book in g.get("bookmakers", []):
                    for mkt in book.get("markets", []):
                        if mkt["key"] != "h2h":
                            continue
                        prices = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                        mh = prices.get(home)
                        ma = prices.get(away)
                        if mh and ma and 1.005 < float(mh) < 25 and 1.005 < float(ma) < 25:
                            best_ml = {"home": float(mh), "away": float(ma), "book": book["key"]}
                            break
                    if best_ml:
                        break

            if best_ml:
                if game_key not in games or games[game_key]["snap_dt"] < snap_dt:
                    games[game_key] = {
                        "snap_dt": snap_dt,
                        "date": ctime[:10],
                        "home_team": home,
                        "away_team": away,
                        "moneyline_home": best_ml["home"],
                        "moneyline_away": best_ml["away"],
                        "spread_home": "",
                        "total": "",
                        "book": best_ml["book"],
                        "source": "local_snapshot_decimal",
                    }

    rows = []
    for entry in games.values():
        rows.append({
            "date": entry["date"],
            "home_team": entry["home_team"],
            "away_team": entry["away_team"],
            "moneyline_home": entry["moneyline_home"],
            "moneyline_away": entry["moneyline_away"],
            "spread_home": entry["spread_home"],
            "total": entry["total"],
            "book": entry["book"],
            "source": entry["source"],
        })

    print(f"[Snapshots] Extracted {len(rows)} games from local snapshots")
    return rows


# ─── Dedup and merge ──────────────────────────────────────────────────────────

def deduplicate(rows):
    """
    Deduplicate rows by (date, home_team, away_team).
    Priority: mgm_kaggle > sbr_scrape > local_snapshot_decimal
    """
    SOURCE_PRIORITY = {
        "mgm_kaggle": 1,
        "sbr_scrape": 2,
        "local_snapshot_decimal": 3,
    }
    seen = {}
    for row in rows:
        key = (row["date"], row["home_team"], row["away_team"])
        if key not in seen:
            seen[key] = row
        else:
            current_priority = SOURCE_PRIORITY.get(seen[key]["source"], 99)
            new_priority = SOURCE_PRIORITY.get(row["source"], 99)
            if new_priority < current_priority:
                seen[key] = row
    return sorted(seen.values(), key=lambda r: (r["date"], r["home_team"]))


# ─── Write output ─────────────────────────────────────────────────────────────

def write_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape 2025-26 NBA season moneylines.")
    parser.add_argument("--from-date", default="2025-10-21", help="Start date YYYY-MM-DD")
    parser.add_argument("--to-date", default=date.today().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    parser.add_argument("--source", choices=["all", "mgm", "sbr", "snapshots"], default="all",
                        help="Which source to use")
    parser.add_argument("--sbr-start", default=None,
                        help="Override SBR start date (defaults to MGM end date + 1)")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV path")
    parser.add_argument("--delay", type=float, default=2.5,
                        help="Seconds between SBR requests (default 2.5)")
    args = parser.parse_args()

    all_rows = []

    # ── Source 1: MGM Kaggle (Oct 2025 - Feb 12, 2026) ──
    if args.source in ("all", "mgm"):
        print("\n=== Source 1: MGM Kaggle Dataset ===")
        mgm_end = "2026-02-12"
        mgm_rows = extract_mgm_season(
            min_date=args.from_date,
            max_date=min(mgm_end, args.to_date)
        )
        all_rows.extend(mgm_rows)
        print(f"  -> {len(mgm_rows)} rows from MGM")

    # ── Source 3: Local snapshots (Mar 15-17, 2026) ──
    if args.source in ("all", "snapshots"):
        print("\n=== Source 3: Local Odds Snapshots ===")
        snap_rows = extract_local_snapshots()
        all_rows.extend(snap_rows)
        print(f"  -> {len(snap_rows)} rows from local snapshots")

    # ── Source 2: SBR scraper (fills all gaps) ──
    if args.source in ("all", "sbr"):
        print("\n=== Source 2: SportsBettingReview Scraper ===")

        # Determine SBR date range
        if args.sbr_start:
            sbr_start = args.sbr_start
        elif args.source == "sbr":
            sbr_start = args.from_date
        else:
            # Fill Feb 13 - today (SBR has better data than MGM for recent games)
            sbr_start = "2026-02-13"

        sbr_end = args.to_date
        print(f"  Scraping {sbr_start} to {sbr_end}...")
        sbr_rows = sbr_scrape_date_range(sbr_start, sbr_end, delay=args.delay)
        all_rows.extend(sbr_rows)
        print(f"  -> {len(sbr_rows)} rows from SBR")

    # ── Dedup and write ──
    print(f"\n=== Deduplicating {len(all_rows)} total rows ===")
    final_rows = deduplicate(all_rows)
    print(f"Final: {len(final_rows)} unique games")

    # Print date coverage summary
    if final_rows:
        from collections import Counter
        months = Counter(r["date"][:7] for r in final_rows)
        print("\nCoverage by month:")
        for ym in sorted(months):
            print(f"  {ym}: {months[ym]} games")

    write_csv(final_rows, args.output)
    return final_rows


if __name__ == "__main__":
    main()
