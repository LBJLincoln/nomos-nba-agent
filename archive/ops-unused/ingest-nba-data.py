#!/usr/bin/env python3
"""
NBA Historical + Live Data Ingestion
=====================================
Pulls structured game/player/team data via nba_api for quant prediction models.

Sources:
  - nba_api  : games, box scores, team stats, player stats, standings
  - The Odds API : live + historical odds
  - Local odds snapshots : data/odds-*.json

Usage:
  python3 ingest-nba-data.py                   # full historical pull (4 seasons)
  python3 ingest-nba-data.py --season 2024-25  # single season
  python3 ingest-nba-data.py --quick           # today only (schedule + injuries + odds)
  python3 ingest-nba-data.py --daemon          # run every 6 hours
"""

import os
import sys
import json
import time
import ssl
import urllib.request
import urllib.parse
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_DIR = ROOT / "data" / "historical"
ODDS_SNAPSHOTS_DIR = ROOT / "data"  # odds-YYYYMMDD-HHMM.json live here

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ingest-nba-data")

# ─── Env ──────────────────────────────────────────────────────────────────────

def load_env():
    """Load .env.local from mon-ipad or repo root."""
    candidates = [
        ROOT / ".env.local",
        ROOT.parent / "mon-ipad" / ".env.local",
        Path("/home/termius/mon-ipad/.env.local"),
    ]
    for env_file in candidates:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            log.info(f"Loaded env from {env_file}")
            return
    log.warning("No .env.local found — API keys may be missing")


load_env()

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

# ─── nba_api import guard ─────────────────────────────────────────────────────

try:
    from nba_api.stats.endpoints import (
        leaguegamefinder,
        boxscoretraditionalv2,
        leaguestandingsv3,
        leaguedashteamstats,
        leaguedashplayerstats,
        commonteamroster,
        playercareerstats,
        leaguegamelog,
        scoreboardv2,
    )
    from nba_api.stats.static import teams as nba_teams_static
    from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
    NBA_API_AVAILABLE = True
    log.info("nba_api loaded OK")
except ImportError:
    NBA_API_AVAILABLE = False
    log.error(
        "nba_api is NOT installed. Run:  pip install nba_api\n"
        "Then re-run this script."
    )

# ─── HTTP helpers ─────────────────────────────────────────────────────────────

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

NBA_API_RATE_LIMIT = 0.65  # seconds between nba_api calls (generous margin)
_last_nba_call: float = 0.0


def _nba_rate_limit():
    """Enforce minimum gap between nba_api requests."""
    global _last_nba_call
    gap = time.time() - _last_nba_call
    if gap < NBA_API_RATE_LIMIT:
        time.sleep(NBA_API_RATE_LIMIT - gap)
    _last_nba_call = time.time()


def http_get(url: str, headers: Optional[dict] = None, timeout: int = 30):
    hdrs = headers or {}
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status
    except Exception as e:
        return {"error": str(e)}, 0


# ─── Season helpers ───────────────────────────────────────────────────────────

def current_season() -> str:
    """Return NBA season string for today, e.g. '2025-26'."""
    now = datetime.now()
    # NBA season starts in October; if we're before October it's still last season
    year = now.year if now.month >= 10 else now.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def season_to_nba_format(season: str) -> str:
    """'2025-26' → '2025-26' (already correct for nba_api)."""
    return season


def seasons_to_pull(user_seasons: Optional[list[str]] = None) -> list[str]:
    """Return list of seasons to ingest (current + last 3 by default)."""
    if user_seasons:
        return user_seasons
    cur = current_season()
    year = int(cur.split("-")[0])
    seasons = []
    for i in range(4):  # current season + 3 prior
        y = year - i
        seasons.append(f"{y}-{str(y + 1)[-2:]}")
    return seasons


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HISTORICAL GAME DATA
# ═══════════════════════════════════════════════════════════════════════════════

def pull_games_for_season(season: str) -> list[dict]:
    """
    Pull all NBA games for a given season using LeagueGameFinder.
    Returns list of game dicts with home/away designation and basic stats.
    """
    if not NBA_API_AVAILABLE:
        return []

    log.info(f"[GAMES] Pulling games for season {season}…")
    _nba_rate_limit()

    try:
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_to_nba_format(season),
            league_id_nullable="00",  # NBA
        )
        df = finder.get_data_frames()[0]
    except Exception as e:
        log.error(f"[GAMES] LeagueGameFinder failed for {season}: {e}")
        return []

    if df.empty:
        log.warning(f"[GAMES] No games found for {season}")
        return []

    games_by_id: dict[str, dict] = {}

    for _, row in df.iterrows():
        gid = str(row.get("GAME_ID", ""))
        if not gid:
            continue

        if gid not in games_by_id:
            games_by_id[gid] = {
                "game_id": gid,
                "season": season,
                "game_date": str(row.get("GAME_DATE", "")),
                "matchup": str(row.get("MATCHUP", "")),
                "home_team": None,
                "away_team": None,
                "home": {},
                "away": {},
            }

        entry = games_by_id[gid]
        matchup = str(row.get("MATCHUP", ""))
        is_home = "vs." in matchup  # "BOS vs. LAL" = home

        team_stats = {
            "team_id": str(row.get("TEAM_ID", "")),
            "team_abbr": str(row.get("TEAM_ABBREVIATION", "")),
            "team_name": str(row.get("TEAM_NAME", "")),
            "wl": str(row.get("WL", "")),
            "pts": _safe_float(row.get("PTS")),
            "fg_pct": _safe_float(row.get("FG_PCT")),
            "fg3_pct": _safe_float(row.get("FG3_PCT")),
            "ft_pct": _safe_float(row.get("FT_PCT")),
            "reb": _safe_float(row.get("REB")),
            "ast": _safe_float(row.get("AST")),
            "tov": _safe_float(row.get("TOV")),
            "stl": _safe_float(row.get("STL")),
            "blk": _safe_float(row.get("BLK")),
            "plus_minus": _safe_float(row.get("PLUS_MINUS")),
        }

        if is_home:
            entry["home_team"] = team_stats["team_abbr"]
            entry["home"] = team_stats
        else:
            entry["away_team"] = team_stats["team_abbr"]
            entry["away"] = team_stats

    games = list(games_by_id.values())
    log.info(f"[GAMES] {len(games)} games collected for {season}")
    return games


def ingest_historical_games(seasons: list[str]) -> dict[str, list]:
    """Pull games for each season and save to data/historical/games-{season}.json."""
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for season in seasons:
        out_path = HISTORICAL_DIR / f"games-{season}.json"
        log.info(f"[GAMES] Season {season} → {out_path.name}")

        games = pull_games_for_season(season)
        results[season] = games

        payload = {
            "season": season,
            "pulled_at": _utcnow(),
            "game_count": len(games),
            "games": games,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        log.info(f"[GAMES] Saved {len(games)} games → {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TEAM STANDINGS & ADVANCED STATS
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_team_stats(season: Optional[str] = None) -> dict:
    """
    Pull current team standings + advanced stats (ORtg, DRtg, Pace, eFG%, etc.)
    Saves to data/historical/team-stats-current.json
    """
    if not NBA_API_AVAILABLE:
        return {}

    season = season or current_season()
    log.info(f"[TEAMS] Pulling team stats for {season}…")
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    # --- Standings ---
    standings_data = []
    _nba_rate_limit()
    try:
        standings = leaguestandingsv3.LeagueStandingsV3(
            season=season_to_nba_format(season),
            season_type="Regular Season",
            league_id="00",
        )
        df_st = standings.get_data_frames()[0]
        for _, row in df_st.iterrows():
            standings_data.append({
                "team_id": str(row.get("TeamID", "")),
                "team": str(row.get("TeamName", "")),
                "city": str(row.get("TeamCity", "")),
                "conference": str(row.get("Conference", "")),
                "division": str(row.get("Division", "")),
                "wins": int(row.get("WINS", 0) or 0),
                "losses": int(row.get("LOSSES", 0) or 0),
                "win_pct": _safe_float(row.get("WinPCT")),
                "conf_rank": int(row.get("PlayoffRank", 0) or 0),
                "home_record": str(row.get("HOME", "")),
                "road_record": str(row.get("ROAD", "")),
                "last_10": str(row.get("L10", "")),
                "streak": str(row.get("strCurrentStreak", "")),
                "clinch_indicator": str(row.get("clinchIndicator", "")),
            })
        log.info(f"[TEAMS] {len(standings_data)} teams in standings")
    except Exception as e:
        log.error(f"[TEAMS] Standings failed: {e}")

    # --- Advanced stats (ORtg, DRtg, Pace, eFG%, TOV%, ORB%, FTr) ---
    advanced_data = []
    _nba_rate_limit()
    try:
        adv = leaguedashteamstats.LeagueDashTeamStats(
            season=season_to_nba_format(season),
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        )
        df_adv = adv.get_data_frames()[0]
        for _, row in df_adv.iterrows():
            advanced_data.append({
                "team_id": str(row.get("TEAM_ID", "")),
                "team_name": str(row.get("TEAM_NAME", "")),
                "off_rating": _safe_float(row.get("OFF_RATING")),
                "def_rating": _safe_float(row.get("DEF_RATING")),
                "net_rating": _safe_float(row.get("NET_RATING")),
                "pace": _safe_float(row.get("PACE")),
                "efg_pct": _safe_float(row.get("EFG_PCT")),
                "tov_pct": _safe_float(row.get("TM_TOV_PCT")),
                "orb_pct": _safe_float(row.get("OREB_PCT")),
                "ftr": _safe_float(row.get("FTA_RATE")),
                "ts_pct": _safe_float(row.get("TS_PCT")),
                "ast_pct": _safe_float(row.get("AST_PCT")),
                "pie": _safe_float(row.get("PIE")),
                "games_played": int(row.get("GP", 0) or 0),
                "wins": int(row.get("W", 0) or 0),
                "losses": int(row.get("L", 0) or 0),
            })
        log.info(f"[TEAMS] {len(advanced_data)} teams advanced stats")
    except Exception as e:
        log.error(f"[TEAMS] Advanced stats failed: {e}")

    # --- Base stats per game ---
    base_data = []
    _nba_rate_limit()
    try:
        base = leaguedashteamstats.LeagueDashTeamStats(
            season=season_to_nba_format(season),
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
        )
        df_base = base.get_data_frames()[0]
        for _, row in df_base.iterrows():
            base_data.append({
                "team_id": str(row.get("TEAM_ID", "")),
                "team_name": str(row.get("TEAM_NAME", "")),
                "pts": _safe_float(row.get("PTS")),
                "reb": _safe_float(row.get("REB")),
                "ast": _safe_float(row.get("AST")),
                "tov": _safe_float(row.get("TOV")),
                "stl": _safe_float(row.get("STL")),
                "blk": _safe_float(row.get("BLK")),
                "fg_pct": _safe_float(row.get("FG_PCT")),
                "fg3_pct": _safe_float(row.get("FG3_PCT")),
                "ft_pct": _safe_float(row.get("FT_PCT")),
                "plus_minus": _safe_float(row.get("PLUS_MINUS")),
            })
        log.info(f"[TEAMS] {len(base_data)} teams base stats")
    except Exception as e:
        log.error(f"[TEAMS] Base stats failed: {e}")

    # Merge by team_id
    adv_map = {r["team_id"]: r for r in advanced_data}
    base_map = {r["team_id"]: r for r in base_data}
    standings_map = {r["team_id"]: r for r in standings_data}

    all_team_ids = set(adv_map) | set(base_map) | set(standings_map)
    merged = []
    for tid in sorted(all_team_ids):
        entry = {"team_id": tid}
        entry.update(standings_map.get(tid, {}))
        entry.update(base_map.get(tid, {}))
        entry.update(adv_map.get(tid, {}))
        merged.append(entry)

    payload = {
        "season": season,
        "pulled_at": _utcnow(),
        "team_count": len(merged),
        "teams": merged,
    }

    out_path = HISTORICAL_DIR / "team-stats-current.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info(f"[TEAMS] Saved {len(merged)} teams → {out_path}")
    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PLAYER STATS
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_player_stats(season: Optional[str] = None, min_minutes: float = 12.0) -> dict:
    """
    Pull per-game player stats for the season.
    Filters to players averaging >= min_minutes to keep the dataset manageable.
    Saves to data/historical/player-stats-{season}.json
    """
    if not NBA_API_AVAILABLE:
        return {}

    season = season or current_season()
    log.info(f"[PLAYERS] Pulling player stats for {season} (min {min_minutes} min/g)…")
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    # Base per-game stats
    base_players = []
    _nba_rate_limit()
    try:
        base = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_to_nba_format(season),
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
        )
        df = base.get_data_frames()[0]
        for _, row in df.iterrows():
            min_pg = _safe_float(row.get("MIN")) or 0.0
            if min_pg < min_minutes:
                continue
            base_players.append({
                "player_id": str(row.get("PLAYER_ID", "")),
                "player_name": str(row.get("PLAYER_NAME", "")),
                "team_id": str(row.get("TEAM_ID", "")),
                "team_abbr": str(row.get("TEAM_ABBREVIATION", "")),
                "age": _safe_float(row.get("AGE")),
                "games_played": int(row.get("GP", 0) or 0),
                "games_started": int(row.get("GS", 0) or 0),
                "min_pg": min_pg,
                "pts": _safe_float(row.get("PTS")),
                "reb": _safe_float(row.get("REB")),
                "ast": _safe_float(row.get("AST")),
                "tov": _safe_float(row.get("TOV")),
                "stl": _safe_float(row.get("STL")),
                "blk": _safe_float(row.get("BLK")),
                "fg_pct": _safe_float(row.get("FG_PCT")),
                "fg3_pct": _safe_float(row.get("FG3_PCT")),
                "ft_pct": _safe_float(row.get("FT_PCT")),
                "fga_pg": _safe_float(row.get("FGA")),
                "fg3a_pg": _safe_float(row.get("FG3A")),
                "fta_pg": _safe_float(row.get("FTA")),
                "plus_minus": _safe_float(row.get("PLUS_MINUS")),
            })
        log.info(f"[PLAYERS] {len(base_players)} qualifying players (base stats)")
    except Exception as e:
        log.error(f"[PLAYERS] Base stats failed: {e}")

    # Advanced per-game stats (PER, TS%, USG%, BPM etc.)
    adv_players = []
    _nba_rate_limit()
    try:
        adv = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_to_nba_format(season),
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        )
        df_adv = adv.get_data_frames()[0]
        for _, row in df_adv.iterrows():
            adv_players.append({
                "player_id": str(row.get("PLAYER_ID", "")),
                "per": _safe_float(row.get("PIE")),       # NBA.com uses PIE; PER from bbref
                "ts_pct": _safe_float(row.get("TS_PCT")),
                "usg_pct": _safe_float(row.get("USG_PCT")),
                "ast_pct": _safe_float(row.get("AST_PCT")),
                "reb_pct": _safe_float(row.get("REB_PCT")),
                "to_pct": _safe_float(row.get("TM_TOV_PCT")),
                "off_rating": _safe_float(row.get("OFF_RATING")),
                "def_rating": _safe_float(row.get("DEF_RATING")),
                "net_rating": _safe_float(row.get("NET_RATING")),
                "pace": _safe_float(row.get("PACE")),
            })
        log.info(f"[PLAYERS] {len(adv_players)} players advanced stats")
    except Exception as e:
        log.error(f"[PLAYERS] Advanced stats failed: {e}")

    # Merge advanced into base by player_id
    adv_map = {p["player_id"]: p for p in adv_players}
    for player in base_players:
        adv = adv_map.get(player["player_id"], {})
        for k, v in adv.items():
            if k != "player_id" and k not in player:
                player[k] = v

    payload = {
        "season": season,
        "pulled_at": _utcnow(),
        "player_count": len(base_players),
        "min_minutes_filter": min_minutes,
        "players": base_players,
    }

    out_path = HISTORICAL_DIR / f"player-stats-{season}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info(f"[PLAYERS] Saved {len(base_players)} players → {out_path}")
    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# 4. INJURY REPORTS
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_injuries() -> dict:
    """
    Pull current NBA injury report from nba_api (CommonPlayerInfo / daily roster)
    and supplement with The Odds API player props injury signals if available.

    nba_api does not have a direct injury endpoint; we use the live Scoreboard
    to find today's games, then note any players listed as inactive.

    Saves to data/historical/injuries-current.json
    """
    if not NBA_API_AVAILABLE:
        return {}

    log.info("[INJURIES] Pulling current injury report…")
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    injuries = []

    # Strategy: use nba_api live scoreboard to get today's games and inactives
    _nba_rate_limit()
    try:
        board = live_scoreboard.ScoreBoard()
        board_data = board.get_dict()
        games = board_data.get("scoreboard", {}).get("games", [])

        log.info(f"[INJURIES] Found {len(games)} games on today's scoreboard")

        for game in games:
            game_id = game.get("gameId", "")
            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})
            game_time = game.get("gameTimeUTC", "")

            for side, team_info in [("home", home), ("away", away)]:
                team_abbr = team_info.get("teamTricode", "")
                for inactive in team_info.get("teamSlate", {}).get("injuries", []):
                    injuries.append({
                        "player_id": str(inactive.get("personId", "")),
                        "player_name": inactive.get("firstName", "") + " " + inactive.get("familyName", ""),
                        "team_abbr": team_abbr,
                        "game_id": game_id,
                        "game_time_utc": game_time,
                        "side": side,
                        "status": inactive.get("status", ""),
                        "description": inactive.get("description", ""),
                    })

    except Exception as e:
        log.warning(f"[INJURIES] Live scoreboard injuries failed: {e} — trying game log inactives")

    # Fallback: pull today's game log and check status flags
    if not injuries:
        _nba_rate_limit()
        try:
            today_str = datetime.now().strftime("%m/%d/%Y")
            gamelog = leaguegamelog.LeagueGameLog(
                season=current_season(),
                season_type_all_star="Regular Season",
                date_from_nullable=today_str,
                date_to_nullable=today_str,
            )
            df = gamelog.get_data_frames()[0]
            log.info(f"[INJURIES] Today's game log: {len(df)} rows (no direct injury data in this endpoint)")
        except Exception as e:
            log.warning(f"[INJURIES] Game log fallback failed: {e}")

    payload = {
        "pulled_at": _utcnow(),
        "injury_count": len(injuries),
        "note": (
            "nba_api does not expose a dedicated injury report endpoint. "
            "Data sourced from live scoreboard inactives. "
            "For richer injury data consider: "
            "Rotowire NBA injuries RSS, ESPN API, or The Odds API player props signals."
        ),
        "injuries": injuries,
    }

    out_path = HISTORICAL_DIR / "injuries-current.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info(f"[INJURIES] Saved {len(injuries)} injury entries → {out_path}")
    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ODDS — LIVE + HISTORICAL CONSOLIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_live_odds_structured() -> list[dict]:
    """
    Fetch current NBA odds from The Odds API (h2h, spreads, totals).
    Returns a flat list of game-level odds objects.
    """
    if not ODDS_API_KEY:
        log.warning("[ODDS] No ODDS_API_KEY in env — skipping live odds")
        return []

    url = (
        "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
        f"?apiKey={ODDS_API_KEY}"
        "&regions=us,eu"
        "&markets=h2h,spreads,totals"
        "&oddsFormat=decimal"
        "&dateFormat=iso"
    )
    data, status = http_get(url, timeout=30)
    if "error" in data or not isinstance(data, list):
        log.error(f"[ODDS] Live fetch failed (status {status}): {data.get('error','')}")
        return []

    log.info(f"[ODDS] Live: {len(data)} games from The Odds API")
    return data


def consolidate_odds_snapshots(max_files: int = 200) -> list[dict]:
    """
    Scan all odds-YYYYMMDD-HHMM.json snapshot files in data/ and consolidate
    into a deduplicated chronological list of game-odds records.
    """
    snapshots = sorted(ODDS_SNAPSHOTS_DIR.glob("odds-*.json"))[-max_files:]
    log.info(f"[ODDS-HIST] Found {len(snapshots)} snapshot files to consolidate")

    seen_keys: set[str] = set()
    records = []

    for snap_path in snapshots:
        # Parse timestamp from filename: odds-20260315-1314.json
        ts_str = snap_path.stem.replace("odds-", "")  # "20260315-1314"
        try:
            snap_dt = datetime.strptime(ts_str, "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)
            snap_iso = snap_dt.isoformat()
        except ValueError:
            snap_iso = ts_str

        try:
            raw = json.loads(snap_path.read_text())
        except Exception as e:
            log.warning(f"[ODDS-HIST] Could not parse {snap_path.name}: {e}")
            continue

        games = raw if isinstance(raw, list) else raw.get("games", [])

        for game in games:
            game_id = game.get("id", "")
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            commence = game.get("commence_time", "")

            for bk in game.get("bookmakers", []):
                bk_key = bk.get("key", "")
                bk_update = bk.get("last_update", snap_iso)

                for market in bk.get("markets", []):
                    market_key = market.get("key", "")
                    dedup_key = f"{game_id}:{bk_key}:{market_key}:{bk_update}"

                    if dedup_key in seen_keys:
                        continue
                    seen_keys.add(dedup_key)

                    outcomes = {}
                    for o in market.get("outcomes", []):
                        name = o.get("name", "")
                        outcomes[name] = {
                            "price": o.get("price"),
                            "point": o.get("point"),
                        }

                    records.append({
                        "snapshot_file": snap_path.name,
                        "snapshot_ts": snap_iso,
                        "game_id": game_id,
                        "home_team": home,
                        "away_team": away,
                        "commence_time": commence,
                        "bookmaker": bk_key,
                        "market": market_key,
                        "last_update": bk_update,
                        "outcomes": outcomes,
                    })

    # Sort chronologically by snapshot_ts
    records.sort(key=lambda r: r["snapshot_ts"])
    log.info(f"[ODDS-HIST] Consolidated {len(records)} unique odds records from snapshots")
    return records


def ingest_odds(quick: bool = False) -> dict:
    """
    Pull live odds + consolidate all local snapshots.
    Saves to data/historical/odds-history.jsonl
    """
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    # Live odds
    live = fetch_live_odds_structured()

    # Save live odds to a snapshot too
    if live:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        snap_path = ODDS_SNAPSHOTS_DIR / f"odds-{ts}.json"
        snap_path.write_text(json.dumps(live, indent=2))
        log.info(f"[ODDS] Live snapshot saved → {snap_path.name}")

    if quick:
        # In quick mode, just save live odds as-is without full consolidation
        out_path = HISTORICAL_DIR / "odds-live.json"
        out_path.write_text(json.dumps({
            "pulled_at": _utcnow(),
            "game_count": len(live),
            "games": live,
        }, indent=2))
        log.info(f"[ODDS] Quick mode: saved {len(live)} live games → {out_path.name}")
        return {"live_games": len(live)}

    # Full consolidation of all historical snapshots
    history = consolidate_odds_snapshots()

    out_path = HISTORICAL_DIR / "odds-history.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for record in history:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(f"[ODDS] Saved {len(history)} odds records → {out_path}")

    return {
        "live_games": len(live),
        "historical_records": len(history),
        "output": str(out_path),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TODAY'S SCHEDULE (for --quick mode)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_todays_schedule() -> dict:
    """
    Fetch today's NBA schedule using live ScoreBoard.
    Saves to data/historical/schedule-today.json
    """
    if not NBA_API_AVAILABLE:
        return {}

    log.info("[SCHEDULE] Fetching today's schedule…")
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    _nba_rate_limit()
    try:
        board = live_scoreboard.ScoreBoard()
        data = board.get_dict()
        scoreboard = data.get("scoreboard", {})
        games = scoreboard.get("games", [])

        schedule = []
        for game in games:
            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})
            schedule.append({
                "game_id": game.get("gameId", ""),
                "game_time_utc": game.get("gameTimeUTC", ""),
                "game_status": game.get("gameStatusText", ""),
                "home_team": home.get("teamTricode", ""),
                "home_team_id": str(home.get("teamId", "")),
                "home_score": home.get("score", 0),
                "away_team": away.get("teamTricode", ""),
                "away_team_id": str(away.get("teamId", "")),
                "away_score": away.get("score", 0),
                "arena": game.get("arenaName", ""),
                "city": game.get("arenaCity", ""),
                "period": game.get("period", 0),
                "game_clock": game.get("gameClock", ""),
            })

        payload = {
            "date": scoreboard.get("gameDate", datetime.now().strftime("%Y-%m-%d")),
            "pulled_at": _utcnow(),
            "game_count": len(schedule),
            "games": schedule,
        }

        out_path = HISTORICAL_DIR / "schedule-today.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        log.info(f"[SCHEDULE] {len(schedule)} games today → {out_path.name}")
        return payload

    except Exception as e:
        log.error(f"[SCHEDULE] Failed: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(val) -> Optional[float]:
    try:
        if val is None or val == "":
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _print_summary(label: str, result):
    if isinstance(result, dict):
        count = result.get("game_count") or result.get("team_count") or result.get("player_count") or result.get("injury_count") or 0
        log.info(f"  {label}: done ({count} records)")
    elif isinstance(result, list):
        log.info(f"  {label}: done ({len(result)} records)")
    else:
        log.info(f"  {label}: done")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MODES
# ═══════════════════════════════════════════════════════════════════════════════

def run_full(seasons: list[str]):
    """Full historical ingest: games + players + teams + injuries + odds."""
    log.info("=" * 60)
    log.info(f"NBA FULL INGEST — seasons: {', '.join(seasons)}")
    log.info("=" * 60)

    t0 = time.time()

    # 1. Historical game data (all seasons)
    log.info("\n[1/5] Historical game data…")
    games_result = ingest_historical_games(seasons)
    total_games = sum(len(v) for v in games_result.values())
    log.info(f"  Total: {total_games} games across {len(seasons)} seasons")

    # 2. Team standings + advanced stats (current season)
    log.info("\n[2/5] Team standings + advanced stats…")
    team_result = ingest_team_stats(seasons[0])
    _print_summary("Teams", team_result)

    # 3. Player stats (current + last season)
    log.info("\n[3/5] Player stats…")
    for season in seasons[:2]:
        player_result = ingest_player_stats(season)
        _print_summary(f"Players {season}", player_result)

    # 4. Injuries
    log.info("\n[4/5] Injury report…")
    injury_result = ingest_injuries()
    _print_summary("Injuries", injury_result)

    # 5. Odds
    log.info("\n[5/5] Odds (live + historical consolidation)…")
    odds_result = ingest_odds(quick=False)
    log.info(f"  Live games: {odds_result.get('live_games', 0)}, "
             f"Historical records: {odds_result.get('historical_records', 0)}")

    elapsed = time.time() - t0
    log.info("\n" + "=" * 60)
    log.info(f"INGEST COMPLETE — {elapsed:.1f}s")
    log.info(f"Output: {HISTORICAL_DIR}")
    log.info("=" * 60)


def run_quick():
    """Quick mode: today's schedule + injuries + live odds only."""
    log.info("=" * 60)
    log.info("NBA QUICK INGEST — today's data only")
    log.info("=" * 60)

    t0 = time.time()

    log.info("\n[1/3] Today's schedule…")
    fetch_todays_schedule()

    log.info("\n[2/3] Injury report…")
    ingest_injuries()

    log.info("\n[3/3] Live odds…")
    ingest_odds(quick=True)

    elapsed = time.time() - t0
    log.info(f"\nQUICK INGEST COMPLETE — {elapsed:.1f}s")


def run_daemon(seasons: list[str], interval_hours: int = 6):
    """
    Daemon mode: run full ingest every `interval_hours` hours.
    Runs --quick in between full cycles to keep live data fresh.
    """
    log.info(f"[DAEMON] Starting NBA data daemon (full every {interval_hours}h, quick every 30min)")
    cycle = 0

    while True:
        cycle += 1
        log.info(f"\n[DAEMON] Cycle {cycle} — {_utcnow()}")

        try:
            if cycle == 1 or cycle % (interval_hours * 2) == 0:
                # Every `interval_hours` hours: full ingest
                run_full(seasons)
            else:
                # Otherwise: quick update (schedule + injuries + odds)
                run_quick()
        except KeyboardInterrupt:
            log.info("[DAEMON] Interrupted by user")
            sys.exit(0)
        except Exception as e:
            log.error(f"[DAEMON] Cycle {cycle} failed: {e}")

        sleep_mins = 30
        log.info(f"[DAEMON] Sleeping {sleep_mins} minutes…")
        try:
            time.sleep(sleep_mins * 60)
        except KeyboardInterrupt:
            log.info("[DAEMON] Interrupted during sleep")
            sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NBA Historical + Live Data Ingestion for quant prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full ingest — 4 seasons of game/player/team data + odds consolidation
  python3 ingest-nba-data.py

  # Single season
  python3 ingest-nba-data.py --season 2024-25

  # Multiple seasons
  python3 ingest-nba-data.py --season 2024-25 2023-24

  # Quick mode — today's schedule + injuries + live odds only
  python3 ingest-nba-data.py --quick

  # Daemon mode — full every 6h, quick every 30min
  python3 ingest-nba-data.py --daemon

  # Daemon with custom interval
  python3 ingest-nba-data.py --daemon --interval 12
""",
    )
    parser.add_argument(
        "--season",
        nargs="+",
        metavar="SEASON",
        help="Season(s) to pull, e.g. 2025-26 2024-25 (default: current + last 3)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: today's schedule + injuries + live odds only",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Daemon mode: run every --interval hours",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=6,
        metavar="HOURS",
        help="Daemon interval in hours (default: 6)",
    )
    parser.add_argument(
        "--min-minutes",
        type=float,
        default=12.0,
        metavar="MIN",
        help="Minimum minutes/game filter for player stats (default: 12.0)",
    )

    args = parser.parse_args()

    # Validate nba_api for non-quick operations
    if not args.quick and not NBA_API_AVAILABLE:
        log.error(
            "nba_api is required for this mode. Install it:\n"
            "  pip install nba_api\n"
            "For quick odds-only mode without nba_api, use:  --quick"
        )
        sys.exit(1)

    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        run_quick()
    elif args.daemon:
        target_seasons = seasons_to_pull(args.season)
        run_daemon(target_seasons, interval_hours=args.interval)
    else:
        target_seasons = seasons_to_pull(args.season)
        run_full(target_seasons)


if __name__ == "__main__":
    main()
