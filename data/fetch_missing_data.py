#!/usr/bin/env python3
"""
NBA Quant — Missing Data Fetcher
=================================
Fetches 3 critical missing data sources from nba_api:
  1. Quarter-by-quarter scores (Q1-Q4 for home/away per game)
  2. Referee assignments + computed referee profiles
  3. Injury/inactive reports per game with impact estimates

Designed to run on HF Space (16GB RAM). NOT for VM.

Usage:
  python fetch_missing_data.py                           # All 8 seasons
  python fetch_missing_data.py --seasons 2023-24 2024-25 # Specific seasons
  python fetch_missing_data.py --only quarters           # Only quarter scores
  python fetch_missing_data.py --only referees           # Only referee data
  python fetch_missing_data.py --only injuries           # Only injury data
  python fetch_missing_data.py --resume                  # Resume interrupted run

Rate limits: 0.6s between nba_api calls to avoid IP bans.
Saves progress incrementally per game — safe to interrupt and resume.
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# nba_api imports — lazy so helper functions work without the library
# ---------------------------------------------------------------------------
_nba_api_available = False


def _ensure_nba_api():
    """Import nba_api or raise a clear error."""
    global _nba_api_available
    if _nba_api_available:
        return
    try:
        import nba_api  # noqa: F401
        _nba_api_available = True
    except ImportError:
        print("ERROR: nba_api is not installed. Run: pip install nba_api")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_SEASONS = [
    "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
]

DATA_DIR = Path(__file__).resolve().parent
HISTORICAL_DIR = DATA_DIR / "historical"

# Mutable config so rate limit can be adjusted at runtime without `global`
_CONFIG = {
    "rate_limit": 0.6,  # Minimum delay between nba_api calls (seconds)
}

# Average minutes per game for impact estimation
LEAGUE_AVG_MPG = 24.0  # Rough league avg minutes for rotation players


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _rate_limit():
    """Sleep to respect nba_api rate limits."""
    time.sleep(_CONFIG["rate_limit"])


def _safe_float(val, default=0.0):
    """Convert a value to float safely."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _progress_bar(current: int, total: int, prefix: str = "", width: int = 40):
    """Print a simple progress bar to stdout."""
    if total <= 0:
        return
    pct = current / total
    filled = int(width * pct)
    bar = "=" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({pct:.0%})")
    sys.stdout.flush()
    if current >= total:
        print()


def _load_json(path: Path) -> Optional[dict]:
    """Load a JSON file if it exists."""
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_json(path: Path, data: dict):
    """Save data to JSON with atomic write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(path)


def _load_game_ids_for_season(season: str) -> List[dict]:
    """
    Load game IDs and metadata from existing historical games files.
    Returns list of dicts with game_id, home_team, away_team, game_date.
    """
    games_file = HISTORICAL_DIR / f"games-{season}.json"
    if not games_file.exists():
        print(f"  WARNING: {games_file} not found — will fetch game list from nba_api")
        return _fetch_game_ids_from_api(season)

    data = _load_json(games_file)
    if data is None:
        return _fetch_game_ids_from_api(season)

    games = data.get("games", [])
    if not games:
        return _fetch_game_ids_from_api(season)

    return [
        {
            "game_id": g["game_id"],
            "home_team": g.get("home_team", ""),
            "away_team": g.get("away_team", ""),
            "game_date": g.get("game_date", ""),
            "home_team_id": g.get("home", {}).get("team_id", ""),
            "away_team_id": g.get("away", {}).get("team_id", ""),
        }
        for g in games
        if g.get("game_id")
    ]


def _fetch_game_ids_from_api(season: str) -> List[dict]:
    """Fetch all game IDs for a season via nba_api LeagueGameFinder."""
    _ensure_nba_api()
    from nba_api.stats.endpoints import leaguegamefinder

    print(f"  Fetching game list for {season} from nba_api...")
    _rate_limit()

    try:
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable="Regular Season",
        )
        df = finder.get_data_frames()[0]
    except Exception as e:
        print(f"  ERROR fetching game list: {e}")
        return []

    # Also get playoffs
    _rate_limit()
    try:
        finder_po = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable="Playoffs",
        )
        df_po = finder_po.get_data_frames()[0]
        import pandas as pd
        df = pd.concat([df, df_po], ignore_index=True)
    except Exception:
        pass  # Playoffs may not exist for current season

    if df.empty:
        return []

    # Group by GAME_ID to get home/away
    games = {}
    for _, row in df.iterrows():
        gid = str(row.get("GAME_ID", ""))
        if not gid:
            continue
        matchup = str(row.get("MATCHUP", ""))
        team_abbr = str(row.get("TEAM_ABBREVIATION", ""))
        team_id = str(row.get("TEAM_ID", ""))
        game_date = str(row.get("GAME_DATE", ""))

        if gid not in games:
            games[gid] = {
                "game_id": gid,
                "game_date": game_date,
                "home_team": "",
                "away_team": "",
                "home_team_id": "",
                "away_team_id": "",
            }

        # "vs." means home, "@" means away
        if " vs. " in matchup:
            games[gid]["home_team"] = team_abbr
            games[gid]["home_team_id"] = team_id
        elif " @ " in matchup:
            games[gid]["away_team"] = team_abbr
            games[gid]["away_team_id"] = team_id

    return list(games.values())


# ---------------------------------------------------------------------------
# 1. QUARTER-BY-QUARTER SCORES
# ---------------------------------------------------------------------------

def fetch_quarter_scores(season: str, resume: bool = True) -> Dict[str, dict]:
    """
    Fetch quarter-by-quarter scores for all games in a season.

    Uses BoxScoreSummaryV2 → LineScore to get Q1-Q4 (and OT) points.

    Returns:
        Dict mapping game_id to quarter score breakdown.
    """
    _ensure_nba_api()
    from nba_api.stats.endpoints import boxscoresummaryv2

    output_path = HISTORICAL_DIR / f"quarter-scores-{season}.json"
    existing = {}
    if resume:
        existing = _load_json(output_path) or {}
        # Strip metadata keys
        existing = {k: v for k, v in existing.items() if not k.startswith("_")}

    games = _load_game_ids_for_season(season)
    if not games:
        print(f"  No games found for {season}")
        return existing

    total = len(games)
    fetched = 0
    errors = []

    print(f"\n--- Quarter Scores: {season} ({total} games, {len(existing)} cached) ---")

    for i, game in enumerate(games):
        gid = game["game_id"]

        if gid in existing:
            _progress_bar(i + 1, total, prefix=f"  Q-Scores {season}")
            continue

        _rate_limit()
        try:
            box = boxscoresummaryv2.BoxScoreSummaryV2(game_id=gid)
            line_score = box.line_score.get_dict()
            headers = line_score.get("headers", [])
            rows = line_score.get("data", [])

            if not rows or not headers:
                errors.append({"game_id": gid, "error": "empty line_score"})
                _progress_bar(i + 1, total, prefix=f"  Q-Scores {season}")
                continue

            # Build header index
            h_idx = {h: idx for idx, h in enumerate(headers)}

            record = {
                "game_date": game.get("game_date", ""),
                "home_team": game.get("home_team", ""),
                "away_team": game.get("away_team", ""),
            }

            for row in rows:
                team_id = str(row[h_idx.get("TEAM_ID", 0)])
                team_abbr = str(row[h_idx.get("TEAM_ABBREVIATION", 1)])

                # Determine if home or away
                is_home = (
                    team_id == str(game.get("home_team_id", ""))
                    or team_abbr == game.get("home_team", "")
                )
                prefix = "home" if is_home else "away"

                record[f"{prefix}_team_abbr"] = team_abbr
                record[f"{prefix}_q1"] = _safe_float(row[h_idx["PTS_QTR1"]] if "PTS_QTR1" in h_idx else None)
                record[f"{prefix}_q2"] = _safe_float(row[h_idx["PTS_QTR2"]] if "PTS_QTR2" in h_idx else None)
                record[f"{prefix}_q3"] = _safe_float(row[h_idx["PTS_QTR3"]] if "PTS_QTR3" in h_idx else None)
                record[f"{prefix}_q4"] = _safe_float(row[h_idx["PTS_QTR4"]] if "PTS_QTR4" in h_idx else None)

                # Overtime periods if present
                ot_total = 0.0
                for ot_n in range(1, 11):  # Up to 10 OT periods
                    ot_key = f"PTS_OT{ot_n}"
                    if ot_key in h_idx:
                        ot_pts = _safe_float(row[h_idx[ot_key]])
                        if ot_pts > 0:
                            record[f"{prefix}_ot{ot_n}"] = ot_pts
                            ot_total += ot_pts
                record[f"{prefix}_ot_total"] = ot_total
                record[f"{prefix}_pts"] = _safe_float(row[h_idx["PTS"]] if "PTS" in h_idx else None)

            # Compute derived features useful for the engine
            home_q1 = record.get("home_q1", 0)
            away_q1 = record.get("away_q1", 0)
            home_q3 = record.get("home_q3", 0)
            away_q3 = record.get("away_q3", 0)
            home_q4 = record.get("home_q4", 0)
            away_q4 = record.get("away_q4", 0)
            home_h1 = record.get("home_q1", 0) + record.get("home_q2", 0)
            away_h1 = record.get("away_q1", 0) + record.get("away_q2", 0)
            home_h2 = record.get("home_q3", 0) + record.get("home_q4", 0)
            away_h2 = record.get("away_q3", 0) + record.get("away_q4", 0)

            record["home_h1"] = home_h1
            record["home_h2"] = home_h2
            record["away_h1"] = away_h1
            record["away_h2"] = away_h2
            record["home_half_adjustment"] = home_h2 - home_h1
            record["away_half_adjustment"] = away_h2 - away_h1
            record["q1_margin"] = home_q1 - away_q1
            record["q3_margin"] = home_q3 - away_q3
            record["q4_margin"] = home_q4 - away_q4
            record["has_ot"] = record.get("home_ot_total", 0) > 0

            existing[gid] = record
            fetched += 1

            # Save progress every 50 games
            if fetched % 50 == 0:
                _save_with_meta(output_path, existing, season, errors)

        except Exception as e:
            errors.append({"game_id": gid, "error": str(e)})

        _progress_bar(i + 1, total, prefix=f"  Q-Scores {season}")

    # Final save
    _save_with_meta(output_path, existing, season, errors)
    print(f"  Fetched {fetched} new, {len(existing)} total, {len(errors)} errors")

    return existing


# ---------------------------------------------------------------------------
# 2. REFEREE ASSIGNMENTS + PROFILES
# ---------------------------------------------------------------------------

def fetch_referee_data(season: str, resume: bool = True) -> Dict[str, dict]:
    """
    Fetch referee assignments for all games in a season.

    Uses BoxScoreSummaryV2 → Officials to get referee names/IDs per game.
    Then computes referee profiles: avg fouls, home bias, pace impact, etc.

    Returns:
        Dict mapping game_id to referee data + computed profiles.
    """
    _ensure_nba_api()
    from nba_api.stats.endpoints import boxscoresummaryv2

    output_path = HISTORICAL_DIR / f"referee-data-{season}.json"
    existing = {}
    if resume:
        existing = _load_json(output_path) or {}
        existing = {k: v for k, v in existing.items() if not k.startswith("_")}

    games = _load_game_ids_for_season(season)
    if not games:
        print(f"  No games found for {season}")
        return existing

    # We also need team foul data from BoxScoreTraditionalV2 to compute profiles.
    # We'll collect raw ref assignments first, then compute profiles in a second pass.

    total = len(games)
    fetched = 0
    errors = []

    print(f"\n--- Referee Data: {season} ({total} games, {len(existing)} cached) ---")

    for i, game in enumerate(games):
        gid = game["game_id"]

        if gid in existing and "refs" in existing[gid]:
            _progress_bar(i + 1, total, prefix=f"  Refs {season}")
            continue

        _rate_limit()
        try:
            box = boxscoresummaryv2.BoxScoreSummaryV2(game_id=gid)

            # --- Officials ---
            officials = box.officials.get_dict()
            off_headers = officials.get("headers", [])
            off_data = officials.get("data", [])
            off_idx = {h: idx for idx, h in enumerate(off_headers)}

            refs = []
            for row in off_data:
                ref_info = {
                    "official_id": str(row[off_idx["OFFICIAL_ID"]]) if "OFFICIAL_ID" in off_idx else "",
                    "first_name": str(row[off_idx["FIRST_NAME"]]) if "FIRST_NAME" in off_idx else "",
                    "last_name": str(row[off_idx["LAST_NAME"]]) if "LAST_NAME" in off_idx else "",
                    "jersey_num": str(row[off_idx["JERSEY_NUM"]]) if "JERSEY_NUM" in off_idx else "",
                }
                ref_info["full_name"] = f"{ref_info['first_name']} {ref_info['last_name']}".strip()
                refs.append(ref_info)

            # --- Team fouls from the game summary / line score ---
            # Attempt to get foul data from other_stats or line_score
            home_fouls = 0.0
            away_fouls = 0.0
            home_pts = 0.0
            away_pts = 0.0
            home_pace_proxy = 0.0
            away_pace_proxy = 0.0

            try:
                other_stats = box.other_stats.get_dict()
                os_headers = other_stats.get("headers", [])
                os_data = other_stats.get("data", [])
                os_idx = {h: idx for idx, h in enumerate(os_headers)}

                for row in os_data:
                    team_id = str(row[os_idx.get("TEAM_ID", 0)])
                    team_abbr = str(row[os_idx.get("TEAM_ABBREVIATION", 1)]) if "TEAM_ABBREVIATION" in os_idx else ""
                    is_home = (
                        team_id == str(game.get("home_team_id", ""))
                        or team_abbr == game.get("home_team", "")
                    )
                    pf_val = _safe_float(row[os_idx["PTS_PAINT"]] if "PTS_PAINT" in os_idx else 0)
                    # other_stats has PTS_FB, PTS_PAINT, etc. but not PF directly
                    # We'll get fouls from a different source below
            except Exception:
                pass

            # Get fouls from line_score (PF column if available) or box score
            try:
                line_score = box.line_score.get_dict()
                ls_headers = line_score.get("headers", [])
                ls_data = line_score.get("data", [])
                ls_idx = {h: idx for idx, h in enumerate(ls_headers)}

                for row in ls_data:
                    team_id = str(row[ls_idx.get("TEAM_ID", 0)])
                    team_abbr = str(row[ls_idx.get("TEAM_ABBREVIATION", 1)]) if "TEAM_ABBREVIATION" in ls_idx else ""
                    is_home = (
                        team_id == str(game.get("home_team_id", ""))
                        or team_abbr == game.get("home_team", "")
                    )
                    pts = _safe_float(row[ls_idx["PTS"]] if "PTS" in ls_idx else 0)
                    # FGA as pace proxy
                    fga = _safe_float(row[ls_idx["FGA"]] if "FGA" in ls_idx else 0)

                    if is_home:
                        home_pts = pts
                        home_pace_proxy = fga
                    else:
                        away_pts = pts
                        away_pace_proxy = fga
            except Exception:
                pass

            # Attempt to get personal fouls from boxscoretraditionalv2 team stats
            record_home_fta = 0.0
            record_away_fta = 0.0
            try:
                from nba_api.stats.endpoints import boxscoretraditionalv2
                _rate_limit()
                trad = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=gid)
                team_stats = trad.team_stats.get_dict()
                ts_headers = team_stats.get("headers", [])
                ts_data = team_stats.get("data", [])
                ts_idx = {h: idx for idx, h in enumerate(ts_headers)}

                for row in ts_data:
                    team_id = str(row[ts_idx.get("TEAM_ID", 0)])
                    team_abbr = str(row[ts_idx.get("TEAM_ABBREVIATION", 1)]) if "TEAM_ABBREVIATION" in ts_idx else ""
                    is_home = (
                        team_id == str(game.get("home_team_id", ""))
                        or team_abbr == game.get("home_team", "")
                    )
                    pf = _safe_float(row[ts_idx["PF"]] if "PF" in ts_idx else 0)
                    fta = _safe_float(row[ts_idx["FTA"]] if "FTA" in ts_idx else 0)

                    if is_home:
                        home_fouls = pf
                        record_home_fta = fta
                    else:
                        away_fouls = pf
                        record_away_fta = fta
            except Exception:
                # Fallback: estimate fouls from FTA
                record_home_fta = 0.0
                record_away_fta = 0.0

            total_fouls = home_fouls + away_fouls
            total_pace = home_pace_proxy + away_pace_proxy

            record = {
                "game_date": game.get("game_date", ""),
                "home_team": game.get("home_team", ""),
                "away_team": game.get("away_team", ""),
                "refs": [r["full_name"] for r in refs],
                "ref_ids": [r["official_id"] for r in refs],
                "ref_details": refs,
                "home_fouls": home_fouls,
                "away_fouls": away_fouls,
                "total_fouls": total_fouls,
                "foul_differential": home_fouls - away_fouls,
                "home_fta": record_home_fta,
                "away_fta": record_away_fta,
                "home_pts": home_pts,
                "away_pts": away_pts,
                "pace_proxy_fga": total_pace,
            }

            existing[gid] = record
            fetched += 1

            # Save progress every 50 games
            if fetched % 50 == 0:
                _save_with_meta(output_path, existing, season, errors)

        except Exception as e:
            errors.append({"game_id": gid, "error": str(e)})

        _progress_bar(i + 1, total, prefix=f"  Refs {season}")

    # --- Second pass: compute referee profiles ---
    print(f"\n  Computing referee profiles for {season}...")
    ref_profiles = _compute_referee_profiles(existing)

    # Enrich each game record with computed profile stats
    for gid, record in existing.items():
        if gid.startswith("_"):
            continue
        ref_names = record.get("refs", [])
        if not ref_names:
            continue

        # Average profile across the 3 refs assigned to this game
        avg_profile = _average_ref_profiles(ref_names, ref_profiles)
        record.update(avg_profile)

    # Final save
    _save_with_meta(output_path, existing, season, errors)
    print(f"  Fetched {fetched} new, {len(existing)} total, {len(errors)} errors")
    print(f"  Referee profiles computed for {len(ref_profiles)} unique referees")

    return existing


def _compute_referee_profiles(game_data: Dict[str, dict]) -> Dict[str, dict]:
    """
    Compute aggregate referee profiles from per-game data.

    For each referee, compute:
    - total_games: number of games officiated
    - avg_total_fouls: average total fouls per game
    - avg_foul_differential: average (home - away) foul diff
    - home_foul_bias: avg_foul_diff / avg_total_fouls (positive = more calls on home)
    - home_win_rate: % of games where home team won
    - avg_pace_fga: average combined FGA (pace proxy)
    - over_tendency: fraction of games going over league avg total points
    """
    ref_stats = defaultdict(lambda: {
        "games": 0,
        "total_fouls_sum": 0.0,
        "foul_diff_sum": 0.0,
        "home_wins": 0,
        "pace_sum": 0.0,
        "total_pts_list": [],
        "fta_home_sum": 0.0,
        "fta_away_sum": 0.0,
        "tech_count": 0,
    })

    for gid, rec in game_data.items():
        if gid.startswith("_"):
            continue
        ref_names = rec.get("refs", [])
        total_fouls = _safe_float(rec.get("total_fouls", 0))
        foul_diff = _safe_float(rec.get("foul_differential", 0))
        home_pts = _safe_float(rec.get("home_pts", 0))
        away_pts = _safe_float(rec.get("away_pts", 0))
        pace = _safe_float(rec.get("pace_proxy_fga", 0))
        home_fta = _safe_float(rec.get("home_fta", 0))
        away_fta = _safe_float(rec.get("away_fta", 0))

        home_won = 1 if home_pts > away_pts else 0
        total_pts = home_pts + away_pts

        for ref_name in ref_names:
            if not ref_name:
                continue
            s = ref_stats[ref_name]
            s["games"] += 1
            s["total_fouls_sum"] += total_fouls
            s["foul_diff_sum"] += foul_diff
            s["home_wins"] += home_won
            s["pace_sum"] += pace
            s["total_pts_list"].append(total_pts)
            s["fta_home_sum"] += home_fta
            s["fta_away_sum"] += away_fta

    # League average total points for over/under tendency
    all_pts = []
    for s in ref_stats.values():
        all_pts.extend(s["total_pts_list"])
    league_avg_pts = sum(all_pts) / len(all_pts) if all_pts else 210.0

    profiles = {}
    for ref_name, s in ref_stats.items():
        n = s["games"]
        if n == 0:
            continue

        avg_fouls = s["total_fouls_sum"] / n
        avg_diff = s["foul_diff_sum"] / n
        home_wr = s["home_wins"] / n
        avg_pace = s["pace_sum"] / n
        avg_fta_home = s["fta_home_sum"] / n
        avg_fta_away = s["fta_away_sum"] / n

        # Over tendency: fraction of games above league avg
        overs = sum(1 for p in s["total_pts_list"] if p > league_avg_pts)
        over_tendency = overs / n

        # Foul rate vs league: normalize against ~42 average fouls
        league_avg_fouls = 42.0
        foul_rate_vs_league = avg_fouls / league_avg_fouls if league_avg_fouls > 0 else 1.0

        # Home FT advantage: positive means home gets more FTs
        home_ft_advantage = avg_fta_home - avg_fta_away

        # Pace impact: deviation from average pace
        # League average FGA ~ 88 per team, 176 total
        league_avg_pace = 176.0
        pace_impact = (avg_pace - league_avg_pace) / league_avg_pace if league_avg_pace > 0 else 0.0

        profiles[ref_name] = {
            "experience_games": n,
            "total_fouls_avg": round(avg_fouls, 2),
            "home_foul_bias": round(avg_diff / avg_fouls, 4) if avg_fouls > 0 else 0.0,
            "foul_rate_vs_league": round(foul_rate_vs_league, 4),
            "home_ft_advantage": round(home_ft_advantage, 2),
            "home_win_rate": round(home_wr, 4),
            "over_tendency": round(over_tendency, 4),
            "pace_impact": round(pace_impact, 4),
            "close_game_bias": 0.5,  # Placeholder — would need play-by-play data
            "tech_foul_rate": 0.3,   # Placeholder — nba_api doesn't expose this cleanly
        }

    return profiles


def _average_ref_profiles(
    ref_names: List[str], profiles: Dict[str, dict]
) -> dict:
    """Average the profiles of the referees assigned to a game."""
    keys = [
        "home_foul_bias", "total_fouls_avg", "foul_rate_vs_league",
        "home_ft_advantage", "experience_games", "over_tendency",
        "close_game_bias", "tech_foul_rate", "home_win_rate", "pace_impact",
    ]
    result = {}
    for k in keys:
        vals = []
        for name in ref_names:
            p = profiles.get(name)
            if p and k in p:
                vals.append(p[k])
        if vals:
            result[k] = round(sum(vals) / len(vals), 4)
        else:
            # Defaults that match what engine.py expects
            defaults = {
                "home_foul_bias": 0.0,
                "total_fouls_avg": 42.0,
                "foul_rate_vs_league": 1.0,
                "home_ft_advantage": 0.0,
                "experience_games": 40,
                "over_tendency": 0.5,
                "close_game_bias": 0.5,
                "tech_foul_rate": 0.3,
                "home_win_rate": 0.58,
                "pace_impact": 0.0,
            }
            result[k] = defaults.get(k, 0.0)

    return result


# ---------------------------------------------------------------------------
# 3. INJURY / INACTIVE REPORTS
# ---------------------------------------------------------------------------

def fetch_injury_data(season: str, resume: bool = True) -> Dict[str, dict]:
    """
    Fetch inactive/injured player data for all games in a season.

    Strategy:
    - Use BoxScoreSummaryV2 → InactivePlayers for inactive list
    - Use BoxScoreTraditionalV2 → PlayerStats to identify DNP players
    - Cross-reference with player season averages to estimate impact

    Returns:
        Dict mapping game_id to injury/inactive data with impact scores.
    """
    _ensure_nba_api()
    from nba_api.stats.endpoints import boxscoresummaryv2, boxscoretraditionalv2

    output_path = HISTORICAL_DIR / f"injuries-{season}.json"
    existing = {}
    if resume:
        existing = _load_json(output_path) or {}
        existing = {k: v for k, v in existing.items() if not k.startswith("_")}

    games = _load_game_ids_for_season(season)
    if not games:
        print(f"  No games found for {season}")
        return existing

    # --- First: build player season averages for impact estimation ---
    player_avgs = _load_or_fetch_player_averages(season)

    total = len(games)
    fetched = 0
    errors = []

    print(f"\n--- Injury Data: {season} ({total} games, {len(existing)} cached) ---")

    for i, game in enumerate(games):
        gid = game["game_id"]

        if gid in existing and "home_inactive" in existing[gid]:
            _progress_bar(i + 1, total, prefix=f"  Injuries {season}")
            continue

        _rate_limit()
        try:
            box = boxscoresummaryv2.BoxScoreSummaryV2(game_id=gid)

            # --- InactivePlayers endpoint ---
            home_inactive_players = []
            away_inactive_players = []

            try:
                inactive = box.inactive_players.get_dict()
                in_headers = inactive.get("headers", [])
                in_data = inactive.get("data", [])
                in_idx = {h: idx for idx, h in enumerate(in_headers)}

                for row in in_data:
                    player_id = str(row[in_idx["PLAYER_ID"]]) if "PLAYER_ID" in in_idx else ""
                    first_name = str(row[in_idx["FIRST_NAME"]]) if "FIRST_NAME" in in_idx else ""
                    last_name = str(row[in_idx["LAST_NAME"]]) if "LAST_NAME" in in_idx else ""
                    team_id = str(row[in_idx["TEAM_ID"]]) if "TEAM_ID" in in_idx else ""
                    team_abbr = str(row[in_idx["TEAM_ABBREVIATION"]]) if "TEAM_ABBREVIATION" in in_idx else ""
                    jersey = str(row[in_idx["JERSEY_NUM"]]) if "JERSEY_NUM" in in_idx else ""

                    full_name = f"{first_name} {last_name}".strip()

                    player_info = {
                        "player_id": player_id,
                        "name": full_name,
                        "team_abbr": team_abbr,
                        "jersey": jersey,
                    }

                    # Look up season averages for impact
                    avg = player_avgs.get(player_id, {})
                    player_info["season_mpg"] = avg.get("mpg", 0.0)
                    player_info["season_ppg"] = avg.get("ppg", 0.0)
                    player_info["season_gp"] = avg.get("gp", 0)

                    is_home = (
                        team_id == str(game.get("home_team_id", ""))
                        or team_abbr == game.get("home_team", "")
                    )

                    if is_home:
                        home_inactive_players.append(player_info)
                    else:
                        away_inactive_players.append(player_info)

            except Exception:
                pass  # InactivePlayers may not be available for older games

            # --- Also check BoxScoreTraditionalV2 for DNP players (did_not_play) ---
            dnp_home = []
            dnp_away = []

            _rate_limit()
            try:
                trad = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=gid)
                player_stats = trad.player_stats.get_dict()
                ps_headers = player_stats.get("headers", [])
                ps_data = player_stats.get("data", [])
                ps_idx = {h: idx for idx, h in enumerate(ps_headers)}

                for row in ps_data:
                    minutes = row[ps_idx["MIN"]] if "MIN" in ps_idx else None
                    comment = str(row[ps_idx["COMMENT"]]) if "COMMENT" in ps_idx else ""

                    # DNP detection: minutes is None/0 or comment contains DNP/DND
                    is_dnp = False
                    if minutes is None or minutes == "" or minutes == "0" or minutes == 0:
                        is_dnp = True
                    if "DNP" in comment.upper() or "DND" in comment.upper() or "NWT" in comment.upper():
                        is_dnp = True

                    if not is_dnp:
                        continue

                    player_id = str(row[ps_idx["PLAYER_ID"]]) if "PLAYER_ID" in ps_idx else ""
                    player_name = str(row[ps_idx["PLAYER_NAME"]]) if "PLAYER_NAME" in ps_idx else ""
                    team_id = str(row[ps_idx["TEAM_ID"]]) if "TEAM_ID" in ps_idx else ""
                    team_abbr = str(row[ps_idx["TEAM_ABBREVIATION"]]) if "TEAM_ABBREVIATION" in ps_idx else ""

                    avg = player_avgs.get(player_id, {})

                    dnp_info = {
                        "player_id": player_id,
                        "name": player_name,
                        "team_abbr": team_abbr,
                        "comment": comment.strip(),
                        "season_mpg": avg.get("mpg", 0.0),
                        "season_ppg": avg.get("ppg", 0.0),
                        "season_gp": avg.get("gp", 0),
                    }

                    is_home = (
                        team_id == str(game.get("home_team_id", ""))
                        or team_abbr == game.get("home_team", "")
                    )

                    if is_home:
                        dnp_home.append(dnp_info)
                    else:
                        dnp_away.append(dnp_info)

            except Exception:
                pass

            # --- Compute impact scores ---
            # Impact = sum of (player_mpg / 48) for all inactive/DNP players
            # This represents the fraction of total minutes lost
            # A star player at 36 mpg = 0.75 impact; bench player at 8 mpg = 0.17

            home_all_out = _merge_inactive_lists(home_inactive_players, dnp_home)
            away_all_out = _merge_inactive_lists(away_inactive_players, dnp_away)

            home_impact = _compute_team_impact(home_all_out)
            away_impact = _compute_team_impact(away_all_out)

            record = {
                "game_date": game.get("game_date", ""),
                "home_team": game.get("home_team", ""),
                "away_team": game.get("away_team", ""),
                "home_inactive": [p["name"] for p in home_all_out],
                "away_inactive": [p["name"] for p in away_all_out],
                "home_inactive_details": home_all_out,
                "away_inactive_details": away_all_out,
                "home_inactive_count": len(home_all_out),
                "away_inactive_count": len(away_all_out),
                "home_impact": round(home_impact, 4),
                "away_impact": round(away_impact, 4),
                "impact_differential": round(home_impact - away_impact, 4),
                "home_star_out": any(p.get("season_mpg", 0) >= 30 for p in home_all_out),
                "away_star_out": any(p.get("season_mpg", 0) >= 30 for p in away_all_out),
                "home_minutes_lost": round(sum(p.get("season_mpg", 0) for p in home_all_out), 1),
                "away_minutes_lost": round(sum(p.get("season_mpg", 0) for p in away_all_out), 1),
            }

            existing[gid] = record
            fetched += 1

            # Save progress every 50 games
            if fetched % 50 == 0:
                _save_with_meta(output_path, existing, season, errors)

        except Exception as e:
            errors.append({"game_id": gid, "error": str(e)})

        _progress_bar(i + 1, total, prefix=f"  Injuries {season}")

    # Final save
    _save_with_meta(output_path, existing, season, errors)
    print(f"  Fetched {fetched} new, {len(existing)} total, {len(errors)} errors")

    return existing


def _load_or_fetch_player_averages(season: str) -> Dict[str, dict]:
    """
    Load or fetch player season averages for impact estimation.

    Returns dict mapping player_id → {mpg, ppg, rpg, apg, gp}
    """
    cache_path = HISTORICAL_DIR / f"player-averages-{season}.json"
    cached = _load_json(cache_path)
    if cached and len(cached) > 100:
        print(f"  Loaded {len(cached)} player averages from cache")
        return cached

    _ensure_nba_api()
    from nba_api.stats.endpoints import leaguedashplayerstats

    print(f"  Fetching player season averages for {season}...")
    _rate_limit()

    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
        )
        df = stats.get_data_frames()[0]
    except Exception as e:
        print(f"  WARNING: Could not fetch player averages: {e}")
        return {}

    result = {}
    for _, row in df.iterrows():
        pid = str(row.get("PLAYER_ID", ""))
        if not pid:
            continue
        result[pid] = {
            "name": str(row.get("PLAYER_NAME", "")),
            "team": str(row.get("TEAM_ABBREVIATION", "")),
            "gp": int(row.get("GP", 0)),
            "mpg": round(_safe_float(row.get("MIN", 0)), 1),
            "ppg": round(_safe_float(row.get("PTS", 0)), 1),
            "rpg": round(_safe_float(row.get("REB", 0)), 1),
            "apg": round(_safe_float(row.get("AST", 0)), 1),
            "spg": round(_safe_float(row.get("STL", 0)), 1),
            "bpg": round(_safe_float(row.get("BLK", 0)), 1),
        }

    _save_json(cache_path, result)
    print(f"  Cached {len(result)} player averages")
    return result


def _merge_inactive_lists(
    inactive: List[dict], dnp: List[dict]
) -> List[dict]:
    """Merge inactive and DNP lists, deduplicating by player_id."""
    seen = set()
    merged = []
    for p in inactive:
        pid = p.get("player_id", "")
        if pid and pid not in seen:
            seen.add(pid)
            p["source"] = "inactive_list"
            merged.append(p)
    for p in dnp:
        pid = p.get("player_id", "")
        if pid and pid not in seen:
            seen.add(pid)
            p["source"] = "dnp_boxscore"
            merged.append(p)
    return merged


def _compute_team_impact(players_out: List[dict]) -> float:
    """
    Compute aggregate impact score for missing players.

    Impact formula:
    - Each player's impact = (mpg / 48.0) * weight
    - Weight is higher for high-minute players (stars matter more)
    - Sum is capped at 1.0

    Returns float 0.0 to 1.0 (0 = no impact, 1 = catastrophic).
    """
    if not players_out:
        return 0.0

    total = 0.0
    for p in players_out:
        mpg = _safe_float(p.get("season_mpg", 0))
        # Non-linear: stars matter disproportionately more
        # A 36 mpg player = 0.75 * 1.3 = 0.975 impact weight
        # A 12 mpg player = 0.25 * 0.8 = 0.200 impact weight
        base = mpg / 48.0
        if mpg >= 30:
            weight = 1.3  # Star player multiplier
        elif mpg >= 20:
            weight = 1.0  # Rotation player
        elif mpg >= 10:
            weight = 0.8  # Bench player
        else:
            weight = 0.5  # Deep bench / end of roster
        total += base * weight

    return min(1.0, total)


# ---------------------------------------------------------------------------
# Metadata wrapper for saves
# ---------------------------------------------------------------------------

def _save_with_meta(path: Path, data: dict, season: str, errors: list):
    """Save data with metadata fields."""
    output = dict(data)
    output["_metadata"] = {
        "season": season,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "game_count": len([k for k in data if not k.startswith("_")]),
        "error_count": len(errors),
        "errors_last_10": errors[-10:] if errors else [],
    }
    _save_json(path, output)


# ---------------------------------------------------------------------------
# PUBLIC LOADER FUNCTIONS — for use by the feature engine
# ---------------------------------------------------------------------------

def load_quarter_data(season: str) -> Dict[str, dict]:
    """
    Load quarter-by-quarter scores for a season.

    Returns dict mapping game_id to quarter score data.
    Compatible with engine.py's quarter_data parameter.

    Example:
        qd = load_quarter_data("2024-25")
        qd["0022400123"]["home_q1"]  # → 28.0
    """
    path = HISTORICAL_DIR / f"quarter-scores-{season}.json"
    data = _load_json(path)
    if data is None:
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_referee_data(season: str) -> Dict[str, dict]:
    """
    Load referee data for a season.

    Returns dict mapping game_id to referee profiles.
    Compatible with engine.py's referee_data parameter.

    Keys available per game (matching engine.py expectations):
        - home_foul_bias: float (positive = more fouls on home team)
        - total_fouls_avg: float (avg total fouls for these refs)
        - foul_rate_vs_league: float (1.0 = league average)
        - home_ft_advantage: float (home FTA - away FTA avg)
        - experience_games: int (avg games officiated)
        - over_tendency: float (fraction of games going over)
        - close_game_bias: float (placeholder 0.5)
        - tech_foul_rate: float (placeholder 0.3)
        - home_win_rate: float (home team win% with these refs)
        - pace_impact: float (FGA deviation from league avg)

    Example:
        rd = load_referee_data("2024-25")
        rd["0022400123"]["home_foul_bias"]  # → 0.03
    """
    path = HISTORICAL_DIR / f"referee-data-{season}.json"
    data = _load_json(path)
    if data is None:
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_injury_data(season: str) -> Dict[str, dict]:
    """
    Load injury/inactive data for a season.

    Returns dict mapping game_id to injury impact data.
    Compatible with engine.py's player_data parameter.

    Keys available per game:
        - home_inactive: list of player names
        - away_inactive: list of player names
        - home_impact: float 0-1 (0 = no impact, 1 = catastrophic)
        - away_impact: float 0-1
        - impact_differential: float (home - away impact)
        - home_star_out: bool (any 30+ mpg player out)
        - away_star_out: bool
        - home_minutes_lost: float (total MPG of missing players)
        - away_minutes_lost: float

    Example:
        id_ = load_injury_data("2024-25")
        id_["0022400123"]["home_impact"]  # → 0.15
    """
    path = HISTORICAL_DIR / f"injuries-{season}.json"
    data = _load_json(path)
    if data is None:
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_all_data(seasons: Optional[List[str]] = None) -> Tuple[dict, dict, dict]:
    """
    Load all three data sources across multiple seasons, merged into single dicts.

    Returns:
        (quarter_data, referee_data, injury_data) — each mapping game_id → data
    """
    if seasons is None:
        seasons = ALL_SEASONS

    quarters = {}
    referees = {}
    injuries = {}

    for season in seasons:
        quarters.update(load_quarter_data(season))
        referees.update(load_referee_data(season))
        injuries.update(load_injury_data(season))

    return quarters, referees, injuries


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch missing NBA data: quarter scores, referees, injuries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_missing_data.py                               # All data, all seasons
  python fetch_missing_data.py --seasons 2024-25             # Single season
  python fetch_missing_data.py --only quarters --seasons 2023-24 2024-25
  python fetch_missing_data.py --resume                      # Resume interrupted run
  python fetch_missing_data.py --check                       # Check what data exists
        """,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=ALL_SEASONS,
        help=f"Seasons to fetch (default: all 8 from {ALL_SEASONS[0]} to {ALL_SEASONS[-1]})",
    )
    parser.add_argument(
        "--only",
        choices=["quarters", "referees", "injuries"],
        default=None,
        help="Fetch only one data type",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from previously saved progress (default: True)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="Ignore cached data and re-fetch everything",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="Only check what data exists, don't fetch anything",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=_CONFIG["rate_limit"],
        help=f"Seconds between API calls (default: {_CONFIG['rate_limit']})",
    )

    args = parser.parse_args()

    # Apply rate limit override
    _CONFIG["rate_limit"] = args.rate_limit

    resume = not args.fresh

    # Ensure output directory exists
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.check:
        _check_data_status(args.seasons)
        return

    print("=" * 60)
    print("NBA Quant — Missing Data Fetcher")
    print("=" * 60)
    print(f"Seasons: {', '.join(args.seasons)}")
    print(f"Mode: {'single (' + args.only + ')' if args.only else 'all 3 sources'}")
    print(f"Resume: {resume}")
    print(f"Rate limit: {_CONFIG['rate_limit']}s")
    print(f"Output: {HISTORICAL_DIR}")
    print("=" * 60)

    start_time = time.time()

    for season in args.seasons:
        print(f"\n{'='*40}")
        print(f"  SEASON: {season}")
        print(f"{'='*40}")

        if args.only is None or args.only == "quarters":
            try:
                fetch_quarter_scores(season, resume=resume)
            except Exception as e:
                print(f"\n  FATAL ERROR (quarters {season}): {e}")
                traceback.print_exc()

        if args.only is None or args.only == "referees":
            try:
                fetch_referee_data(season, resume=resume)
            except Exception as e:
                print(f"\n  FATAL ERROR (referees {season}): {e}")
                traceback.print_exc()

        if args.only is None or args.only == "injuries":
            try:
                fetch_injury_data(season, resume=resume)
            except Exception as e:
                print(f"\n  FATAL ERROR (injuries {season}): {e}")
                traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")

    # Final status check
    _check_data_status(args.seasons)


def _check_data_status(seasons: List[str]):
    """Print a status table of what data exists."""
    print(f"\n{'='*60}")
    print("  DATA STATUS")
    print(f"{'='*60}")
    print(f"  {'Season':<12} {'Quarters':<15} {'Referees':<15} {'Injuries':<15}")
    print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*15}")

    for season in seasons:
        q_path = HISTORICAL_DIR / f"quarter-scores-{season}.json"
        r_path = HISTORICAL_DIR / f"referee-data-{season}.json"
        i_path = HISTORICAL_DIR / f"injuries-{season}.json"

        q_count = _count_games_in_file(q_path)
        r_count = _count_games_in_file(r_path)
        i_count = _count_games_in_file(i_path)

        q_str = f"{q_count} games" if q_count > 0 else "MISSING"
        r_str = f"{r_count} games" if r_count > 0 else "MISSING"
        i_str = f"{i_count} games" if i_count > 0 else "MISSING"

        print(f"  {season:<12} {q_str:<15} {r_str:<15} {i_str:<15}")

    print()


def _count_games_in_file(path: Path) -> int:
    """Count non-metadata entries in a JSON data file."""
    data = _load_json(path)
    if data is None:
        return 0
    return len([k for k in data if not k.startswith("_")])


if __name__ == "__main__":
    main()
