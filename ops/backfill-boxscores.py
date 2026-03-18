#!/usr/bin/env python3
"""
Backfill real box score data for historical games.
Fetches FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, PF per team
from NBA API boxscoretraditionalv2 endpoint.

Must run on HF Space (16GB RAM), NOT on VM.
"""
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime

# Try nba_api import
try:
    from nba_api.stats.endpoints import boxscoretraditionalv2
except ImportError:
    print("[ERROR] nba_api not installed. Run: pip install nba_api")
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / "data" / "historical"
PROGRESS_FILE = DATA_DIR / "backfill-progress.json"
RATE_LIMIT = 1.0  # seconds between API calls


def load_progress():
    """Load resume state from progress file."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "failed": [], "last_season": None, "last_index": 0}


def save_progress(progress):
    """Save resume state."""
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def game_has_boxscore(game):
    """Check if game already has real box score data."""
    home = game.get("home", {})
    # Real box scores have fga key (not estimated)
    return isinstance(home, dict) and "fga" in home


def fetch_boxscore(game_id):
    """Fetch real box score from NBA API."""
    try:
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        data = box.get_normalized_dict()

        # TeamStats has 2 rows: one per team
        team_stats = data.get("TeamStats", [])
        if len(team_stats) < 2:
            return None

        result = {}
        for ts in team_stats:
            team_id = str(ts.get("TEAM_ID", ""))
            team_abbr = ts.get("TEAM_ABBREVIATION", "")
            result[team_id] = {
                "team_abbr": team_abbr,
                "fgm": ts.get("FGM", 0),
                "fga": ts.get("FGA", 0),
                "fg3m": ts.get("FG3M", 0),
                "fg3a": ts.get("FG3A", 0),
                "ftm": ts.get("FTM", 0),
                "fta": ts.get("FTA", 0),
                "oreb": ts.get("OREB", 0),
                "dreb": ts.get("DREB", 0),
                "pf": ts.get("PF", 0),
                "tov": ts.get("TO", ts.get("TOV", 0)),
            }
        return result
    except Exception as e:
        print(f"  [ERROR] {game_id}: {e}")
        return None


def backfill_season(season_file, progress):
    """Backfill box scores for one season file."""
    games = json.loads(season_file.read_text())
    modified = False
    total = len(games)
    enriched = sum(1 for g in games if game_has_boxscore(g))

    print(f"\n[SEASON] {season_file.name}: {total} games, {enriched} already enriched")

    for i, game in enumerate(games):
        game_id = game.get("game_id", "")

        if not game_id:
            continue

        if game_has_boxscore(game):
            continue

        if game_id in progress["completed"]:
            continue

        if game_id in progress["failed"]:
            continue

        print(f"  [{i+1}/{total}] Fetching {game_id} ({game.get('matchup', '?')})...", end=" ")

        box = fetch_boxscore(game_id)
        if box is None:
            progress["failed"].append(game_id)
            print("FAILED")
            time.sleep(RATE_LIMIT)
            continue

        # Match box score teams to game's home/away
        home_team_id = str(game.get("home", {}).get("team_id", ""))
        away_team_id = str(game.get("away", {}).get("team_id", ""))

        home_box = box.get(home_team_id, {})
        away_box = box.get(away_team_id, {})

        # If team_id matching fails, try by abbreviation
        if not home_box:
            home_abbr = game.get("home", {}).get("team_abbr", "")
            for tid, bdata in box.items():
                if bdata.get("team_abbr") == home_abbr:
                    home_box = bdata
                    break
        if not away_box:
            away_abbr = game.get("away", {}).get("team_abbr", "")
            for tid, bdata in box.items():
                if bdata.get("team_abbr") == away_abbr:
                    away_box = bdata
                    break

        # Merge into game dict
        if home_box:
            for k, v in home_box.items():
                if k != "team_abbr":
                    game["home"][k] = v
        if away_box:
            for k, v in away_box.items():
                if k != "team_abbr":
                    game["away"][k] = v

        progress["completed"].append(game_id)
        modified = True
        print(f"OK (FGA: H={home_box.get('fga', '?')} A={away_box.get('fga', '?')})")

        # Save progress every 50 games
        if len(progress["completed"]) % 50 == 0:
            save_progress(progress)
            season_file.write_text(json.dumps(games, indent=2))
            print(f"  [CHECKPOINT] Saved {len(progress['completed'])} games")

        time.sleep(RATE_LIMIT)

    # Final save
    if modified:
        season_file.write_text(json.dumps(games, indent=2))
        print(f"  [SAVED] {season_file.name}")

    return modified


def main():
    """Main backfill loop across all seasons."""
    print("=" * 60)
    print("NBA BOX SCORE BACKFILL")
    print(f"Data dir: {DATA_DIR}")
    print(f"Start time: {datetime.now().isoformat()}")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"[ERROR] Data directory not found: {DATA_DIR}")
        sys.exit(1)

    progress = load_progress()
    print(f"[RESUME] {len(progress['completed'])} completed, {len(progress['failed'])} failed")

    season_files = sorted(DATA_DIR.glob("games-*.json"))
    if not season_files:
        print("[ERROR] No season files found")
        sys.exit(1)

    print(f"[FOUND] {len(season_files)} season files")

    for sf in season_files:
        try:
            backfill_season(sf, progress)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Saving progress...")
            save_progress(progress)
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] {sf.name}: {e}")
            save_progress(progress)

    save_progress(progress)
    print(f"\n[DONE] Backfill complete. {len(progress['completed'])} enriched, {len(progress['failed'])} failed.")
    print(f"End time: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
