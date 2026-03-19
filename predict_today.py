#!/usr/bin/env python3
"""
NBA Daily Prediction Pipeline — Real Bet Recommendations
==========================================================
Pulls today's NBA games, runs ensemble model predictions, compares
to live market odds, sizes bets via Kelly, and generates player
prop recommendations.

Pipeline:
  1. Fetch today's NBA schedule (nba_api + The Odds API)
  2. Run ensemble model (power ratings + ELO + Poisson + Monte Carlo)
  3. Pull live odds from The Odds API (h2h, spreads, totals, player_props)
  4. Compare model probability to market implied probability
  5. Calculate edge, Kelly stake, and value bets
  6. Generate player prop predictions
  7. Output comprehensive JSON + summary table

Usage:
  python3 predict_today.py                   # Full pipeline
  python3 predict_today.py --no-props        # Skip player props (faster)
  python3 predict_today.py --bankroll 500    # Custom bankroll
  python3 predict_today.py --dry-run         # No API calls, use cached/simulated data
"""

import os, sys, json, ssl, time, math, hashlib, urllib.request, traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta, date
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

# ── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Load .env.local ──────────────────────────────────────────────────────────
def _load_env():
    for env_path in [ROOT / ".env.local", Path("/home/termius/mon-ipad/.env.local")]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

_load_env()

# ── Imports from our models ──────────────────────────────────────────────────
from models.power_ratings import (
    NBA_TEAMS, get_team, predict_matchup, batch_power_rankings,
    HOME_COURT_ADVANTAGE, SIGMOID_SCALE,
)
from models.predictor import (
    ensemble_predict, elo_win_probability, poisson_predict,
    monte_carlo_predict, save_prediction, MC_SIMULATIONS,
)
from models.kelly import (
    BetOpportunity, evaluate_bet, evaluate_multiple_bets,
    implied_probability, american_to_decimal, decimal_to_american,
    kelly_fraction, edge_percentage, expected_value,
    FRACTIONAL_KELLY, DEFAULT_BANKROLL, MIN_EDGE_THRESHOLD,
)
from models.odds_analyzer import (
    fetch_live_odds, analyze_game_odds, find_value_bets,
    _match_team_name, PRIORITY_BOOKMAKERS, save_odds_snapshot,
)

# ── Constants ────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "959eab3a6b0b731ef1766579e355f51d")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

DATA_DIR = ROOT / "data"
AGENT_DIR = DATA_DIR / "nba-agent"
PREDICTIONS_DIR = DATA_DIR / "predictions"
PROPS_DIR = DATA_DIR / "player-props"
RESULTS_DIR = DATA_DIR / "results"

TODAY = date.today().isoformat()

# Star players per team — used for player prop analysis
# Tier: "superstar" (top 10 NBA), "all_star", "starter"
STAR_PLAYERS = {
    "BOS": [
        {"name": "Jayson Tatum", "pos": "SF", "tier": "superstar", "avg_pts": 27.5, "avg_ast": 5.2, "avg_reb": 8.8, "avg_3pm": 2.8},
        {"name": "Jaylen Brown", "pos": "SG", "tier": "all_star", "avg_pts": 23.0, "avg_ast": 3.5, "avg_reb": 5.5, "avg_3pm": 2.0},
        {"name": "Derrick White", "pos": "PG", "tier": "starter", "avg_pts": 16.0, "avg_ast": 4.5, "avg_reb": 4.2, "avg_3pm": 2.5},
    ],
    "CLE": [
        {"name": "Donovan Mitchell", "pos": "SG", "tier": "superstar", "avg_pts": 25.5, "avg_ast": 4.8, "avg_reb": 4.5, "avg_3pm": 3.0},
        {"name": "Darius Garland", "pos": "PG", "tier": "all_star", "avg_pts": 21.0, "avg_ast": 7.0, "avg_reb": 2.8, "avg_3pm": 2.5},
        {"name": "Evan Mobley", "pos": "C", "tier": "all_star", "avg_pts": 18.5, "avg_ast": 3.0, "avg_reb": 9.0, "avg_3pm": 0.5},
    ],
    "OKC": [
        {"name": "Shai Gilgeous-Alexander", "pos": "PG", "tier": "superstar", "avg_pts": 31.5, "avg_ast": 6.0, "avg_reb": 5.5, "avg_3pm": 1.5},
        {"name": "Jalen Williams", "pos": "SF", "tier": "all_star", "avg_pts": 21.0, "avg_ast": 5.0, "avg_reb": 5.8, "avg_3pm": 1.5},
        {"name": "Chet Holmgren", "pos": "C", "tier": "all_star", "avg_pts": 16.5, "avg_ast": 2.5, "avg_reb": 8.0, "avg_3pm": 1.5},
    ],
    "DEN": [
        {"name": "Nikola Jokic", "pos": "C", "tier": "superstar", "avg_pts": 26.5, "avg_ast": 9.5, "avg_reb": 12.5, "avg_3pm": 1.0},
        {"name": "Jamal Murray", "pos": "PG", "tier": "all_star", "avg_pts": 21.0, "avg_ast": 6.5, "avg_reb": 4.0, "avg_3pm": 2.5},
    ],
    "NYK": [
        {"name": "Jalen Brunson", "pos": "PG", "tier": "superstar", "avg_pts": 28.0, "avg_ast": 7.5, "avg_reb": 3.5, "avg_3pm": 2.0},
        {"name": "Karl-Anthony Towns", "pos": "C", "tier": "all_star", "avg_pts": 24.5, "avg_ast": 3.0, "avg_reb": 11.0, "avg_3pm": 2.0},
        {"name": "Mikal Bridges", "pos": "SF", "tier": "starter", "avg_pts": 18.0, "avg_ast": 3.5, "avg_reb": 4.5, "avg_3pm": 2.0},
    ],
    "MIL": [
        {"name": "Giannis Antetokounmpo", "pos": "PF", "tier": "superstar", "avg_pts": 31.0, "avg_ast": 6.5, "avg_reb": 11.5, "avg_3pm": 0.8},
        {"name": "Damian Lillard", "pos": "PG", "tier": "all_star", "avg_pts": 25.0, "avg_ast": 7.0, "avg_reb": 4.5, "avg_3pm": 3.5},
    ],
    "PHX": [
        {"name": "Kevin Durant", "pos": "SF", "tier": "superstar", "avg_pts": 27.0, "avg_ast": 5.0, "avg_reb": 6.5, "avg_3pm": 2.0},
        {"name": "Devin Booker", "pos": "SG", "tier": "superstar", "avg_pts": 26.5, "avg_ast": 6.5, "avg_reb": 4.5, "avg_3pm": 2.5},
    ],
    "DAL": [
        {"name": "Luka Doncic", "pos": "PG", "tier": "superstar", "avg_pts": 33.5, "avg_ast": 9.5, "avg_reb": 9.0, "avg_3pm": 3.5},
        {"name": "Kyrie Irving", "pos": "SG", "tier": "all_star", "avg_pts": 25.0, "avg_ast": 5.0, "avg_reb": 5.0, "avg_3pm": 2.5},
    ],
    "LAL": [
        {"name": "LeBron James", "pos": "SF", "tier": "superstar", "avg_pts": 25.5, "avg_ast": 8.5, "avg_reb": 7.5, "avg_3pm": 2.0},
        {"name": "Anthony Davis", "pos": "C", "tier": "superstar", "avg_pts": 24.5, "avg_ast": 3.5, "avg_reb": 12.0, "avg_3pm": 0.5},
    ],
    "GSW": [
        {"name": "Stephen Curry", "pos": "PG", "tier": "superstar", "avg_pts": 26.0, "avg_ast": 5.0, "avg_reb": 4.5, "avg_3pm": 4.5},
    ],
    "MIN": [
        {"name": "Anthony Edwards", "pos": "SG", "tier": "superstar", "avg_pts": 27.0, "avg_ast": 5.5, "avg_reb": 5.5, "avg_3pm": 3.0},
        {"name": "Rudy Gobert", "pos": "C", "tier": "all_star", "avg_pts": 14.0, "avg_ast": 1.5, "avg_reb": 12.5, "avg_3pm": 0.0},
    ],
    "MIA": [
        {"name": "Jimmy Butler", "pos": "SF", "tier": "all_star", "avg_pts": 20.0, "avg_ast": 5.5, "avg_reb": 5.5, "avg_3pm": 1.0},
        {"name": "Bam Adebayo", "pos": "C", "tier": "all_star", "avg_pts": 19.5, "avg_ast": 4.5, "avg_reb": 10.0, "avg_3pm": 0.2},
    ],
    "PHI": [
        {"name": "Joel Embiid", "pos": "C", "tier": "superstar", "avg_pts": 27.0, "avg_ast": 4.0, "avg_reb": 11.0, "avg_3pm": 1.0},
        {"name": "Tyrese Maxey", "pos": "PG", "tier": "all_star", "avg_pts": 25.0, "avg_ast": 6.0, "avg_reb": 3.5, "avg_3pm": 2.5},
    ],
    "SAC": [
        {"name": "De'Aaron Fox", "pos": "PG", "tier": "all_star", "avg_pts": 26.5, "avg_ast": 6.5, "avg_reb": 4.5, "avg_3pm": 1.5},
        {"name": "Domantas Sabonis", "pos": "C", "tier": "all_star", "avg_pts": 19.5, "avg_ast": 6.5, "avg_reb": 13.0, "avg_3pm": 0.5},
    ],
    "IND": [
        {"name": "Tyrese Haliburton", "pos": "PG", "tier": "all_star", "avg_pts": 20.0, "avg_ast": 10.5, "avg_reb": 4.0, "avg_3pm": 3.0},
        {"name": "Pascal Siakam", "pos": "PF", "tier": "all_star", "avg_pts": 21.0, "avg_ast": 4.5, "avg_reb": 7.0, "avg_3pm": 1.0},
    ],
    "ORL": [
        {"name": "Paolo Banchero", "pos": "PF", "tier": "all_star", "avg_pts": 23.0, "avg_ast": 5.5, "avg_reb": 7.5, "avg_3pm": 1.0},
        {"name": "Franz Wagner", "pos": "SF", "tier": "all_star", "avg_pts": 21.0, "avg_ast": 5.0, "avg_reb": 5.5, "avg_3pm": 1.5},
    ],
    "MEM": [
        {"name": "Ja Morant", "pos": "PG", "tier": "superstar", "avg_pts": 25.0, "avg_ast": 8.0, "avg_reb": 5.5, "avg_3pm": 1.5},
    ],
    "HOU": [
        {"name": "Jalen Green", "pos": "SG", "tier": "starter", "avg_pts": 21.5, "avg_ast": 3.5, "avg_reb": 5.0, "avg_3pm": 2.5},
        {"name": "Alperen Sengun", "pos": "C", "tier": "starter", "avg_pts": 18.5, "avg_ast": 5.0, "avg_reb": 9.5, "avg_3pm": 0.5},
    ],
    "ATL": [
        {"name": "Trae Young", "pos": "PG", "tier": "all_star", "avg_pts": 25.5, "avg_ast": 10.5, "avg_reb": 3.0, "avg_3pm": 2.5},
    ],
    "CHI": [
        {"name": "Zach LaVine", "pos": "SG", "tier": "all_star", "avg_pts": 24.0, "avg_ast": 4.5, "avg_reb": 5.0, "avg_3pm": 3.0},
    ],
    "LAC": [
        {"name": "James Harden", "pos": "PG", "tier": "all_star", "avg_pts": 22.0, "avg_ast": 8.0, "avg_reb": 5.0, "avg_3pm": 2.5},
        {"name": "Kawhi Leonard", "pos": "SF", "tier": "superstar", "avg_pts": 23.5, "avg_ast": 4.0, "avg_reb": 6.0, "avg_3pm": 1.5},
    ],
    "NOP": [
        {"name": "Zion Williamson", "pos": "PF", "tier": "all_star", "avg_pts": 23.0, "avg_ast": 5.0, "avg_reb": 6.5, "avg_3pm": 0.2},
        {"name": "Brandon Ingram", "pos": "SF", "tier": "starter", "avg_pts": 22.0, "avg_ast": 5.5, "avg_reb": 5.5, "avg_3pm": 1.5},
    ],
    "TOR": [
        {"name": "Scottie Barnes", "pos": "SF", "tier": "all_star", "avg_pts": 20.0, "avg_ast": 6.5, "avg_reb": 7.5, "avg_3pm": 1.0},
    ],
    "BKN": [
        {"name": "Cam Thomas", "pos": "SG", "tier": "starter", "avg_pts": 24.0, "avg_ast": 3.5, "avg_reb": 3.5, "avg_3pm": 2.0},
    ],
    "POR": [
        {"name": "Scoot Henderson", "pos": "PG", "tier": "starter", "avg_pts": 17.0, "avg_ast": 6.0, "avg_reb": 3.5, "avg_3pm": 1.5},
    ],
    "SAS": [
        {"name": "Victor Wembanyama", "pos": "C", "tier": "superstar", "avg_pts": 22.5, "avg_ast": 3.5, "avg_reb": 10.5, "avg_3pm": 2.0},
    ],
    "DET": [
        {"name": "Cade Cunningham", "pos": "PG", "tier": "all_star", "avg_pts": 22.5, "avg_ast": 7.5, "avg_reb": 4.5, "avg_3pm": 2.0},
    ],
    "CHA": [
        {"name": "LaMelo Ball", "pos": "PG", "tier": "all_star", "avg_pts": 23.5, "avg_ast": 8.0, "avg_reb": 5.5, "avg_3pm": 3.0},
    ],
    "WAS": [
        {"name": "Jordan Poole", "pos": "SG", "tier": "starter", "avg_pts": 18.0, "avg_ast": 4.5, "avg_reb": 3.0, "avg_3pm": 2.5},
    ],
    "UTA": [
        {"name": "Lauri Markkanen", "pos": "PF", "tier": "all_star", "avg_pts": 23.0, "avg_ast": 2.0, "avg_reb": 8.0, "avg_3pm": 2.5},
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# HTTP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def http_get(url: str, headers: dict = None, timeout: int = 30) -> Tuple[Any, int]:
    """HTTP GET with SSL bypass and error handling."""
    hdrs = headers or {}
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data, resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        return {"error": f"HTTP {e.code}: {body}"}, e.code
    except Exception as e:
        return {"error": str(e)}, 0


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: FETCH TODAY'S GAMES
# ══════════════════════════════════════════════════════════════════════════════

def fetch_todays_games_nba_api() -> List[Dict]:
    """
    Fetch today's NBA schedule from nba_api.
    Returns list of {home_team, away_team, game_id, game_time}.
    """
    try:
        from nba_api.stats.endpoints import scoreboardv2
        from nba_api.stats.library.parameters import GameDate

        today_str = datetime.now().strftime("%m/%d/%Y")
        print(f"[NBA_API] Fetching scoreboard for {today_str}...")

        sb = scoreboardv2.ScoreboardV2(
            game_date=today_str,
            day_offset=0,
            league_id="00",
        )

        # Parse GameHeader dataset
        game_header = sb.game_header.get_data_frame()

        if game_header is None or game_header.empty:
            print(f"[NBA_API] No games found for today")
            return []

        games = []
        for _, row in game_header.iterrows():
            game_id = row.get("GAME_ID", "")
            home_team_id = row.get("HOME_TEAM_ID", "")
            away_team_id = row.get("VISITOR_TEAM_ID", "")
            game_status = row.get("GAME_STATUS_TEXT", "")
            game_time_et = row.get("GAME_DATE_EST", "")

            # Map team IDs to names using the team sets
            home_name = _nba_api_team_name(home_team_id, sb)
            away_name = _nba_api_team_name(away_team_id, sb)

            if home_name and away_name:
                games.append({
                    "game_id": str(game_id),
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_team_id": int(home_team_id) if home_team_id else 0,
                    "away_team_id": int(away_team_id) if away_team_id else 0,
                    "status": str(game_status).strip(),
                    "source": "nba_api",
                })

        print(f"[NBA_API] Found {len(games)} games for today")
        return games

    except Exception as e:
        print(f"[NBA_API] Error fetching scoreboard: {e}")
        traceback.print_exc()
        return []


def _nba_api_team_name(team_id, scoreboard) -> Optional[str]:
    """Resolve NBA team ID to full team name from scoreboard data."""
    try:
        team_id = int(team_id)
        # NBA team ID to name mapping (standard NBA team IDs)
        NBA_TEAM_ID_MAP = {
            1610612737: "Atlanta Hawks", 1610612738: "Boston Celtics",
            1610612751: "Brooklyn Nets", 1610612766: "Charlotte Hornets",
            1610612741: "Chicago Bulls", 1610612739: "Cleveland Cavaliers",
            1610612742: "Dallas Mavericks", 1610612743: "Denver Nuggets",
            1610612765: "Detroit Pistons", 1610612744: "Golden State Warriors",
            1610612745: "Houston Rockets", 1610612754: "Indiana Pacers",
            1610612746: "LA Clippers", 1610612747: "Los Angeles Lakers",
            1610612763: "Memphis Grizzlies", 1610612748: "Miami Heat",
            1610612749: "Milwaukee Bucks", 1610612750: "Minnesota Timberwolves",
            1610612740: "New Orleans Pelicans", 1610612752: "New York Knicks",
            1610612760: "Oklahoma City Thunder", 1610612753: "Orlando Magic",
            1610612755: "Philadelphia 76ers", 1610612756: "Phoenix Suns",
            1610612757: "Portland Trail Blazers", 1610612758: "Sacramento Kings",
            1610612759: "San Antonio Spurs", 1610612761: "Toronto Raptors",
            1610612762: "Utah Jazz", 1610612764: "Washington Wizards",
        }
        return NBA_TEAM_ID_MAP.get(team_id)
    except (ValueError, TypeError):
        return None


def fetch_todays_games_odds_api() -> List[Dict]:
    """
    Fetch today's games from The Odds API (as fallback/supplement).
    Returns raw Odds API game objects.
    """
    if not ODDS_API_KEY:
        print("[ODDS] No API key, skipping Odds API game fetch")
        return []

    url = (
        f"{ODDS_API_BASE}/sports/basketball_nba/odds/"
        f"?apiKey={ODDS_API_KEY}"
        f"&regions=us,eu,uk,au"
        f"&markets=h2h,spreads,totals"
        f"&oddsFormat=decimal"
        f"&dateFormat=iso"
    )

    print("[ODDS] Fetching live NBA odds...")
    data, status = http_get(url, timeout=20)
    time.sleep(0.5)  # Rate limiting

    if isinstance(data, dict) and "error" in data:
        print(f"[ODDS] API Error: {data['error']}")
        return _load_cached_odds()

    if not isinstance(data, list):
        print(f"[ODDS] Unexpected response, trying cache...")
        return _load_cached_odds()

    print(f"[ODDS] Fetched {len(data)} games with live odds")

    # Cache the odds snapshot
    _cache_odds(data)

    return data


def fetch_player_props_odds_api() -> Dict[str, List[Dict]]:
    """
    Fetch player prop odds from The Odds API.
    Returns dict of game_id -> list of prop markets.
    """
    if not ODDS_API_KEY:
        print("[PROPS] No API key, will generate synthetic props")
        return {}

    prop_markets = "player_points,player_assists,player_rebounds,player_threes"
    url = (
        f"{ODDS_API_BASE}/sports/basketball_nba/odds/"
        f"?apiKey={ODDS_API_KEY}"
        f"&regions=us,eu"
        f"&markets={prop_markets}"
        f"&oddsFormat=decimal"
        f"&dateFormat=iso"
    )

    print("[PROPS] Fetching player prop odds...")
    data, status = http_get(url, timeout=20)
    time.sleep(0.5)

    if isinstance(data, dict) and "error" in data:
        print(f"[PROPS] API Error: {data['error']} — will use synthetic props")
        return {}

    if not isinstance(data, list):
        return {}

    # Organize by game ID
    props_by_game = {}
    for game in data:
        game_id = game.get("id", "")
        props_by_game[game_id] = game.get("bookmakers", [])

    print(f"[PROPS] Got player props for {len(props_by_game)} games")
    return props_by_game


def _load_cached_odds() -> List[Dict]:
    """Load most recent cached odds file."""
    odds_files = sorted(DATA_DIR.glob("odds-*.json"), reverse=True)
    if odds_files:
        try:
            data = json.loads(odds_files[0].read_text())
            if isinstance(data, list):
                print(f"[ODDS] Loaded cached odds: {odds_files[0].name} ({len(data)} games)")
                return data
        except Exception as e:
            print(f"[ODDS] Cache load error: {e}")
    return []


def _cache_odds(data: list):
    """Save odds snapshot for CLV tracking."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    out = DATA_DIR / f"odds-{ts}.json"
    try:
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: LOAD TRAINED MODEL OR USE ENSEMBLE FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def load_evolution_model() -> Optional[Dict]:
    """
    Load the latest evolution model from data/results/.
    Returns model config dict or None if no trained model exists.
    """
    evolution_file = RESULTS_DIR / "evolution-status.json"
    if evolution_file.exists():
        try:
            data = json.loads(evolution_file.read_text())
            best = data.get("best")
            if best and best.get("fitness", {}).get("composite", 0) > 0:
                print(f"[MODEL] Loaded evolution model: "
                      f"Gen {data.get('generation', '?')}, "
                      f"Brier={best['fitness'].get('brier', '?'):.4f}, "
                      f"Features={best.get('n_features', '?')}")
                return best
        except Exception as e:
            print(f"[MODEL] Error loading evolution model: {e}")

    # Check for any evolution JSON files
    for f in sorted(RESULTS_DIR.glob("evolution-*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            best = data.get("best")
            if best:
                print(f"[MODEL] Loaded evolution model from {f.name}")
                return best
        except Exception:
            continue

    print("[MODEL] No trained evolution model found — using ensemble fallback (ELO + Power + Poisson + MC)")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: GENERATE PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_evolved_predictions(games):
    """
    Fetch predictions from S10's evolved model via /api/predict.
    Returns dict: {(home, away): evolved_home_prob}
    """
    S10_URL = os.environ.get("S10_URL", "https://lbjlincoln-nomos-nba-quant.hf.space")
    results = {}
    try:
        import requests
        payload = {"games": [{"home_team": g["home"], "away_team": g["away"]} for g in games]}
        resp = requests.post(f"{S10_URL}/api/predict", json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            for pred in data.get("predictions", []):
                key = (pred.get("home_team", ""), pred.get("away_team", ""))
                results[key] = pred.get("home_win_prob", 0.5)
            print(f"[EVOLVED] Got {len(results)} predictions from S10")
        else:
            print(f"[EVOLVED] S10 returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[EVOLVED] S10 unreachable: {e}")
    return results


def predict_game(home_team: str, away_team: str, evolution_model: Optional[Dict] = None,
                 evolved_prob: Optional[float] = None) -> Dict:
    """
    Generate full prediction for a single game.
    Uses ensemble model (power ratings + ELO + Poisson + Monte Carlo).
    Blends with evolved model probability when available.
    """
    # Resolve team abbreviations
    home_abbrev = _match_team_name(home_team) or home_team
    away_abbrev = _match_team_name(away_team) or away_team

    # Run ensemble prediction
    ensemble = ensemble_predict(home_abbrev, away_abbrev)
    if not ensemble:
        # Pure fallback: use ELO only
        home_prob, away_prob = elo_win_probability(home_abbrev, away_abbrev)
        return {
            "home": home_abbrev,
            "away": away_abbrev,
            "home_win_prob": round(home_prob, 4),
            "away_win_prob": round(away_prob, 4),
            "predicted_spread": 0.0,
            "predicted_total": 220.0,
            "confidence": "LOW",
            "model_version": "elo-fallback",
            "individual_models": {},
        }

    # Extract key values
    ensemble_prob = ensemble["ensemble_home_win_prob"]
    predicted_spread = ensemble.get("predicted_spread", 0)
    predicted_total = ensemble.get("predicted_total", 220)

    # Blend with evolved model
    model_version = "ensemble-v1"
    final_prob = ensemble_prob

    if evolved_prob is not None:
        # Direct blend: 60% evolved + 40% ensemble
        final_prob = 0.6 * evolved_prob + 0.4 * ensemble_prob
        model_version = "blended-v1"
    elif evolution_model:
        evo_fitness = evolution_model.get("fitness", {})
        if evo_fitness.get("brier", 1.0) < 0.25:
            model_version = f"evolution-gen{evolution_model.get('generation', '?')}"

    return {
        "home": home_abbrev,
        "away": away_abbrev,
        "home_name": ensemble.get("home_name", home_team),
        "away_name": ensemble.get("away_name", away_team),
        "home_win_prob": round(final_prob, 4),
        "away_win_prob": round(1 - final_prob, 4),
        "ensemble_prob": round(ensemble_prob, 4),
        "evolved_prob": round(evolved_prob, 4) if evolved_prob is not None else None,
        "predicted_spread": round(predicted_spread, 1),
        "predicted_total": round(predicted_total, 1),
        "confidence": ensemble.get("confidence", "MEDIUM"),
        "model_agreement": ensemble.get("model_agreement", 0.5),
        "model_version": model_version,
        "individual_models": ensemble.get("individual_models", {}),
        "confidence_interval_90": ensemble.get("confidence_interval_90", {}),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: COMPARE WITH MARKET ODDS
# ══════════════════════════════════════════════════════════════════════════════

def extract_market_data(odds_game: Dict, home_abbrev: str, away_abbrev: str) -> Dict:
    """
    Extract best odds, spread, and total from Odds API game data.
    Returns structured market data dict.
    """
    analysis = analyze_game_odds(odds_game)

    market = {
        "h2h": {"home": {}, "away": {}, "best_home_odds": 0, "best_away_odds": 0},
        "spread": {"home": {}, "away": {}},
        "total": {"over": {}, "under": {}, "line": 0},
    }

    # H2H moneyline
    h2h = analysis["markets"].get("h2h", {})
    for outcome_name, best in h2h.get("best_odds", {}).items():
        matched = _match_team_name(outcome_name)
        if matched == home_abbrev:
            market["h2h"]["home"] = best
            market["h2h"]["best_home_odds"] = best["price"]
        elif matched == away_abbrev:
            market["h2h"]["away"] = best
            market["h2h"]["best_away_odds"] = best["price"]

    # Spreads
    spreads = analysis["markets"].get("spreads", {})
    for outcome_name, best in spreads.get("best_odds", {}).items():
        matched = _match_team_name(outcome_name.split("(")[0].strip())
        if matched == home_abbrev:
            market["spread"]["home"] = best
        elif matched == away_abbrev:
            market["spread"]["away"] = best

    # Totals
    totals = analysis["markets"].get("totals", {})
    for outcome_name, best in totals.get("best_odds", {}).items():
        if "over" in outcome_name.lower():
            market["total"]["over"] = best
            market["total"]["line"] = best.get("point", 0)
        elif "under" in outcome_name.lower():
            market["total"]["under"] = best

    return market


def calculate_edge_and_kelly(model_prob: float, decimal_odds: float, bankroll: float) -> Dict:
    """Calculate edge, Kelly stake, and expected value."""
    if decimal_odds <= 1.0:
        return {"edge": 0, "kelly_stake": 0, "ev": 0, "implied_prob": 0}

    impl_prob = implied_probability(decimal_odds)
    edge = model_prob - impl_prob
    ev = edge_percentage(decimal_odds, model_prob)
    kelly = kelly_fraction(decimal_odds, model_prob)
    frac_kelly = kelly * FRACTIONAL_KELLY  # 1/4 Kelly for safety
    capped_kelly = max(0, min(frac_kelly, 0.05))  # Cap at 5%

    return {
        "edge": round(edge, 4),
        "kelly_stake": round(capped_kelly, 4),
        "kelly_bet": round(capped_kelly * bankroll, 2),
        "ev": round(ev, 4),
        "implied_prob": round(impl_prob, 4),
        "full_kelly": round(kelly, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: PLAYER PROP PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def generate_player_props(
    home_abbrev: str,
    away_abbrev: str,
    prediction: Dict,
    props_odds: Dict = None,
) -> List[Dict]:
    """
    Generate player prop predictions for key players in a game.

    Uses:
    - Player season averages
    - Matchup pace adjustments
    - Home/away splits
    - Opponent defensive rating adjustments
    """
    props = []

    # Get team data for matchup context
    _, home_data = get_team(home_abbrev)
    _, away_data = get_team(away_abbrev)

    if not home_data or not away_data:
        return props

    league_avg_drtg = 113.5
    home_pace = home_data.get("pace", 100)
    away_pace = away_data.get("pace", 100)
    game_pace = (home_pace + away_pace) / 2.0
    pace_factor = game_pace / 100.0  # >1 = fast game, <1 = slow

    home_drtg = home_data.get("drtg", league_avg_drtg)
    away_drtg = away_data.get("drtg", league_avg_drtg)

    predicted_total = prediction.get("predicted_total", 220)
    total_pace_mult = predicted_total / 225.0  # Normalize to ~average total

    # Generate props for both teams
    for team_abbrev, opp_drtg, is_home in [
        (home_abbrev, away_drtg, True),
        (away_abbrev, home_drtg, False),
    ]:
        players = STAR_PLAYERS.get(team_abbrev, [])
        for player in players:
            # Defensive adjustment: opponent's DRTG vs league average
            def_adj = opp_drtg / league_avg_drtg  # >1 = bad defense = more points
            home_boost = 1.02 if is_home else 0.98  # Small home court boost

            # Points prediction
            base_pts = player["avg_pts"]
            pred_pts = base_pts * def_adj * pace_factor * home_boost * total_pace_mult
            pred_pts = round(pred_pts, 1)

            # Assists prediction
            base_ast = player["avg_ast"]
            pred_ast = base_ast * pace_factor * home_boost
            pred_ast = round(pred_ast, 1)

            # Rebounds prediction
            base_reb = player["avg_reb"]
            pred_reb = base_reb * pace_factor * home_boost
            pred_reb = round(pred_reb, 1)

            # 3PM prediction
            base_3pm = player["avg_3pm"]
            pred_3pm = base_3pm * pace_factor * home_boost
            pred_3pm = round(pred_3pm, 1)

            # Generate prop predictions for each market
            for market, avg_val, pred_val in [
                ("points", base_pts, pred_pts),
                ("assists", base_ast, pred_ast),
                ("rebounds", base_reb, pred_reb),
                ("threes", base_3pm, pred_3pm),
            ]:
                if avg_val < 1.0:
                    continue  # Skip very low-volume stats

                # Typical sportsbook line is close to the season average
                # We add some noise to simulate real prop lines
                import random
                line_noise = random.uniform(-1.0, 1.0) if market != "threes" else random.uniform(-0.5, 0.5)
                prop_line = round(avg_val + line_noise * 0.5, 1)

                # Ensure reasonable lines
                if market == "threes":
                    prop_line = max(0.5, prop_line)
                else:
                    prop_line = max(1.5, prop_line)

                # Calculate edge
                diff = pred_val - prop_line
                # Estimate confidence from the difference relative to variance
                # NBA player stat std dev: PTS ~6, AST ~2.5, REB ~3, 3PM ~1.2
                stat_stdev = {"points": 6.0, "assists": 2.5, "rebounds": 3.0, "threes": 1.2}
                stdev = stat_stdev.get(market, 5.0)
                z_score = diff / stdev
                # Convert z-score to directional probability (cumulative normal)
                prob_over = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

                pick = "OVER" if diff > 0 else "UNDER"
                confidence = abs(prob_over - 0.5) * 2  # 0 to 1 scale
                edge_val = abs(diff) / max(prop_line, 1)

                # Only include props with meaningful predictions
                if confidence > 0.05:
                    props.append({
                        "player": player["name"],
                        "team": team_abbrev,
                        "market": market,
                        "line": prop_line,
                        "prediction": pred_val,
                        "pick": pick,
                        "confidence": round(confidence, 4),
                        "edge": round(edge_val, 4),
                        "z_score": round(z_score, 2),
                    })

    # Sort by confidence (highest first)
    props.sort(key=lambda p: p["confidence"], reverse=True)
    return props


def match_real_prop_lines(
    props: List[Dict],
    game_props_odds: List[Dict],
    home_team: str,
    away_team: str,
) -> List[Dict]:
    """
    Match our predictions to real prop lines from The Odds API.
    Updates the prop line and calculates real edge.
    """
    if not game_props_odds:
        return props

    # Build lookup of real lines: {(player_name_lower, market_key): best_line}
    real_lines = {}
    market_key_map = {
        "player_points": "points",
        "player_assists": "assists",
        "player_rebounds": "rebounds",
        "player_threes": "threes",
    }

    for bookmaker in game_props_odds:
        for market in bookmaker.get("markets", []):
            mkey = market_key_map.get(market.get("key", ""))
            if not mkey:
                continue
            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description", outcome.get("name", "")).lower()
                point = outcome.get("point")
                price = outcome.get("price", 1.91)
                if point is not None and player_name:
                    key = (player_name, mkey)
                    if key not in real_lines or price > real_lines[key].get("best_price", 0):
                        real_lines[key] = {
                            "line": point,
                            "best_price": price,
                            "bookmaker": bookmaker.get("key", ""),
                        }

    # Update props with real lines
    for prop in props:
        key = (prop["player"].lower(), prop["market"])
        if key in real_lines:
            real = real_lines[key]
            prop["line"] = real["line"]
            prop["bookmaker"] = real["bookmaker"]
            prop["decimal_odds"] = real["best_price"]

            # Recalculate edge with real line
            diff = prop["prediction"] - real["line"]
            stat_stdev = {"points": 6.0, "assists": 2.5, "rebounds": 3.0, "threes": 1.2}
            stdev = stat_stdev.get(prop["market"], 5.0)
            z_score = diff / stdev
            prob_over = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

            prop["pick"] = "OVER" if diff > 0 else "UNDER"
            prop["confidence"] = round(abs(prob_over - 0.5) * 2, 4)
            prop["edge"] = round(abs(diff) / max(real["line"], 1), 4)
            prop["z_score"] = round(z_score, 2)

    return props


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: BUILD COMPLETE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def build_predictions_output(
    games_nba: List[Dict],
    odds_data: List[Dict],
    props_data: Dict[str, List[Dict]],
    evolution_model: Optional[Dict],
    bankroll: float,
    include_props: bool = True,
) -> Dict:
    """
    Build the complete predictions JSON output.
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    model_version = "ensemble-v1"
    if evolution_model:
        gen = evolution_model.get("generation", "?")
        model_version = f"evolution-gen{gen}"

    # Build lookup from odds data: {(home_team_lower, away_team_lower): odds_game}
    odds_lookup = {}
    for og in odds_data:
        home_key = og.get("home_team", "").lower()
        away_key = og.get("away_team", "").lower()
        odds_lookup[(home_key, away_key)] = og

    # Merge game lists (nba_api games + any odds-only games)
    all_games = []
    seen_matchups = set()

    # Start with nba_api games
    for game in games_nba:
        home = game["home_team"]
        away = game["away_team"]
        home_abbrev = _match_team_name(home) or home
        away_abbrev = _match_team_name(away) or away
        key = (home_abbrev, away_abbrev)
        if key not in seen_matchups:
            seen_matchups.add(key)
            all_games.append({
                "home_team_full": home,
                "away_team_full": away,
                "home_abbrev": home_abbrev,
                "away_abbrev": away_abbrev,
                "game_id": game.get("game_id", ""),
                "status": game.get("status", ""),
            })

    # Add any odds-only games not already seen
    for og in odds_data:
        home = og.get("home_team", "")
        away = og.get("away_team", "")
        home_abbrev = _match_team_name(home)
        away_abbrev = _match_team_name(away)
        if home_abbrev and away_abbrev:
            key = (home_abbrev, away_abbrev)
            if key not in seen_matchups:
                seen_matchups.add(key)
                all_games.append({
                    "home_team_full": home,
                    "away_team_full": away,
                    "home_abbrev": home_abbrev,
                    "away_abbrev": away_abbrev,
                    "game_id": og.get("id", ""),
                    "commence_time": og.get("commence_time", ""),
                })

    # Generate predictions for each game
    output_games = []
    all_value_bets = []

    for game_info in all_games:
        home_abbrev = game_info["home_abbrev"]
        away_abbrev = game_info["away_abbrev"]
        home_full = game_info["home_team_full"]
        away_full = game_info["away_team_full"]

        print(f"\n{'='*60}")
        print(f"  {away_full} @ {home_full}")
        print(f"{'='*60}")

        # Run model prediction
        pred = predict_game(home_full, away_full, evolution_model)

        # Find matching odds data
        odds_game = None
        for og in odds_data:
            og_home = _match_team_name(og.get("home_team", ""))
            og_away = _match_team_name(og.get("away_team", ""))
            if og_home == home_abbrev and og_away == away_abbrev:
                odds_game = og
                break

        # Extract market data
        market = extract_market_data(odds_game, home_abbrev, away_abbrev) if odds_game else None

        # Build game entry
        game_entry = {
            "home": home_abbrev,
            "away": away_abbrev,
            "home_name": pred.get("home_name", home_full),
            "away_name": pred.get("away_name", away_full),
            "home_win_prob": pred["home_win_prob"],
            "away_win_prob": pred.get("away_win_prob", 1 - pred["home_win_prob"]),
            "confidence": pred.get("confidence", "MEDIUM"),
            "model_agreement": pred.get("model_agreement", 0.5),
        }

        # Market comparison
        if market:
            best_home_odds = market["h2h"].get("best_home_odds", 0)
            best_away_odds = market["h2h"].get("best_away_odds", 0)

            # Home team market implied prob
            home_implied = implied_probability(best_home_odds) if best_home_odds > 1 else 0.5
            away_implied = implied_probability(best_away_odds) if best_away_odds > 1 else 0.5

            game_entry["market_implied"] = round(home_implied, 4)
            game_entry["edge"] = round(pred["home_win_prob"] - home_implied, 4)

            # Kelly for best side
            if pred["home_win_prob"] > home_implied and best_home_odds > 1:
                kelly_data = calculate_edge_and_kelly(pred["home_win_prob"], best_home_odds, bankroll)
                game_entry["kelly_stake"] = kelly_data["kelly_stake"]
                best_book = market["h2h"].get("home", {}).get("bookmaker", "")
                game_entry["best_odds"] = {
                    "book": best_book,
                    "odds": decimal_to_american(best_home_odds),
                    "decimal": best_home_odds,
                }
                game_entry["bet_side"] = "HOME"

                # Value bet?
                if kelly_data["edge"] >= MIN_EDGE_THRESHOLD:
                    all_value_bets.append({
                        "game": f"{away_abbrev} @ {home_abbrev}",
                        "type": "moneyline",
                        "pick": f"{home_abbrev} ML",
                        "odds": decimal_to_american(best_home_odds),
                        "book": best_book,
                        "edge": kelly_data["edge"],
                        "kelly": kelly_data["kelly_stake"],
                        "kelly_bet": kelly_data["kelly_bet"],
                        "ev": kelly_data["ev"],
                        "model_prob": pred["home_win_prob"],
                        "implied_prob": kelly_data["implied_prob"],
                        "confidence": pred.get("confidence", "MEDIUM"),
                    })

            elif pred.get("away_win_prob", 0) > away_implied and best_away_odds > 1:
                kelly_data = calculate_edge_and_kelly(pred["away_win_prob"], best_away_odds, bankroll)
                game_entry["kelly_stake"] = kelly_data["kelly_stake"]
                best_book = market["h2h"].get("away", {}).get("bookmaker", "")
                game_entry["best_odds"] = {
                    "book": best_book,
                    "odds": decimal_to_american(best_away_odds),
                    "decimal": best_away_odds,
                }
                game_entry["bet_side"] = "AWAY"

                if kelly_data["edge"] >= MIN_EDGE_THRESHOLD:
                    all_value_bets.append({
                        "game": f"{away_abbrev} @ {home_abbrev}",
                        "type": "moneyline",
                        "pick": f"{away_abbrev} ML",
                        "odds": decimal_to_american(best_away_odds),
                        "book": best_book,
                        "edge": kelly_data["edge"],
                        "kelly": kelly_data["kelly_stake"],
                        "kelly_bet": kelly_data["kelly_bet"],
                        "ev": kelly_data["ev"],
                        "model_prob": pred["away_win_prob"],
                        "implied_prob": kelly_data["implied_prob"],
                        "confidence": pred.get("confidence", "MEDIUM"),
                    })
            else:
                game_entry["kelly_stake"] = 0
                game_entry["best_odds"] = {}
                game_entry["bet_side"] = "PASS"

            # Spread analysis
            home_spread_data = market["spread"].get("home", {})
            spread_line = home_spread_data.get("point", 0) if home_spread_data else 0
            model_spread = pred.get("predicted_spread", 0)
            spread_edge = abs(model_spread) - abs(spread_line) if spread_line != 0 else 0

            game_entry["spread"] = {
                "line": spread_line,
                "model_spread": model_spread,
                "edge": round(spread_edge, 1),
            }

            # If spread edge is significant, add to value bets
            if abs(spread_edge) >= 2.0 and spread_line != 0:
                spread_pick_side = home_abbrev if model_spread < spread_line else away_abbrev
                spread_pick_direction = "covers" if (model_spread < spread_line) == (spread_line < 0) else "covers"
                all_value_bets.append({
                    "game": f"{away_abbrev} @ {home_abbrev}",
                    "type": "spread",
                    "pick": f"{home_abbrev} {spread_line:+g}" if model_spread < spread_line else f"{away_abbrev} {-spread_line:+g}",
                    "edge": round(abs(spread_edge) / max(abs(spread_line), 1), 4),
                    "kelly": round(FRACTIONAL_KELLY * 0.02, 4),  # Conservative for spreads
                    "kelly_bet": round(bankroll * FRACTIONAL_KELLY * 0.02, 2),
                    "model_spread": model_spread,
                    "market_spread": spread_line,
                    "confidence": pred.get("confidence", "MEDIUM"),
                })

            # Total analysis
            total_line = market["total"].get("line", 0)
            model_total = pred.get("predicted_total", 220)
            total_edge = model_total - total_line if total_line > 0 else 0
            total_pick = "OVER" if total_edge > 0 else "UNDER"

            game_entry["total"] = {
                "line": total_line,
                "model_total": model_total,
                "edge": round(total_edge, 1),
                "pick": total_pick,
            }

            # If total edge is significant, add to value bets
            if abs(total_edge) >= 3.0 and total_line > 0:
                all_value_bets.append({
                    "game": f"{away_abbrev} @ {home_abbrev}",
                    "type": "total",
                    "pick": f"{total_pick} {total_line}",
                    "edge": round(abs(total_edge) / total_line, 4),
                    "kelly": round(FRACTIONAL_KELLY * 0.015, 4),
                    "kelly_bet": round(bankroll * FRACTIONAL_KELLY * 0.015, 2),
                    "model_total": model_total,
                    "market_total": total_line,
                    "confidence": pred.get("confidence", "MEDIUM"),
                })

        else:
            # No odds data available
            game_entry["market_implied"] = None
            game_entry["edge"] = None
            game_entry["kelly_stake"] = 0
            game_entry["best_odds"] = {}
            game_entry["bet_side"] = "NO_ODDS"
            game_entry["spread"] = {"line": 0, "model_spread": pred.get("predicted_spread", 0), "edge": 0}
            game_entry["total"] = {
                "line": 0,
                "model_total": pred.get("predicted_total", 220),
                "edge": 0,
                "pick": "N/A",
            }

        # Player props
        if include_props:
            props = generate_player_props(home_abbrev, away_abbrev, pred)

            # Try to match with real prop odds
            if odds_game:
                game_odds_id = odds_game.get("id", "")
                real_props = props_data.get(game_odds_id, [])
                if real_props:
                    props = match_real_prop_lines(props, real_props, home_full, away_full)

            # Only keep top props per game (most confident)
            top_props = [p for p in props if p["confidence"] >= 0.10][:10]
            game_entry["player_props"] = top_props

            # Add high-confidence props to value bets
            for prop in top_props:
                if prop["confidence"] >= 0.25 and prop["edge"] >= 0.05:
                    all_value_bets.append({
                        "game": f"{away_abbrev} @ {home_abbrev}",
                        "type": "player_prop",
                        "pick": f"{prop['player']} {prop['pick']} {prop['line']} {prop['market'].upper()}",
                        "edge": prop["edge"],
                        "kelly": round(FRACTIONAL_KELLY * prop["confidence"] * 0.02, 4),
                        "kelly_bet": round(bankroll * FRACTIONAL_KELLY * prop["confidence"] * 0.02, 2),
                        "prediction": prop["prediction"],
                        "line": prop["line"],
                        "confidence": _confidence_label(prop["confidence"]),
                    })
        else:
            game_entry["player_props"] = []

        output_games.append(game_entry)

        # Print game summary
        _print_game_summary(game_entry)

    # Sort value bets by edge (highest first)
    all_value_bets.sort(key=lambda vb: vb.get("edge", 0), reverse=True)

    output = {
        "date": TODAY,
        "generated_at": generated_at,
        "model_version": model_version,
        "bankroll": bankroll,
        "games_count": len(output_games),
        "games": output_games,
        "value_bets": all_value_bets,
        "value_bets_count": len(all_value_bets),
        "metadata": {
            "odds_api_key_active": bool(ODDS_API_KEY),
            "nba_api_games": len(games_nba),
            "odds_api_games": len(odds_data),
            "evolution_model_loaded": evolution_model is not None,
            "include_props": include_props,
        },
    }

    return output


def _confidence_label(confidence: float) -> str:
    """Map numeric confidence to label."""
    if confidence >= 0.5:
        return "VERY HIGH"
    elif confidence >= 0.35:
        return "HIGH"
    elif confidence >= 0.2:
        return "MEDIUM"
    else:
        return "LOW"


def _print_game_summary(game: Dict):
    """Print a compact summary for one game."""
    home = game.get("home", "?")
    away = game.get("away", "?")
    home_prob = game.get("home_win_prob", 0.5)
    conf = game.get("confidence", "?")
    edge = game.get("edge")
    spread = game.get("spread", {})
    total = game.get("total", {})

    edge_str = f"{edge:+.1%}" if edge is not None else "N/A"
    spread_line = spread.get("line", 0)
    model_spread = spread.get("model_spread", 0)
    total_line = total.get("line", 0)
    model_total = total.get("model_total", 0)
    total_pick = total.get("pick", "N/A")

    print(f"  Home Win:   {home_prob:.1%} | Edge: {edge_str} | Conf: {conf}")
    print(f"  Spread:     {spread_line:+g} (model: {model_spread:+.1f})")
    print(f"  Total:      {total_line} (model: {model_total:.1f}) -> {total_pick}")

    props = game.get("player_props", [])
    if props:
        print(f"  Top Props:  {len(props)} predictions")
        for p in props[:3]:
            print(f"    {p['player']}: {p['pick']} {p['line']} {p['market']} "
                  f"(pred: {p['prediction']}, conf: {p['confidence']:.0%})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: SAVE & DISTRIBUTE
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(output: Dict):
    """Save prediction outputs to disk."""
    # Ensure directories exist
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    PROPS_DIR.mkdir(parents=True, exist_ok=True)

    # Main predictions file
    pred_file = AGENT_DIR / "predictions-today.json"
    pred_file.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    print(f"\n[SAVE] predictions-today.json -> {pred_file}")

    # Value bets only
    vb_file = AGENT_DIR / "value-bets.json"
    vb_data = {
        "date": output["date"],
        "generated_at": output["generated_at"],
        "bankroll": output["bankroll"],
        "value_bets": output["value_bets"],
        "count": output["value_bets_count"],
    }
    vb_file.write_text(json.dumps(vb_data, indent=2, ensure_ascii=False, default=str))
    print(f"[SAVE] value-bets.json -> {vb_file}")

    # Date-stamped archive
    archive_file = PREDICTIONS_DIR / f"predictions-{TODAY}.json"
    archive_file.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    print(f"[SAVE] Archive -> {archive_file}")

    # Player props archive
    all_props = []
    for game in output.get("games", []):
        for prop in game.get("player_props", []):
            prop["game"] = f"{game['away']} @ {game['home']}"
            prop["date"] = TODAY
            all_props.append(prop)

    if all_props:
        props_file = PROPS_DIR / f"props-{TODAY}.json"
        props_file.write_text(json.dumps(all_props, indent=2, ensure_ascii=False, default=str))
        print(f"[SAVE] Player props -> {props_file}")

    # Store predictions to Supabase for evaluation tracking
    store_predictions_supabase(output)


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE PREDICTION STORAGE
# ══════════════════════════════════════════════════════════════════════════════

def _get_supabase_conn():
    """Get a psycopg2 connection to Supabase. Returns None if unavailable."""
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return None
    try:
        import psycopg2
        conn = psycopg2.connect(db_url, options="-c search_path=public")
        return conn
    except Exception as e:
        print(f"[SUPABASE] Connection failed: {e}")
        return None


def store_predictions_supabase(output: Dict):
    """Store today's predictions in nba_predictions table for later evaluation."""
    conn = _get_supabase_conn()
    if not conn:
        print("[SUPABASE] Skipping — no DATABASE_URL")
        return

    games = output.get("games", [])
    if not games:
        print("[SUPABASE] No games to store")
        conn.close()
        return

    model_version = output.get("model_version", "unknown")
    game_date = output.get("date", TODAY)
    stored = 0

    try:
        with conn.cursor() as cur:
            for g in games:
                home = g.get("home", "")
                away = g.get("away", "")
                home_prob = g.get("home_win_prob", 0.5)
                market_prob = g.get("market_implied")
                edge = g.get("edge")
                confidence = g.get("confidence", "")
                best_odds = g.get("best_odds", {})
                odds_home = best_odds.get("decimal") if isinstance(best_odds, dict) else None
                odds_away = None  # single best odds for now

                # Check if prediction already exists for this game
                cur.execute("""
                    SELECT id FROM nba_predictions
                    WHERE game_date = %s AND home_team = %s AND away_team = %s
                """, (game_date, home, away))
                existing = cur.fetchone()

                if existing:
                    # Update with latest prediction
                    cur.execute("""
                        UPDATE nba_predictions SET
                            predicted_home_prob = %s, market_home_prob = %s,
                            edge = %s, market_odds_home = %s, market_odds_away = %s,
                            confidence = %s, model_version = %s
                        WHERE id = %s
                    """, (home_prob, market_prob, edge, odds_home, odds_away,
                          confidence, model_version, existing[0]))
                else:
                    cur.execute("""
                        INSERT INTO nba_predictions
                            (game_date, home_team, away_team, predicted_home_prob,
                             market_home_prob, edge, market_odds_home, market_odds_away,
                             confidence, model_version)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (game_date, home, away, home_prob,
                          market_prob, edge, odds_home, odds_away,
                          confidence, model_version))
                stored += 1

            conn.commit()
        print(f"[SUPABASE] Stored {stored} predictions in nba_predictions")
    except Exception as e:
        print(f"[SUPABASE] Error storing predictions: {e}")
        conn.rollback()
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(output: Dict):
    """Print a comprehensive summary table of all predictions."""
    games = output.get("games", [])
    value_bets = output.get("value_bets", [])

    print(f"\n{'='*80}")
    print(f"  NBA PREDICTIONS — {output['date']} | Model: {output['model_version']}")
    print(f"  Generated: {output['generated_at'][:19]}Z | Bankroll: ${output['bankroll']:.0f}")
    print(f"{'='*80}\n")

    # Games table
    print(f"  {'GAME':<25s} {'HOME%':>7s} {'EDGE':>7s} {'SPREAD':>8s} {'MODEL':>7s} {'TOTAL':>7s} {'PICK':>6s} {'CONF':>10s}")
    print(f"  {'-'*77}")

    for g in games:
        matchup = f"{g['away']} @ {g['home']}"
        home_pct = f"{g['home_win_prob']:.0%}"
        edge = g.get("edge")
        edge_str = f"{edge:+.1%}" if edge is not None else "N/A"
        spread = g.get("spread", {})
        spread_str = f"{spread.get('line', 0):+g}"
        model_spread = f"{spread.get('model_spread', 0):+.1f}"
        total = g.get("total", {})
        total_str = f"{total.get('line', 0):.0f}" if total.get("line") else "N/A"
        pick = total.get("pick", "N/A")
        conf = g.get("confidence", "?")

        print(f"  {matchup:<25s} {home_pct:>7s} {edge_str:>7s} {spread_str:>8s} {model_spread:>7s} {total_str:>7s} {pick:>6s} {conf:>10s}")

    # Value bets table
    if value_bets:
        print(f"\n  {'='*77}")
        print(f"  VALUE BETS ({len(value_bets)} found)")
        print(f"  {'='*77}\n")
        print(f"  {'#':>3s} {'GAME':<20s} {'TYPE':<12s} {'PICK':<30s} {'EDGE':>7s} {'KELLY':>7s} {'BET':>8s} {'CONF':>10s}")
        print(f"  {'-'*97}")

        for i, vb in enumerate(value_bets, 1):
            game = vb.get("game", "?")
            bet_type = vb.get("type", "?")
            pick = vb.get("pick", "?")
            edge = vb.get("edge", 0)
            kelly = vb.get("kelly", 0)
            kelly_bet = vb.get("kelly_bet", 0)
            conf = vb.get("confidence", "?")

            print(f"  {i:>3d} {game:<20s} {bet_type:<12s} {pick:<30s} {edge:>6.1%} {kelly:>6.2%} ${kelly_bet:>7.2f} {conf:>10s}")

        total_risk = sum(vb.get("kelly_bet", 0) for vb in value_bets)
        print(f"\n  Total Exposure: ${total_risk:.2f} ({total_risk/output['bankroll']:.1%} of bankroll)")
    else:
        print(f"\n  No value bets found for today.")

    # Player props summary
    total_props = sum(len(g.get("player_props", [])) for g in games)
    high_conf_props = sum(
        1 for g in games
        for p in g.get("player_props", [])
        if p.get("confidence", 0) >= 0.25
    )

    if total_props > 0:
        print(f"\n  Player Props: {total_props} total, {high_conf_props} high-confidence")
        # Show top 5 props
        all_props = []
        for g in games:
            for p in g.get("player_props", []):
                p["_game"] = f"{g['away']}@{g['home']}"
                all_props.append(p)

        all_props.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        if all_props:
            print(f"\n  Top Player Props:")
            print(f"  {'PLAYER':<22s} {'GAME':<12s} {'MARKET':<8s} {'LINE':>6s} {'PRED':>6s} {'PICK':>6s} {'CONF':>6s}")
            print(f"  {'-'*68}")
            for p in all_props[:8]:
                print(f"  {p['player']:<22s} {p['_game']:<12s} {p['market']:<8s} "
                      f"{p['line']:>6.1f} {p['prediction']:>6.1f} {p['pick']:>6s} {p['confidence']:>5.0%}")

    print(f"\n{'='*80}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(bankroll: float = 100.0, include_props: bool = True, dry_run: bool = False):
    """
    Run the complete daily prediction pipeline.

    Args:
        bankroll: Current bankroll in USD
        include_props: Whether to generate player prop predictions
        dry_run: If True, skip live API calls and use cached/simulated data
    """
    start_time = time.time()

    print(f"\n{'#'*80}")
    print(f"#  NBA DAILY PREDICTION PIPELINE — {TODAY}")
    print(f"#  Bankroll: ${bankroll:.0f} | Props: {'ON' if include_props else 'OFF'} | Dry Run: {dry_run}")
    print(f"{'#'*80}\n")

    # Step 1: Load evolution model (or fallback)
    print("[STEP 1] Loading model...")
    evolution_model = load_evolution_model()

    # Step 2: Fetch today's games from nba_api
    print("\n[STEP 2] Fetching today's NBA schedule...")
    if dry_run:
        games_nba = []
        print("[DRY RUN] Skipping nba_api")
    else:
        games_nba = fetch_todays_games_nba_api()
        time.sleep(1.0)  # Rate limiting for nba_api

    # Step 3: Fetch live odds from The Odds API
    print("\n[STEP 3] Fetching live odds...")
    if dry_run:
        odds_data = _load_cached_odds()
        if not odds_data:
            odds_data = fetch_live_odds()  # Will use simulated if no key
    else:
        odds_data = fetch_todays_games_odds_api()
        time.sleep(0.5)

    # Step 4: Fetch player prop odds
    print("\n[STEP 4] Fetching player prop odds...")
    props_data = {}
    if include_props and not dry_run:
        props_data = fetch_player_props_odds_api()
        time.sleep(0.5)

    # If we got no games from either source, try to get them from cached odds
    if not games_nba and not odds_data:
        print("\n[WARNING] No games found from any source. Using simulated data.")
        odds_data = fetch_live_odds()  # Will generate simulated if no key

    # Step 5: Build predictions
    print("\n[STEP 5] Generating predictions...")
    output = build_predictions_output(
        games_nba=games_nba,
        odds_data=odds_data,
        props_data=props_data,
        evolution_model=evolution_model,
        bankroll=bankroll,
        include_props=include_props,
    )

    # Step 6: Save outputs
    print("\n[STEP 6] Saving outputs...")
    save_outputs(output)

    # Step 7: Print summary
    elapsed = time.time() - start_time
    print(f"\n[DONE] Pipeline completed in {elapsed:.1f}s")
    print_summary_table(output)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NBA Daily Prediction Pipeline — Generate real bet recommendations"
    )
    parser.add_argument(
        "--bankroll", type=float, default=100.0,
        help="Current bankroll in USD (default: $100)"
    )
    parser.add_argument(
        "--no-props", action="store_true",
        help="Skip player prop predictions (faster)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="No live API calls — use cached/simulated data"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON to stdout"
    )

    args = parser.parse_args()

    output = main(
        bankroll=args.bankroll,
        include_props=not args.no_props,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(output, indent=2, default=str))
