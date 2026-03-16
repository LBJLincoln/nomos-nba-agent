#!/usr/bin/env python3
"""
Karpathy-Style Continuous Training Loop for NBA Quant Agent.

Philosophy: simple code + massive data + continuous training.
Runs nightly. Each cycle: COLLECT -> FEATURES -> TRAIN -> ENSEMBLE -> PREDICT -> EVALUATE -> REPEAT.

Modes:
  --daemon     Run nightly at 3 AM UTC (after all games finish)
  --train      One-shot: build features from all data, train all models
  --predict    Predict today's games using current best ensemble
  --eval       Evaluate yesterday's predictions against actual results
  --full       Run complete cycle once (train + predict + eval)
  --status     Show current model performance and training history

Markets: moneyline, spread (margin), totals (total points).
Bootstrap: works with zero historical data by synthesizing from power ratings.
"""

import os, sys, json, math, time, copy, hashlib, signal
import ssl, urllib.request
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

# ── Load env ──────────────────────────────────────────────────────────────────
def load_env():
    for env_file in [ROOT / ".env.local", Path("/home/termius/mon-ipad/.env.local")]:
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

load_env()

# ── Config ────────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# Directories
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
TRAINING_DIR = DATA_DIR / "training"
FEATURES_DIR = DATA_DIR / "features"
PREDICTIONS_DIR = DATA_DIR / "predictions"
RESULTS_DIR = DATA_DIR / "results"
HISTORICAL_DIR = DATA_DIR / "historical"

for d in [MODELS_DIR, TRAINING_DIR, FEATURES_DIR, PREDICTIONS_DIR, RESULTS_DIR, HISTORICAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = TRAINING_DIR / "karpathy-loop.log"
HISTORY_FILE = TRAINING_DIR / "training-history.jsonl"
ELO_STATE_FILE = MODELS_DIR / "elo-ratings.json"
ENSEMBLE_WEIGHTS_FILE = MODELS_DIR / "ensemble-weights.json"
BEST_WEIGHTS_FILE = MODELS_DIR / "best-weights.json"

# ── Import existing models ────────────────────────────────────────────────────
from power_ratings import NBA_TEAMS, get_team, predict_matchup, HOME_COURT_ADVANTAGE

# ── Graceful imports for ML libraries ─────────────────────────────────────────
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
    from sklearn.calibration import calibration_curve
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    import pickle

# ── Shutdown handling ─────────────────────────────────────────────────────────
_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    _shutdown = True
    log("Shutdown signal received, finishing current step...")

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ==============================================================================
# LOGGING
# ==============================================================================

def log(msg: str, level: str = "INFO"):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level:5s}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def log_training_event(data: dict):
    """Append structured event to training history JSONL."""
    data["ts"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")
    except Exception as e:
        log(f"Failed to write training history: {e}", "WARN")


# ==============================================================================
# HTTP
# ==============================================================================

def http_get(url: str, headers: dict = None, timeout: int = 30) -> Tuple[Any, int]:
    hdrs = headers or {}
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status
    except Exception as e:
        return {"error": str(e)}, 0


# ==============================================================================
# ELO RATING SYSTEM
# ==============================================================================

class EloSystem:
    """
    Rolling Elo rating system for NBA teams.

    Calibration:
    - Start at 1500
    - K-factor adjustable (default 20)
    - Home advantage: +100 Elo points (~ +3 real points)
    - Margin-of-victory multiplier (FiveThirtyEight-style)
    - Season reset: regress 1/3 toward mean at season start
    """

    def __init__(self, k_factor: float = 20.0, home_advantage: float = 100.0,
                 initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
        self.history: List[dict] = []  # (game_date, home, away, home_won, margin)
        self._bootstrap()

    def _bootstrap(self):
        """Initialize Elo from power ratings for cold start."""
        for abbrev, team in NBA_TEAMS.items():
            # Map power rating [-12, +12] to Elo [1260, 1740]
            self.ratings[abbrev] = self.initial_rating + (team["base_power"] * 20)

    def expected(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def win_probability(self, home_team: str, away_team: str) -> float:
        """P(home wins) including home advantage."""
        h_abbrev, _ = get_team(home_team)
        a_abbrev, _ = get_team(away_team)
        if not h_abbrev or not a_abbrev:
            return 0.5
        h_elo = self.ratings.get(h_abbrev, self.initial_rating) + self.home_advantage
        a_elo = self.ratings.get(a_abbrev, self.initial_rating)
        return self.expected(h_elo, a_elo)

    def update(self, home_team: str, away_team: str, home_won: bool,
               margin: int = 0, game_date: str = ""):
        """Update Elo after a completed game."""
        h_abbrev, _ = get_team(home_team)
        a_abbrev, _ = get_team(away_team)
        if not h_abbrev or not a_abbrev:
            return

        h_elo = self.ratings[h_abbrev]
        a_elo = self.ratings[a_abbrev]

        expected_h = self.expected(h_elo + self.home_advantage, a_elo)
        actual_h = 1.0 if home_won else 0.0

        # Margin of victory multiplier (FiveThirtyEight formula)
        elo_diff = abs(h_elo - a_elo)
        mov_mult = math.log(max(abs(margin), 1) + 1) * (2.2 / (1.0 + 0.001 * elo_diff))
        k = self.k_factor * max(mov_mult, 0.5)

        self.ratings[h_abbrev] += k * (actual_h - expected_h)
        self.ratings[a_abbrev] += k * (expected_h - actual_h)

        self.history.append({
            "date": game_date,
            "home": h_abbrev, "away": a_abbrev,
            "home_won": home_won, "margin": margin,
            "h_elo_after": round(self.ratings[h_abbrev], 1),
            "a_elo_after": round(self.ratings[a_abbrev], 1),
        })

    def get_rating(self, team: str) -> float:
        abbrev, _ = get_team(team)
        return self.ratings.get(abbrev, self.initial_rating) if abbrev else self.initial_rating

    def get_diff(self, home_team: str, away_team: str) -> float:
        """Elo difference (home - away), including home advantage."""
        h_abbrev, _ = get_team(home_team)
        a_abbrev, _ = get_team(away_team)
        if not h_abbrev or not a_abbrev:
            return 0.0
        return (self.ratings.get(h_abbrev, self.initial_rating) + self.home_advantage
                - self.ratings.get(a_abbrev, self.initial_rating))

    def season_reset(self):
        """Regress all ratings 1/3 toward 1500 at season start."""
        for team in self.ratings:
            self.ratings[team] = self.ratings[team] * (2/3) + self.initial_rating * (1/3)
        log("Elo ratings reset for new season (1/3 regression to mean)")

    def save(self):
        state = {
            "ratings": {k: round(v, 1) for k, v in self.ratings.items()},
            "k_factor": self.k_factor,
            "home_advantage": self.home_advantage,
            "games_processed": len(self.history),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        ELO_STATE_FILE.write_text(json.dumps(state, indent=2))

    def load(self):
        if ELO_STATE_FILE.exists():
            try:
                state = json.loads(ELO_STATE_FILE.read_text())
                self.ratings = {k: float(v) for k, v in state.get("ratings", {}).items()}
                self.k_factor = state.get("k_factor", self.k_factor)
                log(f"Loaded Elo state: {len(self.ratings)} teams, "
                    f"{state.get('games_processed', 0)} games processed")
                return True
            except Exception as e:
                log(f"Failed to load Elo state: {e}", "WARN")
        return False


# Global Elo instance
elo_system = EloSystem()


# ==============================================================================
# STEP 1: COLLECT — Load all available data
# ==============================================================================

def collect_results() -> List[dict]:
    """
    Load all game results from:
    1. data/results/scores-*.json files (from Odds API)
    2. data/historical/*.jsonl files
    3. Fresh fetch from Odds API (completed games)

    Returns list of game dicts with: home_team, away_team, home_score, away_score, date
    """
    games = []
    seen = set()

    def _add_game(g: dict):
        key = f"{g.get('home_team','')}-{g.get('away_team','')}-{g.get('date','')}"
        if key not in seen and g.get("home_score") is not None:
            seen.add(key)
            games.append(g)

    # 1. Load from results dir (scores-*.json from Odds API)
    for f in sorted(RESULTS_DIR.glob("scores-*.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                for game in data:
                    if not game.get("completed"):
                        continue
                    scores = {s["name"]: int(s["score"]) for s in game.get("scores", []) if s.get("score")}
                    home = game.get("home_team", "")
                    away = game.get("away_team", "")
                    if home in scores and away in scores:
                        _add_game({
                            "home_team": home, "away_team": away,
                            "home_score": scores[home], "away_score": scores[away],
                            "date": game.get("commence_time", "")[:10],
                            "commence_time": game.get("commence_time", ""),
                            "source": "results_dir",
                        })
        except Exception as e:
            log(f"Error loading {f.name}: {e}", "WARN")

    # 2. Load from historical dir (JSONL format)
    for f in sorted(HISTORICAL_DIR.glob("*.jsonl")):
        try:
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                game = json.loads(line)
                _add_game(game)
        except Exception as e:
            log(f"Error loading {f.name}: {e}", "WARN")

    # 3. Load from historical JSON files (including nba_api games-*.json)
    for f in sorted(HISTORICAL_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            game_list = []
            if isinstance(data, list):
                game_list = data
            elif isinstance(data, dict) and "games" in data:
                game_list = data["games"]
            for game in game_list:
                # Convert nba_api format (home.pts / away.pts) to flat format
                if "home" in game and isinstance(game["home"], dict) and "pts" in game["home"]:
                    h = game["home"]
                    a = game.get("away", {})
                    if h.get("pts") is not None and a.get("pts") is not None:
                        _add_game({
                            "home_team": h.get("team_name", game.get("home_team", "")),
                            "away_team": a.get("team_name", game.get("away_team", "")),
                            "home_score": int(h["pts"]),
                            "away_score": int(a["pts"]),
                            "date": game.get("game_date", ""),
                            "commence_time": game.get("game_date", ""),
                            "source": f"nba_api:{f.name}",
                        })
                else:
                    _add_game(game)
        except Exception as e:
            log(f"Error loading {f.name}: {e}", "WARN")

    # 4. Fetch recent completed games from Odds API
    if ODDS_API_KEY:
        try:
            url = (f"https://api.the-odds-api.com/v4/sports/basketball_nba/scores/"
                   f"?apiKey={ODDS_API_KEY}&daysFrom=3&dateFormat=iso")
            data, status = http_get(url, timeout=20)
            if status == 200 and isinstance(data, list):
                for game in data:
                    if not game.get("completed"):
                        continue
                    scores = {s["name"]: int(s["score"]) for s in game.get("scores", []) if s.get("score")}
                    home = game.get("home_team", "")
                    away = game.get("away_team", "")
                    if home in scores and away in scores:
                        _add_game({
                            "home_team": home, "away_team": away,
                            "home_score": scores[home], "away_score": scores[away],
                            "date": game.get("commence_time", "")[:10],
                            "commence_time": game.get("commence_time", ""),
                            "source": "odds_api_live",
                        })
                # Save fetched scores for future use
                ts = datetime.now(timezone.utc).strftime("%Y%m%d")
                out = RESULTS_DIR / f"scores-{ts}.json"
                if not out.exists():
                    out.write_text(json.dumps(data, indent=2))
                log(f"Fetched {len([g for g in data if g.get('completed')])} completed games from Odds API")
        except Exception as e:
            log(f"Odds API fetch failed: {e}", "WARN")

    # Sort by date
    games.sort(key=lambda g: g.get("commence_time", g.get("date", "")))

    log(f"COLLECT: {len(games)} total games loaded from all sources")
    return games


def collect_predictions() -> List[dict]:
    """Load all predictions from predictions.jsonl."""
    preds = []
    pred_file = PREDICTIONS_DIR / "predictions.jsonl"
    if pred_file.exists():
        for line in pred_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                preds.append(json.loads(line))
            except Exception:
                pass
    log(f"COLLECT: {len(preds)} predictions loaded")
    return preds


# ==============================================================================
# DATA ENRICHMENT — Load team stats, player stats, injuries, odds history
# ==============================================================================

def load_team_stats() -> Dict[str, dict]:
    """
    Load team advanced stats from data/historical/team-stats-current.json.
    Returns dict keyed by team abbreviation (e.g. 'BOS') with ORtg, DRtg, Pace, etc.
    """
    path = HISTORICAL_DIR / "team-stats-current.json"
    if not path.exists():
        log("No team-stats-current.json found — run ingest-nba-data.py first", "WARN")
        return {}

    try:
        data = json.loads(path.read_text())
        teams = data.get("teams", [])
        result = {}
        for t in teams:
            # Map team name to abbreviation
            name = t.get("team_name", t.get("team", ""))
            abbrev = resolve_team(name)
            if not abbrev:
                # Try city + team
                city = t.get("city", "")
                if city:
                    abbrev = resolve_team(f"{city} {name}")
            if not abbrev:
                continue
            result[abbrev] = {
                "off_rating": t.get("off_rating", 0) or 0,
                "def_rating": t.get("def_rating", 0) or 0,
                "net_rating": t.get("net_rating", 0) or 0,
                "pace": t.get("pace", 0) or 0,
                "efg_pct": t.get("efg_pct", 0) or 0,
                "tov_pct": t.get("tov_pct", 0) or 0,
                "ts_pct": t.get("ts_pct", 0) or 0,
                "orb_pct": t.get("orb_pct", 0) or 0,
                "ftr": t.get("ftr", 0) or 0,
                "pie": t.get("pie", 0) or 0,
                "wins": t.get("wins", 0) or 0,
                "losses": t.get("losses", 0) or 0,
                "win_pct": t.get("win_pct", 0) or 0,
                "pts": t.get("pts", 0) or 0,
                "fg_pct": t.get("fg_pct", 0) or 0,
                "fg3_pct": t.get("fg3_pct", 0) or 0,
                "plus_minus": t.get("plus_minus", 0) or 0,
            }
        log(f"ENRICH: Loaded team stats for {len(result)} teams")
        return result
    except Exception as e:
        log(f"Failed to load team stats: {e}", "WARN")
        return {}


def load_player_stats() -> Dict[str, dict]:
    """
    Load player stats and aggregate per team (top-5 player impact score).
    Returns dict keyed by team abbreviation with aggregated player metrics.
    """
    # Find the most recent player stats file
    candidates = sorted(HISTORICAL_DIR.glob("player-stats-*.json"), reverse=True)
    if not candidates:
        log("No player-stats files found — run ingest-nba-data.py first", "WARN")
        return {}

    path = candidates[0]
    try:
        data = json.loads(path.read_text())
        players = data.get("players", [])

        # Group players by team
        team_players = defaultdict(list)
        for p in players:
            abbrev = p.get("team_abbr", "")
            if not abbrev:
                continue
            # Normalize abbreviation
            resolved = resolve_team(abbrev)
            if resolved:
                abbrev = resolved
            team_players[abbrev].append(p)

        result = {}
        for team, roster in team_players.items():
            # Sort by minutes (highest minutes = most important)
            roster.sort(key=lambda p: p.get("min_pg", 0) or 0, reverse=True)
            top5 = roster[:5]

            # Aggregate top-5 player metrics
            top5_pts = sum(p.get("pts", 0) or 0 for p in top5)
            top5_pie = sum(p.get("per", 0) or 0 for p in top5)  # PIE from nba_api
            top5_usg = sum(p.get("usg_pct", 0) or 0 for p in top5) / max(len(top5), 1)
            top5_net_rtg = sum(p.get("net_rating", 0) or 0 for p in top5) / max(len(top5), 1)
            star_player_pts = top5[0].get("pts", 0) or 0 if top5 else 0
            roster_depth = len(roster)

            result[team] = {
                "top5_pts": round(top5_pts, 1),
                "top5_pie": round(top5_pie, 4),
                "top5_avg_usg": round(top5_usg, 4),
                "top5_avg_net_rtg": round(top5_net_rtg, 1),
                "star_player_pts": round(star_player_pts, 1),
                "roster_depth": roster_depth,
                "roster": roster,  # Keep full roster for player props
            }

        log(f"ENRICH: Loaded player stats for {len(result)} teams ({len(players)} players)")
        return result
    except Exception as e:
        log(f"Failed to load player stats: {e}", "WARN")
        return {}


def load_injuries() -> Dict[str, List[dict]]:
    """
    Load current injuries grouped by team abbreviation.
    Returns dict keyed by team abbreviation with list of injured players.
    """
    path = HISTORICAL_DIR / "injuries-current.json"
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
        injuries = data.get("injuries", [])
        result = defaultdict(list)
        for inj in injuries:
            abbrev = inj.get("team_abbr", "")
            resolved = resolve_team(abbrev)
            if resolved:
                result[resolved].append(inj)
        log(f"ENRICH: Loaded {len(injuries)} injury entries across {len(result)} teams")
        return dict(result)
    except Exception as e:
        log(f"Failed to load injuries: {e}", "WARN")
        return {}


def load_odds_history() -> Dict[str, List[dict]]:
    """
    Load odds history and index by game key (home_team-away_team or game_id).
    Returns dict with line movement data per game.
    """
    path = HISTORICAL_DIR / "odds-history.jsonl"
    if not path.exists():
        # Fall back to scanning odds snapshot files directly
        return _load_odds_from_snapshots()

    try:
        game_odds = defaultdict(list)
        count = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                game_id = record.get("game_id", "")
                home = record.get("home_team", "")
                away = record.get("away_team", "")
                h_abbrev = resolve_team(home)
                a_abbrev = resolve_team(away)
                key = f"{h_abbrev}-{a_abbrev}" if h_abbrev and a_abbrev else game_id
                game_odds[key].append(record)
                count += 1
            except Exception:
                pass
        log(f"ENRICH: Loaded {count} odds records for {len(game_odds)} games")
        return dict(game_odds)
    except Exception as e:
        log(f"Failed to load odds history: {e}", "WARN")
        return {}


def _load_odds_from_snapshots() -> Dict[str, List[dict]]:
    """Load odds directly from snapshot files in data/."""
    snapshots = sorted(DATA_DIR.glob("odds-*.json"))[-50:]
    if not snapshots:
        return {}

    game_odds = defaultdict(list)
    for snap in snapshots:
        try:
            raw = json.loads(snap.read_text())
            games = raw if isinstance(raw, list) else []
            ts = snap.stem.replace("odds-", "")
            for game in games:
                home = game.get("home_team", "")
                away = game.get("away_team", "")
                h_abbrev = resolve_team(home)
                a_abbrev = resolve_team(away)
                if not h_abbrev or not a_abbrev:
                    continue
                key = f"{h_abbrev}-{a_abbrev}"
                # Extract consensus odds
                ml_probs = []
                spreads = []
                totals = []
                for bk in game.get("bookmakers", []):
                    for mkt in bk.get("markets", []):
                        mk = mkt.get("key", "")
                        for o in mkt.get("outcomes", []):
                            if mk == "h2h" and o.get("name") == home:
                                ml_probs.append(1.0 / max(o.get("price", 2.0), 1.01))
                            elif mk == "spreads" and o.get("name") == home:
                                spreads.append(o.get("point", 0))
                            elif mk == "totals" and o.get("name") == "Over":
                                totals.append(o.get("point", 0))
                game_odds[key].append({
                    "snapshot_ts": ts,
                    "home_ml_prob": sum(ml_probs) / len(ml_probs) if ml_probs else 0,
                    "market_spread": sum(spreads) / len(spreads) if spreads else 0,
                    "market_total": sum(totals) / len(totals) if totals else 0,
                })
        except Exception:
            pass

    log(f"ENRICH: Loaded odds from {len(snapshots)} snapshots for {len(game_odds)} games")
    return dict(game_odds)


def get_market_features(game_key: str, odds_history: Dict[str, List[dict]]) -> dict:
    """
    Extract market features for a specific game from odds history.
    Returns: opening line, closing line, line movement, market spread, market total.
    """
    records = odds_history.get(game_key, [])
    if not records:
        return {
            "market_home_prob": 0.0,
            "market_spread": 0.0,
            "market_total": 220.0,
            "line_movement": 0.0,
        }

    # Sort by timestamp
    records.sort(key=lambda r: r.get("snapshot_ts", r.get("last_update", "")))

    # Opening and closing lines
    opening = records[0]
    closing = records[-1]

    open_prob = opening.get("home_ml_prob", 0)
    close_prob = closing.get("home_ml_prob", 0)

    # For records from odds-history.jsonl (different format)
    if "outcomes" in opening:
        open_prob = _extract_prob_from_record(opening)
        close_prob = _extract_prob_from_record(closing)

    market_spread = closing.get("market_spread", 0)
    market_total = closing.get("market_total", 220)
    line_move = close_prob - open_prob if open_prob > 0 else 0

    return {
        "market_home_prob": close_prob,
        "market_spread": market_spread,
        "market_total": market_total,
        "line_movement": line_move,
    }


def _extract_prob_from_record(record: dict) -> float:
    """Extract home ML probability from an odds-history.jsonl record."""
    outcomes = record.get("outcomes", {})
    if not outcomes:
        return 0.0
    # outcomes is a dict with team names as keys
    for name, data in outcomes.items():
        price = data.get("price", 0)
        if price > 1.0:
            return 1.0 / price
    return 0.0


# ==============================================================================
# STEP 2: FEATURES — Build feature matrix
# ==============================================================================

def resolve_team(name: str) -> Optional[str]:
    """Resolve any team name/abbreviation to standard abbreviation."""
    abbrev, _ = get_team(name)
    if abbrev:
        return abbrev
    # Try matching by last word (e.g., "Thunder", "Celtics")
    name_lower = name.strip().lower()
    for ab, team in NBA_TEAMS.items():
        if name_lower in team["name"].lower() or team["name"].lower().endswith(name_lower.split()[-1]):
            return ab
    return None


def build_features(games: List[dict], elo: EloSystem,
                    team_stats: Dict[str, dict] = None,
                    player_stats: Dict[str, dict] = None,
                    injuries: Dict[str, List[dict]] = None,
                    odds_history: Dict[str, List[dict]] = None) -> Tuple:
    """
    Build feature matrix X and target vectors from game results.

    33 features per game:
    [0-16]  Original 17 features (Elo, win%, rest, H2H, power ratings, etc.)
    [17-22] Team advanced stats: home/away ORtg, DRtg, Pace (6)
    [23-26] Team efficiency: home/away eFG%, TOV% (4)
    [27-28] Player impact: top-5 PIE per team (2)
    [29-30] Injury count per team (2)
    [31]    Market consensus home probability
    [32]    Line movement (closing - opening probability)

    Targets:
    y_win:    1 if home team won, 0 otherwise
    y_margin: actual margin (home_score - away_score)
    y_total:  actual total points (home_score + away_score)
    """
    if not HAS_NUMPY:
        log("NumPy not available -- cannot build features", "ERROR")
        return None, None, None, None, None

    team_stats = team_stats or {}
    player_stats = player_stats or {}
    injuries = injuries or {}
    odds_history = odds_history or {}

    # Build rolling stats from game history
    team_results = defaultdict(list)     # team -> [(date, won, margin, opponent)]
    h2h_results = defaultdict(list)      # (team_a, team_b) -> [(date, a_won)]
    team_last_game = {}                  # team -> date string

    X_rows = []
    y_win = []
    y_margin = []
    y_total = []
    game_meta = []  # For debugging: store game info alongside features

    for game in games:
        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        home_abbrev = resolve_team(home_raw)
        away_abbrev = resolve_team(away_raw)

        if not home_abbrev or not away_abbrev:
            continue

        home_score = game.get("home_score")
        away_score = game.get("away_score")
        if home_score is None or away_score is None:
            continue

        home_score = int(home_score)
        away_score = int(away_score)
        game_date = game.get("date", game.get("commence_time", "")[:10])

        # Parse date
        try:
            if len(game_date) >= 10:
                gd = datetime.strptime(game_date[:10], "%Y-%m-%d")
            else:
                gd = datetime.now()
        except Exception:
            gd = datetime.now()

        # ── Compute features BEFORE updating rolling stats ──
        # This ensures no data leakage: features use only past data.

        # Home/Away Elo
        home_elo = elo.get_rating(home_abbrev)
        away_elo = elo.get_rating(away_abbrev)
        elo_diff = elo.get_diff(home_abbrev, away_abbrev)

        # Win % last 10 games
        home_recent = team_results[home_abbrev][-10:]
        away_recent = team_results[away_abbrev][-10:]
        home_win_pct_10 = (sum(1 for r in home_recent if r[1]) / len(home_recent)) if home_recent else 0.5
        away_win_pct_10 = (sum(1 for r in away_recent if r[1]) / len(away_recent)) if away_recent else 0.5

        # Rest days
        home_rest = _days_between(team_last_game.get(home_abbrev), game_date)
        away_rest = _days_between(team_last_game.get(away_abbrev), game_date)

        # H2H last 5
        h2h_key = tuple(sorted([home_abbrev, away_abbrev]))
        h2h_recent = h2h_results[h2h_key][-5:]
        h2h_home_wins = sum(1 for r in h2h_recent if
                            (r[1] and r[2] == home_abbrev) or (not r[1] and r[2] != home_abbrev)
                            ) if h2h_recent else 2.5

        # Home court factor (constant)
        home_court_factor = 1.0

        # Point differential last 10
        home_pt_diff_10 = (sum(r[2] for r in home_recent) / len(home_recent)) if home_recent else 0.0
        away_pt_diff_10 = (sum(r[2] for r in away_recent) / len(away_recent)) if away_recent else 0.0

        # Back-to-back flags
        home_b2b = 1.0 if home_rest == 1 else 0.0
        away_b2b = 1.0 if away_rest == 1 else 0.0

        # Conference game
        home_conf = NBA_TEAMS.get(home_abbrev, {}).get("conference", "")
        away_conf = NBA_TEAMS.get(away_abbrev, {}).get("conference", "")
        conf_game = 1.0 if (home_conf and home_conf == away_conf) else 0.0

        # Season phase (October = 0.0, April = 1.0)
        season_phase = _season_phase(gd)

        # Base power ratings
        home_power = NBA_TEAMS.get(home_abbrev, {}).get("base_power", 0.0)
        away_power = NBA_TEAMS.get(away_abbrev, {}).get("base_power", 0.0)

        # ── NEW: Team advanced stats (ORtg, DRtg, Pace, eFG%, TOV%, TS%) ──
        h_ts = team_stats.get(home_abbrev, {})
        a_ts = team_stats.get(away_abbrev, {})
        home_ortg = h_ts.get("off_rating", 110.0) or 110.0
        home_drtg = h_ts.get("def_rating", 110.0) or 110.0
        home_pace = h_ts.get("pace", 100.0) or 100.0
        home_efg = h_ts.get("efg_pct", 0.50) or 0.50
        home_tov_pct = h_ts.get("tov_pct", 0.13) or 0.13
        home_ts = h_ts.get("ts_pct", 0.56) or 0.56

        away_ortg = a_ts.get("off_rating", 110.0) or 110.0
        away_drtg = a_ts.get("def_rating", 110.0) or 110.0
        away_pace = a_ts.get("pace", 100.0) or 100.0
        away_efg = a_ts.get("efg_pct", 0.50) or 0.50
        away_tov_pct = a_ts.get("tov_pct", 0.13) or 0.13
        away_ts = a_ts.get("ts_pct", 0.56) or 0.56

        # ── NEW: Player impact (top-5 PIE) ──
        h_ps = player_stats.get(home_abbrev, {})
        a_ps = player_stats.get(away_abbrev, {})
        home_top5_pie = h_ps.get("top5_pie", 0.50) or 0.50
        away_top5_pie = a_ps.get("top5_pie", 0.50) or 0.50

        # ── NEW: Injury count ──
        home_inj_count = len(injuries.get(home_abbrev, []))
        away_inj_count = len(injuries.get(away_abbrev, []))

        # ── NEW: Market features (line movement, consensus probability) ──
        game_key = f"{home_abbrev}-{away_abbrev}"
        mkt = get_market_features(game_key, odds_history)
        market_home_prob = mkt["market_home_prob"]
        line_movement = mkt["line_movement"]

        # ── Build feature row (35 features) ──
        row = [
            # Original 17 features
            home_elo, away_elo, elo_diff,
            home_win_pct_10, away_win_pct_10,
            min(home_rest, 7), min(away_rest, 7),
            h2h_home_wins / 5.0,
            home_court_factor,
            home_pt_diff_10, away_pt_diff_10,
            home_b2b, away_b2b,
            conf_game,
            season_phase,
            home_power, away_power,
            # Team advanced stats (6 features)
            home_ortg - 110.0, home_drtg - 110.0, home_pace - 100.0,
            away_ortg - 110.0, away_drtg - 110.0, away_pace - 100.0,
            # Team efficiency (4 features)
            home_efg, home_tov_pct,
            away_efg, away_tov_pct,
            # Player impact (2 features)
            home_top5_pie, away_top5_pie,
            # Injury impact (2 features)
            min(home_inj_count, 5), min(away_inj_count, 5),
            # Market features (2 features)
            market_home_prob,
            line_movement,
        ]

        X_rows.append(row)

        # Targets
        home_won = home_score > away_score
        margin = home_score - away_score
        total = home_score + away_score

        y_win.append(1 if home_won else 0)
        y_margin.append(margin)
        y_total.append(total)

        game_meta.append({
            "home": home_abbrev, "away": away_abbrev,
            "date": game_date, "score": f"{home_score}-{away_score}",
        })

        # ── NOW update rolling stats with this game's result ──
        team_results[home_abbrev].append((game_date, home_won, margin, away_abbrev))
        team_results[away_abbrev].append((game_date, not home_won, -margin, home_abbrev))

        h2h_results[h2h_key].append((game_date, home_won, home_abbrev))

        team_last_game[home_abbrev] = game_date
        team_last_game[away_abbrev] = game_date

        # Update Elo
        elo.update(home_abbrev, away_abbrev, home_won, margin, game_date)

    if not X_rows:
        log("No features built -- no valid games found", "WARN")
        return None, None, None, None, None

    X = np.array(X_rows, dtype=np.float64)
    y_w = np.array(y_win, dtype=np.int32)
    y_m = np.array(y_margin, dtype=np.float64)
    y_t = np.array(y_total, dtype=np.float64)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Cache features
    cache_file = FEATURES_DIR / f"features-{datetime.now(timezone.utc).strftime('%Y%m%d')}.npz"
    try:
        np.savez_compressed(cache_file, X=X, y_win=y_w, y_margin=y_m, y_total=y_t)
    except Exception:
        pass

    log(f"FEATURES: Built {X.shape[0]} samples x {X.shape[1]} features "
        f"(team_stats: {len(team_stats)} teams, players: {len(player_stats)} teams, "
        f"injuries: {len(injuries)} teams, odds: {len(odds_history)} games)")
    return X, y_w, y_m, y_t, game_meta


FEATURE_NAMES = [
    # Original 17
    "home_elo", "away_elo", "elo_diff",
    "home_win_pct_10", "away_win_pct_10",
    "home_rest_days", "away_rest_days",
    "h2h_home_win_rate", "home_court_factor",
    "home_pt_diff_10", "away_pt_diff_10",
    "home_b2b", "away_b2b",
    "conference_game", "season_phase",
    "home_power_rating", "away_power_rating",
    # Team advanced (6)
    "home_ortg_adj", "home_drtg_adj", "home_pace_adj",
    "away_ortg_adj", "away_drtg_adj", "away_pace_adj",
    # Team efficiency (4)
    "home_efg_pct", "home_tov_pct",
    "away_efg_pct", "away_tov_pct",
    # Player impact (2)
    "home_top5_pie", "away_top5_pie",
    # Injuries (2)
    "home_injury_count", "away_injury_count",
    # Market (2)
    "market_home_prob", "line_movement",
]


def _days_between(date_str_a: Optional[str], date_str_b: str) -> int:
    """Days between two date strings. Returns 3 (normal rest) if either is missing."""
    if not date_str_a or not date_str_b:
        return 3
    try:
        a = datetime.strptime(str(date_str_a)[:10], "%Y-%m-%d")
        b = datetime.strptime(str(date_str_b)[:10], "%Y-%m-%d")
        return max(0, (b - a).days)
    except Exception:
        return 3


def _season_phase(game_date: datetime) -> float:
    """Map game date to season phase: 0.0 (Oct) to 1.0 (Apr/May)."""
    month = game_date.month
    # NBA season: October (10) through April (4) + playoffs May/June
    if month >= 10:
        # Oct=0.0, Nov=0.14, Dec=0.28
        return (month - 10) / 7.0
    elif month <= 6:
        # Jan=0.43, Feb=0.57, Mar=0.71, Apr=0.86, May=0.93, Jun=1.0
        return (month + 2) / 7.0
    else:
        # Offseason
        return 1.0


# ==============================================================================
# BOOTSTRAP: Generate synthetic training data from power ratings
# ==============================================================================

def bootstrap_synthetic_games(n_games: int = 500) -> List[dict]:
    """
    Generate synthetic historical games using power ratings + noise.
    Used for cold start when no real historical data is available.

    This is the Karpathy philosophy: start training immediately,
    even with imperfect data. Real data will gradually replace synthetic.
    """
    if not HAS_NUMPY:
        return []

    log(f"BOOTSTRAP: Generating {n_games} synthetic games from power ratings")
    teams = list(NBA_TEAMS.keys())
    games = []
    rng = np.random.RandomState(42)  # Reproducible

    # Generate a full synthetic season
    base_date = datetime(2025, 10, 22)  # Season start

    for i in range(n_games):
        home = teams[rng.randint(len(teams))]
        away = teams[rng.randint(len(teams))]
        while away == home:
            away = teams[rng.randint(len(teams))]

        # Use power ratings to generate realistic scores
        pred = predict_matchup(home, away)
        if not pred:
            continue

        home_expected = pred["home_expected_pts"]
        away_expected = pred["away_expected_pts"]

        # Add realistic noise (NBA game stdev ~ 12 points per team)
        home_score = max(80, int(rng.normal(home_expected, 12)))
        away_score = max(80, int(rng.normal(away_expected, 12)))

        # Resolve ties with OT
        while home_score == away_score:
            home_score += max(0, int(rng.normal(5, 2.5)))
            away_score += max(0, int(rng.normal(5, 2.5)))

        game_date = base_date + timedelta(days=int(i * 0.35))  # ~0.35 days between games

        games.append({
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "date": game_date.strftime("%Y-%m-%d"),
            "commence_time": game_date.isoformat(),
            "source": "synthetic_bootstrap",
        })

    # Save for reference
    synth_file = HISTORICAL_DIR / "synthetic-bootstrap.jsonl"
    with open(synth_file, "w") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")

    log(f"BOOTSTRAP: {len(games)} synthetic games saved to {synth_file.name}")
    return games


# ==============================================================================
# STEP 3: TRAIN — Train multiple models
# ==============================================================================

def save_model(model, name: str):
    """Save model to disk."""
    model_path = MODELS_DIR / f"{name}.pkl"
    if HAS_JOBLIB:
        joblib.dump(model, model_path)
    else:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    return model_path


def load_model(name: str):
    """Load model from disk."""
    model_path = MODELS_DIR / f"{name}.pkl"
    if not model_path.exists():
        return None
    try:
        if HAS_JOBLIB:
            return joblib.load(model_path)
        else:
            with open(model_path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        log(f"Failed to load model {name}: {e}", "WARN")
        return None


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate a classification model on test data."""
    try:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_pred_prob = np.full(len(y_test), 0.5)

    y_pred_class = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_class)
    brier = brier_score_loss(y_test, y_pred_prob)

    try:
        ll = log_loss(y_test, y_pred_prob)
    except Exception:
        ll = 999.0

    # Calibration: compare predicted vs actual probabilities in bins
    try:
        fraction_pos, mean_predicted = calibration_curve(y_test, y_pred_prob, n_bins=5, strategy="uniform")
        calibration_error = float(np.mean(np.abs(fraction_pos - mean_predicted)))
    except Exception:
        calibration_error = 0.5

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "calibration_error": round(calibration_error, 4),
        "n_test": len(y_test),
    }

    log(f"  {model_name:<20s} | Acc: {accuracy:.3f} | Brier: {brier:.4f} | "
        f"LogLoss: {ll:.4f} | CalErr: {calibration_error:.4f}")

    return metrics


def train_all_models(X: np.ndarray, y_win: np.ndarray,
                     y_margin: np.ndarray, y_total: np.ndarray) -> dict:
    """
    Train all available models on the data.

    Split: 70% train, 15% validation, 15% test.

    Returns dict of {model_name: {"model": model_obj, "metrics": {...}, "val_brier": float}}
    """
    if not HAS_SKLEARN:
        log("scikit-learn not installed -- cannot train ML models", "ERROR")
        log("Install: pip install scikit-learn", "ERROR")
        return {}

    n = len(y_win)
    if n < 20:
        log(f"Only {n} samples -- need at least 20 for training. "
            f"Run bootstrap or ingest more data.", "WARN")
        return {}

    log(f"\nTRAIN: Starting model training on {n} games")
    log(f"{'='*70}")

    # Split: 70/15/15
    X_temp, X_test, y_w_temp, y_w_test = train_test_split(
        X, y_win, test_size=0.15, random_state=42, shuffle=True
    )
    X_train, X_val, y_w_train, y_w_val = train_test_split(
        X_temp, y_w_temp, test_size=0.176, random_state=42  # 0.176 of 85% ~ 15%
    )

    # Also split margin and total targets in same order
    idx_temp, idx_test = train_test_split(
        np.arange(n), test_size=0.15, random_state=42, shuffle=True
    )
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=0.176, random_state=42
    )

    y_m_train, y_m_val, y_m_test = y_margin[idx_train], y_margin[idx_val], y_margin[idx_test]
    y_t_train, y_t_val, y_t_test = y_total[idx_train], y_total[idx_val], y_total[idx_test]

    log(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    save_model(scaler, "scaler")

    results = {}

    # ── 1. Logistic Regression (baseline) ──
    log("\n  Training Logistic Regression (baseline)...")
    try:
        lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
        lr.fit(X_train_s, y_w_train)
        val_metrics = evaluate_model(lr, X_val_s, y_w_val, "logistic_regression")
        test_metrics = evaluate_model(lr, X_test_s, y_w_test, "logistic_regression")
        save_model(lr, "logistic_regression")
        results["logistic_regression"] = {
            "model": lr,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "val_brier": val_metrics["brier_score"],
            "uses_scaled": True,
        }
    except Exception as e:
        log(f"  Logistic Regression failed: {e}", "ERROR")

    # ── 2. Random Forest ──
    log("\n  Training Random Forest...")
    try:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_w_train)  # RF doesn't need scaling
        val_metrics = evaluate_model(rf, X_val, y_w_val, "random_forest")
        test_metrics = evaluate_model(rf, X_test, y_w_test, "random_forest")
        save_model(rf, "random_forest")
        results["random_forest"] = {
            "model": rf,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "val_brier": val_metrics["brier_score"],
            "uses_scaled": False,
        }

        # Feature importance
        importances = rf.feature_importances_
        top_features = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])[:5]
        log(f"  Top features: {', '.join(f'{n}={v:.3f}' for n, v in top_features)}")
    except Exception as e:
        log(f"  Random Forest failed: {e}", "ERROR")

    # ── 3. XGBoost (primary) ──
    if HAS_XGB:
        log("\n  Training XGBoost (primary)...")
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=3, gamma=0.1,
                reg_alpha=0.1, reg_lambda=1.0,
                eval_metric="logloss", random_state=42,
                use_label_encoder=False,
            )
            xgb_model.fit(
                X_train, y_w_train,
                eval_set=[(X_val, y_w_val)],
                verbose=False,
            )
            val_metrics = evaluate_model(xgb_model, X_val, y_w_val, "xgboost")
            test_metrics = evaluate_model(xgb_model, X_test, y_w_test, "xgboost")
            save_model(xgb_model, "xgboost")
            results["xgboost"] = {
                "model": xgb_model,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "val_brier": val_metrics["brier_score"],
                "uses_scaled": False,
            }
        except Exception as e:
            log(f"  XGBoost failed: {e}", "ERROR")
    else:
        log("  XGBoost not installed (pip install xgboost)", "WARN")

    # ── 4. LightGBM ──
    if HAS_LGB:
        log("\n  Training LightGBM...")
        try:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=5, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbose=-1,
            )
            lgb_model.fit(
                X_train, y_w_train,
                eval_set=[(X_val, y_w_val)],
                callbacks=[lgb.log_evaluation(0)],
            )
            val_metrics = evaluate_model(lgb_model, X_val, y_w_val, "lightgbm")
            test_metrics = evaluate_model(lgb_model, X_test, y_w_test, "lightgbm")
            save_model(lgb_model, "lightgbm")
            results["lightgbm"] = {
                "model": lgb_model,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "val_brier": val_metrics["brier_score"],
                "uses_scaled": False,
            }
        except Exception as e:
            log(f"  LightGBM failed: {e}", "ERROR")
    else:
        log("  LightGBM not installed (pip install lightgbm)", "WARN")

    # Summary
    log(f"\n{'='*70}")
    log(f"TRAIN SUMMARY: {len(results)} models trained")
    if results:
        best = min(results.items(), key=lambda x: x[1]["val_brier"])
        log(f"  Best model: {best[0]} (val Brier: {best[1]['val_brier']:.4f})")

    return results


# ==============================================================================
# STEP 4: ENSEMBLE — Weight models by validation performance
# ==============================================================================

def compute_ensemble_weights(model_results: dict) -> dict:
    """
    Compute ensemble weights inversely proportional to validation Brier score.
    Lower Brier = better calibration = higher weight.

    Also includes the built-in models (power_ratings, elo, poisson, monte_carlo)
    with default weights that can be overridden by ML model performance.
    """
    if not model_results:
        # Fallback to default weights from predictor.py
        default_weights = {
            "power_ratings": 0.35,
            "elo": 0.20,
            "poisson": 0.15,
            "monte_carlo": 0.30,
        }
        log("No trained models -- using default ensemble weights")
        return default_weights

    # Collect all Brier scores
    brier_scores = {}
    for name, result in model_results.items():
        brier_scores[name] = result["val_brier"]

    # Add built-in model estimates (these are heuristic priors)
    # Power ratings + Elo + Poisson + MC typically achieve ~0.23-0.26 Brier on NBA
    builtin_brier_estimates = {
        "power_ratings": 0.240,
        "elo_system": 0.245,
        "poisson": 0.250,
        "monte_carlo": 0.235,
    }

    all_briers = {**builtin_brier_estimates, **brier_scores}

    # Inverse Brier weighting: weight = 1/brier, then normalize
    inverse_briers = {}
    for name, brier in all_briers.items():
        # Clamp brier to avoid division by zero
        inverse_briers[name] = 1.0 / max(brier, 0.01)

    total_inv = sum(inverse_briers.values())
    weights = {name: round(inv / total_inv, 4) for name, inv in inverse_briers.items()}

    # Save
    ensemble_data = {
        "weights": weights,
        "brier_scores": {k: round(v, 4) for k, v in all_briers.items()},
        "n_models": len(weights),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    ENSEMBLE_WEIGHTS_FILE.write_text(json.dumps(ensemble_data, indent=2))

    log(f"\nENSEMBLE weights computed ({len(weights)} models):")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        brier = all_briers.get(name, 0)
        log(f"  {name:<20s} weight={w:.3f} (Brier={brier:.4f})")

    return weights


# ==============================================================================
# STEP 5: PREDICT — Run ensemble on today's games
# ==============================================================================

def predict_today(model_results: dict, weights: dict, elo: EloSystem,
                   team_stats: Dict[str, dict] = None,
                   player_stats: Dict[str, dict] = None,
                   injuries: Dict[str, List[dict]] = None) -> List[dict]:
    """
    Predict today's games using the trained ensemble.

    For each game:
    1. Build 35-feature vector (including team advanced stats, player impact, injuries)
    2. Get probability from each ML model
    3. Get probability from built-in models (power ratings, Elo, Poisson, MC)
    4. Weighted average = ensemble prediction
    5. Predict spread (from margin model or power ratings)
    6. Predict total (from total model or Poisson)
    7. Player props predictions for key players
    """
    team_stats = team_stats or {}
    player_stats = player_stats or {}
    injuries = injuries or {}

    # Fetch today's games from Odds API
    if ODDS_API_KEY:
        url = (f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
               f"?apiKey={ODDS_API_KEY}&regions=us,eu,uk&markets=h2h,spreads,totals"
               f"&oddsFormat=decimal&dateFormat=iso")
        data, status = http_get(url, timeout=20)
        if status == 200 and isinstance(data, list):
            today_games = data
        else:
            log("Failed to fetch today's odds -- no predictions generated", "WARN")
            return []
    else:
        log("ODDS_API_KEY not set -- cannot fetch today's games for prediction", "WARN")
        return []

    if not today_games:
        log("No games scheduled today")
        return []

    log(f"\nPREDICT: {len(today_games)} games found for today")

    # Load scaler if available
    scaler = load_model("scaler")
    predictions = []

    for game in today_games:
        home_name = game.get("home_team", "")
        away_name = game.get("away_team", "")
        home_abbrev = resolve_team(home_name)
        away_abbrev = resolve_team(away_name)

        if not home_abbrev or not away_abbrev:
            log(f"  Skipping unknown matchup: {home_name} vs {away_name}", "WARN")
            continue

        # ── Build 35-feature vector for this game ──
        home_elo_val = elo.get_rating(home_abbrev)
        away_elo_val = elo.get_rating(away_abbrev)
        elo_diff_val = elo.get_diff(home_abbrev, away_abbrev)

        home_power = NBA_TEAMS.get(home_abbrev, {}).get("base_power", 0)
        away_power = NBA_TEAMS.get(away_abbrev, {}).get("base_power", 0)

        # Use team_stats for win % if available, else estimate from power
        h_ts = team_stats.get(home_abbrev, {})
        a_ts = team_stats.get(away_abbrev, {})
        home_win_pct_est = h_ts.get("win_pct", 0) or min(0.85, max(0.15, 0.5 + home_power * 0.02))
        away_win_pct_est = a_ts.get("win_pct", 0) or min(0.85, max(0.15, 0.5 + away_power * 0.02))

        home_conf = NBA_TEAMS.get(home_abbrev, {}).get("conference", "")
        away_conf = NBA_TEAMS.get(away_abbrev, {}).get("conference", "")

        # Team advanced stats
        home_ortg = (h_ts.get("off_rating", 0) or 110.0) - 110.0
        home_drtg = (h_ts.get("def_rating", 0) or 110.0) - 110.0
        home_pace = (h_ts.get("pace", 0) or 100.0) - 100.0
        away_ortg = (a_ts.get("off_rating", 0) or 110.0) - 110.0
        away_drtg = (a_ts.get("def_rating", 0) or 110.0) - 110.0
        away_pace = (a_ts.get("pace", 0) or 100.0) - 100.0

        home_efg = h_ts.get("efg_pct", 0) or 0.50
        home_tov = h_ts.get("tov_pct", 0) or 0.13
        away_efg = a_ts.get("efg_pct", 0) or 0.50
        away_tov = a_ts.get("tov_pct", 0) or 0.13

        # Player impact
        h_ps = player_stats.get(home_abbrev, {})
        a_ps = player_stats.get(away_abbrev, {})
        home_top5_pie = h_ps.get("top5_pie", 0.50) or 0.50
        away_top5_pie = a_ps.get("top5_pie", 0.50) or 0.50

        # Injuries
        home_inj_count = min(len(injuries.get(home_abbrev, [])), 5)
        away_inj_count = min(len(injuries.get(away_abbrev, [])), 5)

        # Market consensus from today's game data
        market_home_prob = _extract_market_prob(game, home_name)
        line_movement = 0.0  # No historical movement for live prediction

        feature_row = np.array([[
            # Original 17
            home_elo_val, away_elo_val, elo_diff_val,
            home_win_pct_est, away_win_pct_est,
            2.0, 2.0,  # Assume normal rest
            0.5,       # H2H neutral
            1.0,       # Home court
            home_power, away_power,  # Proxy for pt diff
            0.0, 0.0,  # No B2B info
            1.0 if home_conf == away_conf else 0.0,
            _season_phase(datetime.now()),
            home_power, away_power,
            # Team advanced (6)
            home_ortg, home_drtg, home_pace,
            away_ortg, away_drtg, away_pace,
            # Team efficiency (4)
            home_efg, home_tov,
            away_efg, away_tov,
            # Player impact (2)
            home_top5_pie, away_top5_pie,
            # Injuries (2)
            home_inj_count, away_inj_count,
            # Market (2)
            market_home_prob, line_movement,
        ]])

        # ── Collect probabilities from all models ──
        probs = {}

        # Built-in models
        pred_pr = predict_matchup(home_abbrev, away_abbrev)
        if pred_pr:
            probs["power_ratings"] = pred_pr["home_win_prob"]

        probs["elo_system"] = elo.win_probability(home_abbrev, away_abbrev)

        # Poisson (import from predictor if available)
        try:
            from predictor import poisson_predict
            poisson_result = poisson_predict(home_abbrev, away_abbrev)
            if poisson_result:
                probs["poisson"] = poisson_result["home_win_prob"]
        except Exception:
            pass

        # Monte Carlo
        try:
            from predictor import monte_carlo_predict
            mc_result = monte_carlo_predict(home_abbrev, away_abbrev, 500)
            if mc_result:
                probs["monte_carlo"] = mc_result["home_win_prob"]
        except Exception:
            pass

        # ML models
        for model_name, result in model_results.items():
            try:
                model = result["model"]
                x_input = feature_row.copy()
                if result.get("uses_scaled") and scaler is not None:
                    x_input = scaler.transform(x_input)
                prob = model.predict_proba(x_input)[0, 1]
                probs[model_name] = float(prob)
            except Exception as e:
                log(f"  {model_name} prediction failed: {e}", "WARN")

        # ── Weighted ensemble ──
        ensemble_prob = 0.0
        total_weight = 0.0
        for model_name, prob in probs.items():
            w = weights.get(model_name, 0.0)
            if w > 0:
                ensemble_prob += prob * w
                total_weight += w

        if total_weight > 0:
            ensemble_prob /= total_weight
        else:
            ensemble_prob = probs.get("power_ratings", 0.5)

        # Clamp
        ensemble_prob = max(0.02, min(0.98, ensemble_prob))

        # ── Market odds ──
        market_prob = _extract_market_prob(game, home_name)

        # ── Spread and total from power ratings ──
        predicted_spread = pred_pr["spread"] if pred_pr else 0.0
        predicted_total = pred_pr["predicted_total"] if pred_pr else 220.0

        # ── CLV (Closing Line Value) ──
        clv = ensemble_prob - market_prob if market_prob > 0 else 0.0

        # ── Player props predictions ──
        player_props = _predict_player_props(
            home_abbrev, away_abbrev,
            player_stats, team_stats, injuries
        )

        # ── Extract market spread and total from game data ──
        market_spread = _extract_market_spread(game, home_name)
        market_total_line = _extract_market_total(game)

        prediction = {
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "home_name": home_name,
            "away_name": away_name,
            "home_win_prob": round(ensemble_prob, 4),
            "model_probs": {k: round(v, 4) for k, v in probs.items()},
            "market_prob": round(market_prob, 4),
            "predicted_spread": round(predicted_spread, 1),
            "predicted_total": round(predicted_total, 1),
            "market_spread": market_spread,
            "market_total": market_total_line,
            "clv": round(clv, 4),
            "confidence": _confidence_label(ensemble_prob),
            "ensemble_weights": {k: v for k, v in weights.items() if k in probs},
            "team_advanced": {
                "home": {"ortg": h_ts.get("off_rating"), "drtg": h_ts.get("def_rating"),
                         "pace": h_ts.get("pace"), "efg": h_ts.get("efg_pct")},
                "away": {"ortg": a_ts.get("off_rating"), "drtg": a_ts.get("def_rating"),
                         "pace": a_ts.get("pace"), "efg": a_ts.get("efg_pct")},
            },
            "injuries": {
                "home": [i.get("player_name", "?") for i in injuries.get(home_abbrev, [])],
                "away": [i.get("player_name", "?") for i in injuries.get(away_abbrev, [])],
            },
            "player_props": player_props,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        predictions.append(prediction)

        conf_str = prediction["confidence"]
        n_props = len(player_props)
        log(f"  {away_name:>25s} @ {home_name:<25s} | "
            f"Home {ensemble_prob*100:5.1f}% | Spread {predicted_spread:+5.1f} | "
            f"Total {predicted_total:5.0f} | CLV {clv*100:+4.1f}% | {conf_str} | "
            f"Props: {n_props}")

    # Save predictions
    if predictions:
        # Append to JSONL
        with open(PREDICTIONS_DIR / "predictions.jsonl", "a") as f:
            for p in predictions:
                compact = {
                    "home_team": p["home_team"], "away_team": p["away_team"],
                    "home_win_prob": p["home_win_prob"],
                    "model_raw_prob": p["model_probs"].get("power_ratings", p["home_win_prob"]),
                    "market_prob": p["market_prob"],
                    "predicted_spread": p["predicted_spread"],
                    "predicted_total": p["predicted_total"],
                    "confidence": p["confidence"],
                    "timestamp": p["timestamp"],
                }
                f.write(json.dumps(compact) + "\n")

        # Save full prediction snapshot
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        snap_file = PREDICTIONS_DIR / f"karpathy-{ts}.json"
        snap_file.write_text(json.dumps(predictions, indent=2, default=str))

        log(f"\nPREDICT: {len(predictions)} predictions saved")

    return predictions


def _extract_market_prob(game: dict, home_name: str) -> float:
    """Extract consensus market probability for home team from bookmaker odds."""
    probs = []
    for bk in game.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == home_name and outcome.get("price", 0) > 1.0:
                    probs.append(1.0 / outcome["price"])
    return sum(probs) / len(probs) if probs else 0.0


def _confidence_label(prob: float) -> str:
    """Map probability to confidence label."""
    dist = abs(prob - 0.5)
    if dist >= 0.35:
        return "VERY HIGH"
    elif dist >= 0.25:
        return "HIGH"
    elif dist >= 0.15:
        return "MEDIUM"
    elif dist >= 0.05:
        return "LOW"
    else:
        return "COIN FLIP"


def _extract_market_spread(game: dict, home_name: str) -> float:
    """Extract consensus market spread for home team."""
    spreads = []
    for bk in game.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market.get("key") != "spreads":
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == home_name and outcome.get("point") is not None:
                    spreads.append(float(outcome["point"]))
    return round(sum(spreads) / len(spreads), 1) if spreads else 0.0


def _extract_market_total(game: dict) -> float:
    """Extract consensus market total."""
    totals = []
    for bk in game.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market.get("key") != "totals":
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == "Over" and outcome.get("point") is not None:
                    totals.append(float(outcome["point"]))
    return round(sum(totals) / len(totals), 1) if totals else 220.0


def _predict_player_props(home_abbrev: str, away_abbrev: str,
                           player_stats: Dict[str, dict],
                           team_stats: Dict[str, dict],
                           injuries: Dict[str, List[dict]]) -> List[dict]:
    """
    Predict player props (points, rebounds, assists, PRA) for key players.

    Method: baseline from season averages, adjusted for:
    - Opponent defensive rating (DRtg)
    - Pace matchup (faster pace = more stats)
    - Injury context (missing teammates = usage boost)
    """
    props = []
    injured_names = set()
    for team_inj in [injuries.get(home_abbrev, []), injuries.get(away_abbrev, [])]:
        for inj in team_inj:
            injured_names.add(inj.get("player_name", "").strip().lower())

    for team_abbrev, opp_abbrev, side in [
        (home_abbrev, away_abbrev, "home"),
        (away_abbrev, home_abbrev, "away"),
    ]:
        ps = player_stats.get(team_abbrev, {})
        roster = ps.get("roster", [])
        if not roster:
            continue

        opp_ts = team_stats.get(opp_abbrev, {})
        opp_drtg = opp_ts.get("def_rating", 110.0) or 110.0
        opp_pace = opp_ts.get("pace", 100.0) or 100.0
        league_avg_drtg = 110.0
        league_avg_pace = 100.0

        # Defensive adjustment factor: bad defense = more points
        def_factor = opp_drtg / league_avg_drtg if league_avg_drtg > 0 else 1.0
        # Pace factor: faster pace = more possessions = more stats
        pace_factor = opp_pace / league_avg_pace if league_avg_pace > 0 else 1.0
        combined_factor = (def_factor * 0.6 + pace_factor * 0.4)

        # Check if any teammate is injured (usage boost for remaining players)
        team_injured = [n for n in injured_names
                        if any(n in (p.get("player_name", "").lower()) for p in roster)]
        usage_boost = 1.0 + (len(team_injured) * 0.03)  # 3% per injured player

        # Top 5 players by minutes
        top_players = sorted(roster, key=lambda p: p.get("min_pg", 0) or 0, reverse=True)[:5]

        for player in top_players:
            pname = player.get("player_name", "")
            if pname.strip().lower() in injured_names:
                continue  # Skip injured players

            base_pts = player.get("pts", 0) or 0
            base_reb = player.get("reb", 0) or 0
            base_ast = player.get("ast", 0) or 0

            # Adjusted predictions
            adj_pts = round(base_pts * combined_factor * usage_boost, 1)
            adj_reb = round(base_reb * pace_factor * usage_boost, 1)
            adj_ast = round(base_ast * pace_factor * usage_boost, 1)
            adj_pra = round(adj_pts + adj_reb + adj_ast, 1)

            props.append({
                "player": pname,
                "team": team_abbrev,
                "side": side,
                "predicted_pts": adj_pts,
                "predicted_reb": adj_reb,
                "predicted_ast": adj_ast,
                "predicted_pra": adj_pra,
                "base_pts": base_pts,
                "base_reb": base_reb,
                "base_ast": base_ast,
                "def_factor": round(def_factor, 3),
                "pace_factor": round(pace_factor, 3),
                "usage_boost": round(usage_boost, 3),
                "min_pg": player.get("min_pg", 0),
            })

    return props


# ==============================================================================
# STEP 6: EVALUATE — Score yesterday's predictions
# ==============================================================================

def evaluate_recent(elo: EloSystem) -> Optional[dict]:
    """
    Compare recent predictions against actual results.

    Calculates:
    - Accuracy (did we pick the winner?)
    - Brier score (calibration quality)
    - Spread error (how far off our spread was)
    - Total error (how far off our total was)
    - CLV tracking (are we beating the closing line?)
    """
    # Load predictions from last 3 days
    predictions = []
    pred_file = PREDICTIONS_DIR / "predictions.jsonl"
    if not pred_file.exists():
        log("No predictions file found", "WARN")
        return None

    cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    for line in pred_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            pred = json.loads(line)
            ts = pred.get("timestamp", "")
            if ts >= cutoff:
                predictions.append(pred)
        except Exception:
            pass

    if not predictions:
        log("No recent predictions to evaluate")
        return None

    # Load actual results
    results = collect_results()
    result_lookup = {}
    for game in results:
        home = resolve_team(game.get("home_team", ""))
        away = resolve_team(game.get("away_team", ""))
        if home and away:
            key = f"{home}-{away}"
            result_lookup[key] = game

    # Match predictions to results
    matched = 0
    correct = 0
    brier_sum = 0.0
    spread_errors = []
    total_errors = []
    clv_values = []

    for pred in predictions:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        h_abbrev = resolve_team(home)
        a_abbrev = resolve_team(away)
        if not h_abbrev or not a_abbrev:
            continue

        key = f"{h_abbrev}-{a_abbrev}"
        actual = result_lookup.get(key)
        if not actual:
            continue

        matched += 1
        home_score = int(actual["home_score"])
        away_score = int(actual["away_score"])
        home_won = home_score > away_score
        actual_margin = home_score - away_score
        actual_total = home_score + away_score

        prob = pred.get("home_win_prob", 0.5)
        predicted_winner_correct = (prob > 0.5 and home_won) or (prob <= 0.5 and not home_won)
        if predicted_winner_correct:
            correct += 1

        # Brier score component
        actual_outcome = 1.0 if home_won else 0.0
        brier_sum += (prob - actual_outcome) ** 2

        # Spread error
        pred_spread = pred.get("predicted_spread", 0)
        spread_errors.append(abs(-pred_spread - actual_margin))  # spread is negative for favorites

        # Total error
        pred_total = pred.get("predicted_total", 220)
        total_errors.append(abs(pred_total - actual_total))

        # CLV
        market_prob = pred.get("market_prob", 0)
        if market_prob > 0:
            clv_values.append(prob - market_prob)

        # Update Elo with actual result
        elo.update(h_abbrev, a_abbrev, home_won, actual_margin,
                   actual.get("date", ""))

    if matched == 0:
        log("No predictions matched to actual results")
        return None

    accuracy = correct / matched
    brier = brier_sum / matched
    avg_spread_err = sum(spread_errors) / len(spread_errors) if spread_errors else 0
    avg_total_err = sum(total_errors) / len(total_errors) if total_errors else 0
    avg_clv = sum(clv_values) / len(clv_values) if clv_values else 0

    eval_result = {
        "n_evaluated": matched,
        "accuracy": round(accuracy, 4),
        "brier_score": round(brier, 4),
        "avg_spread_error": round(avg_spread_err, 1),
        "avg_total_error": round(avg_total_err, 1),
        "avg_clv": round(avg_clv, 4),
        "correct": correct,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    log(f"\nEVALUATE: {matched} predictions scored")
    log(f"  Accuracy:        {accuracy*100:.1f}% ({correct}/{matched})")
    log(f"  Brier Score:     {brier:.4f} (lower is better, 0.25 = random)")
    log(f"  Avg Spread Err:  {avg_spread_err:.1f} points")
    log(f"  Avg Total Err:   {avg_total_err:.1f} points")
    log(f"  Avg CLV:         {avg_clv*100:+.2f}% ({'beating market' if avg_clv > 0 else 'behind market'})")

    # Save eval to performance history
    perf_file = DATA_DIR / "performance" / "history.jsonl"
    perf_file.parent.mkdir(parents=True, exist_ok=True)
    with open(perf_file, "a") as f:
        f.write(json.dumps(eval_result, default=str) + "\n")

    log_training_event({
        "event": "evaluation",
        **eval_result,
    })

    # Save Elo state after updates
    elo.save()

    return eval_result


# ==============================================================================
# STEP 7: REPEAT — Check if we improved, revert if regressed
# ==============================================================================

def check_improvement(current_eval: dict) -> bool:
    """
    Compare current evaluation to previous best.
    If we regressed, revert to previous best weights.

    Returns True if we improved or maintained, False if reverted.
    """
    if not current_eval:
        return True

    current_brier = current_eval.get("brier_score", 1.0)

    # Load previous best
    if BEST_WEIGHTS_FILE.exists():
        try:
            best = json.loads(BEST_WEIGHTS_FILE.read_text())
            best_brier = best.get("brier_score", 1.0)
        except Exception:
            best_brier = 1.0
    else:
        best_brier = 1.0

    if current_brier < best_brier:
        # Improved -- save as new best
        best_data = {
            "brier_score": current_brier,
            "accuracy": current_eval.get("accuracy", 0),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        # Copy current model weights as best
        if ENSEMBLE_WEIGHTS_FILE.exists():
            best_data["weights"] = json.loads(ENSEMBLE_WEIGHTS_FILE.read_text())

        BEST_WEIGHTS_FILE.write_text(json.dumps(best_data, indent=2))

        improvement = (best_brier - current_brier) / max(best_brier, 0.001) * 100
        log(f"\nIMPROVED: Brier {best_brier:.4f} -> {current_brier:.4f} "
            f"({improvement:+.1f}%) -- new best weights saved")

        log_training_event({
            "event": "improvement",
            "old_brier": best_brier,
            "new_brier": current_brier,
            "improvement_pct": round(improvement, 2),
        })

        return True

    elif current_brier > best_brier * 1.05:  # Regressed by >5%
        # Revert to previous best weights
        log(f"\nREGRESSED: Brier {best_brier:.4f} -> {current_brier:.4f} -- "
            f"reverting to previous best weights", "WARN")

        try:
            best = json.loads(BEST_WEIGHTS_FILE.read_text())
            if "weights" in best:
                ENSEMBLE_WEIGHTS_FILE.write_text(json.dumps(best["weights"], indent=2))
                log("Reverted ensemble weights to previous best")
        except Exception as e:
            log(f"Failed to revert weights: {e}", "WARN")

        log_training_event({
            "event": "regression_revert",
            "best_brier": best_brier,
            "current_brier": current_brier,
        })

        return False

    else:
        log(f"\nSTABLE: Brier {current_brier:.4f} (best: {best_brier:.4f}) -- within tolerance")
        return True


# ==============================================================================
# FULL CYCLE
# ==============================================================================

def run_full_cycle():
    """Run the complete Karpathy training loop."""
    log("\n" + "=" * 70)
    log("KARPATHY LOOP — Full Cycle Starting")
    log("=" * 70)
    cycle_start = time.time()

    # Step 0: Load Elo state
    if not elo_system.load():
        log("No saved Elo state -- bootstrapping from power ratings")

    # Step 1: COLLECT
    log("\n--- STEP 1: COLLECT ---")
    games = collect_results()

    # Bootstrap if insufficient data
    if len(games) < 30:
        log(f"Only {len(games)} real games -- bootstrapping with synthetic data")
        synthetic = bootstrap_synthetic_games(500)
        games = synthetic + games
        log(f"Total after bootstrap: {len(games)} games")

    # Step 1.5: ENRICH — Load team stats, player stats, injuries, odds
    log("\n--- STEP 1.5: ENRICH ---")
    ts = load_team_stats()
    ps = load_player_stats()
    inj = load_injuries()
    odds = load_odds_history()

    # Step 2: FEATURES (now with 35 features)
    log("\n--- STEP 2: FEATURES ---")
    X, y_win, y_margin, y_total, meta = build_features(
        games, elo_system,
        team_stats=ts, player_stats=ps,
        injuries=inj, odds_history=odds
    )

    if X is None or len(X) < 20:
        log("Insufficient data for training -- skipping to prediction", "WARN")
        model_results = {}
        weights = compute_ensemble_weights({})
    else:
        # Step 3: TRAIN
        log("\n--- STEP 3: TRAIN ---")
        model_results = train_all_models(X, y_win, y_margin, y_total)

        # Step 4: ENSEMBLE
        log("\n--- STEP 4: ENSEMBLE ---")
        weights = compute_ensemble_weights(model_results)

    # Step 5: PREDICT (now with enrichment data)
    log("\n--- STEP 5: PREDICT ---")
    predictions = predict_today(model_results, weights, elo_system,
                                team_stats=ts, player_stats=ps, injuries=inj)

    # Step 6: EVALUATE
    log("\n--- STEP 6: EVALUATE ---")
    eval_result = evaluate_recent(elo_system)

    # Step 7: REPEAT (check improvement)
    log("\n--- STEP 7: CHECK IMPROVEMENT ---")
    improved = check_improvement(eval_result)

    # Save Elo state
    elo_system.save()

    elapsed = time.time() - cycle_start
    log(f"\n{'='*70}")
    log(f"KARPATHY LOOP COMPLETE in {elapsed:.1f}s")
    log(f"  Games processed:  {len(games)}")
    log(f"  Models trained:   {len(model_results) if model_results else 0}")
    log(f"  Predictions made: {len(predictions)}")
    log(f"  Eval result:      {'IMPROVED' if improved else 'REVERTED'}")
    log(f"{'='*70}\n")

    log_training_event({
        "event": "cycle_complete",
        "games": len(games),
        "models": len(model_results) if model_results else 0,
        "predictions": len(predictions),
        "improved": improved,
        "elapsed_s": round(elapsed, 1),
    })

    return {
        "games": len(games),
        "models": len(model_results) if model_results else 0,
        "predictions": len(predictions),
        "eval": eval_result,
        "improved": improved,
        "elapsed": elapsed,
    }


# ==============================================================================
# STATUS REPORT
# ==============================================================================

def show_status():
    """Show current model performance and training history."""
    log("\n" + "=" * 70)
    log("KARPATHY LOOP — Status Report")
    log("=" * 70)

    # Ensemble weights
    if ENSEMBLE_WEIGHTS_FILE.exists():
        try:
            ew = json.loads(ENSEMBLE_WEIGHTS_FILE.read_text())
            log("\nEnsemble Weights:")
            for name, w in sorted(ew.get("weights", {}).items(), key=lambda x: -x[1]):
                brier = ew.get("brier_scores", {}).get(name, "?")
                log(f"  {name:<20s} w={w:.3f} brier={brier}")
        except Exception:
            log("No ensemble weights file found")

    # Best weights
    if BEST_WEIGHTS_FILE.exists():
        try:
            bw = json.loads(BEST_WEIGHTS_FILE.read_text())
            log(f"\nBest Ever: Brier={bw.get('brier_score', '?')} "
                f"Acc={bw.get('accuracy', '?')} "
                f"(saved {bw.get('saved_at', '?')[:10]})")
        except Exception:
            pass

    # Training history (last 10 entries)
    if HISTORY_FILE.exists():
        lines = HISTORY_FILE.read_text().strip().splitlines()
        recent = lines[-10:] if len(lines) > 10 else lines
        log(f"\nTraining History (last {len(recent)} events):")
        for line in recent:
            try:
                entry = json.loads(line)
                event = entry.get("event", "?")
                ts = entry.get("ts", "?")[:16]
                if event == "cycle_complete":
                    log(f"  [{ts}] Cycle: {entry.get('games',0)} games, "
                        f"{entry.get('models',0)} models, "
                        f"{entry.get('predictions',0)} preds, "
                        f"{'OK' if entry.get('improved') else 'REVERTED'}")
                elif event == "evaluation":
                    log(f"  [{ts}] Eval: acc={entry.get('accuracy',0)*100:.1f}% "
                        f"brier={entry.get('brier_score',0):.4f} "
                        f"n={entry.get('n_evaluated',0)}")
                elif event == "improvement":
                    log(f"  [{ts}] Improved: {entry.get('old_brier',0):.4f} -> "
                        f"{entry.get('new_brier',0):.4f} "
                        f"({entry.get('improvement_pct',0):+.1f}%)")
                elif event == "regression_revert":
                    log(f"  [{ts}] REVERTED: {entry.get('current_brier',0):.4f} > "
                        f"{entry.get('best_brier',0):.4f}")
                else:
                    log(f"  [{ts}] {event}")
            except Exception:
                pass

    # Elo rankings
    if elo_system.ratings:
        log(f"\nElo Rankings (top 10):")
        sorted_elo = sorted(elo_system.ratings.items(), key=lambda x: -x[1])
        for i, (team, elo_val) in enumerate(sorted_elo[:10], 1):
            name = NBA_TEAMS.get(team, {}).get("name", team)
            log(f"  {i:2d}. {team} {name:<25s} {elo_val:.0f}")

    # Model files on disk
    model_files = list(MODELS_DIR.glob("*.pkl"))
    if model_files:
        log(f"\nSaved Models ({len(model_files)}):")
        for mf in sorted(model_files):
            size_kb = mf.stat().st_size / 1024
            log(f"  {mf.name:<30s} {size_kb:.1f} KB")

    log(f"\n{'='*70}\n")


# ==============================================================================
# DAEMON MODE
# ==============================================================================

def run_daemon():
    """
    Run the training loop as a nightly daemon.
    Executes at 3:00 AM UTC (after West Coast games finish).
    """
    log("\n" + "=" * 70)
    log("KARPATHY LOOP — Daemon Mode (nightly at 03:00 UTC)")
    log("=" * 70)

    while not _shutdown:
        now = datetime.now(timezone.utc)
        target_hour = 3
        target_minute = 0

        # Calculate next run time
        next_run = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        if now >= next_run:
            next_run += timedelta(days=1)

        wait_seconds = (next_run - now).total_seconds()
        log(f"Next cycle at {next_run.strftime('%Y-%m-%d %H:%M')} UTC "
            f"(in {wait_seconds/3600:.1f} hours)")

        # Wait, checking for shutdown every 60 seconds
        while wait_seconds > 0 and not _shutdown:
            sleep_time = min(60, wait_seconds)
            time.sleep(sleep_time)
            wait_seconds -= sleep_time

        if _shutdown:
            break

        # Run full cycle
        try:
            result = run_full_cycle()
            log(f"Daemon cycle complete: {result}")
        except Exception as e:
            log(f"Daemon cycle failed: {e}", "ERROR")
            import traceback
            log(traceback.format_exc(), "ERROR")

        # Brief pause before calculating next run
        time.sleep(10)

    log("Daemon shutting down gracefully")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Karpathy-Style Continuous Training Loop for NBA Quant Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 ops/karpathy-loop.py --full        # Run complete cycle (train + predict + eval)
  python3 ops/karpathy-loop.py --train       # Train models on all available data
  python3 ops/karpathy-loop.py --predict     # Predict today's games
  python3 ops/karpathy-loop.py --eval        # Evaluate recent predictions
  python3 ops/karpathy-loop.py --status      # Show model performance
  python3 ops/karpathy-loop.py --daemon      # Run nightly at 3 AM UTC
  python3 ops/karpathy-loop.py --bootstrap   # Generate synthetic training data
        """
    )
    parser.add_argument("--daemon", action="store_true", help="Run as nightly daemon (3 AM UTC)")
    parser.add_argument("--full", action="store_true", help="Run complete cycle once")
    parser.add_argument("--train", action="store_true", help="Train models on available data")
    parser.add_argument("--predict", action="store_true", help="Predict today's games")
    parser.add_argument("--eval", action="store_true", help="Evaluate recent predictions")
    parser.add_argument("--status", action="store_true", help="Show model performance status")
    parser.add_argument("--bootstrap", action="store_true", help="Generate synthetic training data")
    parser.add_argument("--k-factor", type=float, default=20.0, help="Elo K-factor (default: 20)")

    args = parser.parse_args()

    # Apply K-factor
    elo_system.k_factor = args.k_factor

    if args.daemon:
        run_daemon()

    elif args.full:
        run_full_cycle()

    elif args.train:
        elo_system.load()
        games = collect_results()
        if len(games) < 30:
            log(f"Only {len(games)} games -- bootstrapping")
            synthetic = bootstrap_synthetic_games(500)
            games = synthetic + games

        # Load enrichment data
        ts = load_team_stats()
        ps = load_player_stats()
        inj = load_injuries()
        odds = load_odds_history()

        X, y_win, y_margin, y_total, meta = build_features(
            games, elo_system,
            team_stats=ts, player_stats=ps,
            injuries=inj, odds_history=odds
        )
        if X is not None and len(X) >= 20:
            results = train_all_models(X, y_win, y_margin, y_total)
            weights = compute_ensemble_weights(results)
            elo_system.save()
        else:
            log("Cannot train: insufficient feature data", "ERROR")

    elif args.predict:
        elo_system.load()
        # Load saved models
        model_results = {}
        for model_name in ["logistic_regression", "random_forest", "xgboost", "lightgbm"]:
            model = load_model(model_name)
            if model is not None:
                model_results[model_name] = {
                    "model": model,
                    "uses_scaled": model_name == "logistic_regression",
                    "val_brier": 0.25,  # Default
                }
                log(f"Loaded model: {model_name}")

        # Load ensemble weights
        if ENSEMBLE_WEIGHTS_FILE.exists():
            weights = json.loads(ENSEMBLE_WEIGHTS_FILE.read_text()).get("weights", {})
        else:
            weights = compute_ensemble_weights({})

        # Load enrichment data for predictions
        ts = load_team_stats()
        ps = load_player_stats()
        inj = load_injuries()

        predict_today(model_results, weights, elo_system,
                      team_stats=ts, player_stats=ps, injuries=inj)

    elif args.eval:
        elo_system.load()
        evaluate_recent(elo_system)

    elif args.status:
        elo_system.load()
        show_status()

    elif args.bootstrap:
        games = bootstrap_synthetic_games(500)
        log(f"Bootstrap complete: {len(games)} synthetic games generated")

    else:
        # Default: show status + run full cycle
        elo_system.load()
        show_status()
        log("\nRun with --full to execute a training cycle, or --help for options")


if __name__ == "__main__":
    main()
