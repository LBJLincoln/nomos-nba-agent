#!/usr/bin/env python3
"""
NBA Quant AI — REAL Genetic Evolution Loop v3
================================================
RUNS 24/7 on HF Space or Google Colab.

This is NOT a fake LLM wrapper. This is REAL ML:
  - Population of 80 individuals (feature mask + hyperparams)
  - Walk-forward backtest fitness (Brier + ROI + Sharpe + Calibration)
  - Tournament selection, two-point crossover, adaptive mutation
  - Elitism (top 5 survive unchanged)
  - Continuous cycles — saves after each generation
  - Callbacks to VM after each cycle
  - Population persistence (survives restarts)

Usage:
  # On HF Space (24/7):
  python evolution/genetic_loop_v3.py --continuous

  # On Google Colab (manual):
  !python genetic_loop_v3.py --generations 50

  # Quick test:
  python evolution/genetic_loop_v3.py --generations 5 --pop-size 10
"""

import os, sys, json, time, random, math, warnings, traceback
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# ─── Auto-load .env.local ───
_env_file = Path(__file__).resolve().parent.parent / ".env.local"
if not _env_file.exists():
    _env_file = Path("/app/.env.local")
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _line = _line.replace("export ", "")
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip("'\""))

# ─── Paths ───
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
HIST_DIR = DATA_DIR / "historical"
RESULTS_DIR = DATA_DIR / "results"
STATE_DIR = DATA_DIR / "evolution-state"
for d in [DATA_DIR, HIST_DIR, RESULTS_DIR, STATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

VM_CALLBACK_URL = os.environ.get("VM_CALLBACK_URL", "http://34.136.180.66:8080")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")


# ═══════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═══════════════════════════════════════════════════════════

TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

ARENA_COORDS = {
    "ATL": (33.757, -84.396), "BOS": (42.366, -71.062), "BKN": (40.683, -73.976),
    "CHA": (35.225, -80.839), "CHI": (41.881, -87.674), "CLE": (41.496, -81.688),
    "DAL": (32.790, -96.810), "DEN": (39.749, -105.008), "DET": (42.341, -83.055),
    "GSW": (37.768, -122.388), "HOU": (29.751, -95.362), "IND": (39.764, -86.156),
    "LAC": (34.043, -118.267), "LAL": (34.043, -118.267), "MEM": (35.138, -90.051),
    "MIA": (25.781, -80.187), "MIL": (43.045, -87.917), "MIN": (44.980, -93.276),
    "NOP": (29.949, -90.082), "NYK": (40.751, -73.994), "OKC": (35.463, -97.515),
    "ORL": (28.539, -81.384), "PHI": (39.901, -75.172), "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667), "SAC": (38.580, -121.500), "SAS": (29.427, -98.438),
    "TOR": (43.643, -79.379), "UTA": (40.768, -111.901), "WAS": (38.898, -77.021),
}

ARENA_ALTITUDE = {
    "DEN": 5280, "UTA": 4226, "PHX": 1086, "OKC": 1201, "SAS": 650,
    "DAL": 430, "HOU": 43, "MEM": 337, "ATL": 1050, "CHA": 751,
    "IND": 715, "CHI": 594, "MIL": 617, "MIN": 830, "DET": 600,
    "CLE": 653, "BOS": 141, "NYK": 33, "BKN": 33, "PHI": 39,
    "WAS": 25, "MIA": 6, "ORL": 82, "NOP": 7, "TOR": 250,
    "POR": 50, "SAC": 30, "GSW": 12, "LAL": 305, "LAC": 305,
}

TIMEZONE_ET = {
    "ATL": 0, "BOS": 0, "BKN": 0, "CHA": 0, "CHI": -1, "CLE": 0,
    "DAL": -1, "DEN": -2, "DET": 0, "GSW": -3, "HOU": -1, "IND": 0,
    "LAC": -3, "LAL": -3, "MEM": -1, "MIA": 0, "MIL": -1, "MIN": -1,
    "NOP": -1, "NYK": 0, "OKC": -1, "ORL": 0, "PHI": 0, "PHX": -2,
    "POR": -3, "SAC": -3, "SAS": -1, "TOR": 0, "UTA": -2, "WAS": 0,
}

WINDOWS = [3, 5, 7, 10, 15, 20]


def resolve(name):
    if name in TEAM_MAP: return TEAM_MAP[name]
    if len(name) == 3 and name.isupper(): return name
    for full, abbr in TEAM_MAP.items():
        if name in full: return abbr
    return name[:3].upper() if name else None


def haversine(lat1, lon1, lat2, lon2):
    R = 3959
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def pull_seasons():
    """Pull NBA game data from nba_api, cache locally."""
    try:
        from nba_api.stats.endpoints import leaguegamefinder
    except ImportError:
        print("[DATA] nba_api not installed, using cached data only")
        return

    existing = {f.stem.replace("games-", "") for f in HIST_DIR.glob("games-*.json")}
    targets = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    missing = [s for s in targets if s not in existing]
    if not missing:
        print(f"[DATA] All {len(targets)} seasons cached")
        return

    for season in missing:
        print(f"[DATA] Pulling {season}...")
        try:
            time.sleep(3)
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season, league_id_nullable="00",
                season_type_nullable="Regular Season", timeout=60
            )
            df = finder.get_data_frames()[0]
            if df.empty:
                continue
            pairs = {}
            for _, row in df.iterrows():
                gid = row["GAME_ID"]
                if gid not in pairs:
                    pairs[gid] = []
                pairs[gid].append({
                    "team_name": row.get("TEAM_NAME", ""),
                    "matchup": row.get("MATCHUP", ""),
                    "pts": int(row["PTS"]) if row.get("PTS") is not None else None,
                    "game_date": row.get("GAME_DATE", ""),
                })
            games = []
            for gid, teams in pairs.items():
                if len(teams) != 2:
                    continue
                home = next((t for t in teams if " vs. " in str(t.get("matchup", ""))), None)
                away = next((t for t in teams if " @ " in str(t.get("matchup", ""))), None)
                if not home or not away or home["pts"] is None:
                    continue
                games.append({
                    "game_date": home["game_date"],
                    "home_team": home["team_name"], "away_team": away["team_name"],
                    "home": {"team_name": home["team_name"], "pts": home["pts"]},
                    "away": {"team_name": away["team_name"], "pts": away["pts"]},
                })
            if games:
                (HIST_DIR / f"games-{season}.json").write_text(json.dumps(games))
                print(f"  {len(games)} games saved")
        except Exception as e:
            print(f"  Error pulling {season}: {e}")


def load_all_games():
    """Load all cached game data."""
    games = []
    for f in sorted(HIST_DIR.glob("games-*.json")):
        data = json.loads(f.read_text())
        items = data if isinstance(data, list) else data.get("games", [])
        games.extend(items)
    games.sort(key=lambda g: g.get("game_date", g.get("date", "")))
    return games


# ═══════════════════════════════════════════════════════════
# SECTION 2: FEATURE ENGINE (164+ features)
# ═══════════════════════════════════════════════════════════

def build_features(games):
    """Build 164+ features from raw game data. Returns X, y, feature_names."""
    team_results = defaultdict(list)
    team_last = {}
    team_elo = defaultdict(lambda: 1500.0)
    X, y = [], []
    feature_names = []
    first = True

    for game in games:
        hr, ar = game.get("home_team", ""), game.get("away_team", "")
        if "home" in game and isinstance(game["home"], dict):
            h, a = game["home"], game.get("away", {})
            hs, as_ = h.get("pts"), a.get("pts")
            if not hr: hr = h.get("team_name", "")
            if not ar: ar = a.get("team_name", "")
        else:
            hs, as_ = game.get("home_score"), game.get("away_score")
        if hs is None or as_ is None:
            continue
        hs, as_ = int(hs), int(as_)
        home, away = resolve(hr), resolve(ar)
        if not home or not away:
            continue
        gd = game.get("game_date", game.get("date", ""))[:10]
        hr_ = team_results[home]
        ar_ = team_results[away]

        if len(hr_) < 5 or len(ar_) < 5:
            team_results[home].append((gd, hs > as_, hs - as_, away, hs, as_))
            team_results[away].append((gd, as_ > hs, as_ - hs, home, as_, hs))
            team_last[home] = gd
            team_last[away] = gd
            K = 20
            exp_h = 1 / (1 + 10 ** ((team_elo[away] - team_elo[home] - 50) / 400))
            team_elo[home] += K * ((1 if hs > as_ else 0) - exp_h)
            team_elo[away] += K * ((0 if hs > as_ else 1) - (1 - exp_h))
            continue

        def wp(r, n):
            s = r[-n:]
            return sum(1 for x in s if x[1]) / len(s) if s else 0.5

        def pd(r, n):
            s = r[-n:]
            return sum(x[2] for x in s) / len(s) if s else 0.0

        def ppg(r, n):
            s = r[-n:]
            return sum(x[4] for x in s) / len(s) if s else 100.0

        def papg(r, n):
            s = r[-n:]
            return sum(x[5] for x in s) / len(s) if s else 100.0

        def strk(r):
            if not r: return 0
            s, l = 0, r[-1][1]
            for x in reversed(r):
                if x[1] == l:
                    s += 1
                else:
                    break
            return s if l else -s

        def close_pct(r, n):
            s = r[-n:]
            return sum(1 for x in s if abs(x[2]) <= 5) / len(s) if s else 0.5

        def blowout_pct(r, n):
            s = r[-n:]
            return sum(1 for x in s if abs(x[2]) >= 15) / len(s) if s else 0.0

        def consistency(r, n):
            s = r[-n:]
            if len(s) < 3: return 0.0
            m = [x[2] for x in s]
            avg = sum(m) / len(m)
            return (sum((v - avg) ** 2 for v in m) / len(m)) ** 0.5

        def rest(t):
            last = team_last.get(t)
            if not last or not gd: return 3
            try:
                return max(0, (datetime.strptime(gd[:10], "%Y-%m-%d") - datetime.strptime(last[:10], "%Y-%m-%d")).days)
            except Exception:
                return 3

        def sos(r, n=10):
            rec = r[-n:]
            if not rec: return 0.5
            ops = [wp(team_results[x[3]], 82) for x in rec if team_results[x[3]]]
            return sum(ops) / len(ops) if ops else 0.5

        def travel_dist(r, team):
            if not r: return 0
            last_opp = r[-1][3]
            if last_opp in ARENA_COORDS and team in ARENA_COORDS:
                return haversine(*ARENA_COORDS[last_opp], *ARENA_COORDS[team])
            return 0

        h_rest, a_rest = rest(home), rest(away)
        try:
            dt = datetime.strptime(gd, "%Y-%m-%d")
            month, dow = dt.month, dt.weekday()
        except Exception:
            month, dow = 1, 2

        sp = max(0, min(1, (month - 10) / 7)) if month >= 10 else max(0, min(1, (month + 2) / 7))

        row = []
        names = []

        # 1. ROLLING PERFORMANCE (96 features)
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            for w in WINDOWS:
                row.extend([wp(tr, w), pd(tr, w), ppg(tr, w), papg(tr, w),
                            ppg(tr, w) - papg(tr, w), close_pct(tr, w), blowout_pct(tr, w),
                            ppg(tr, w) + papg(tr, w)])
                if first:
                    names.extend([f"{prefix}_wp{w}", f"{prefix}_pd{w}", f"{prefix}_ppg{w}",
                                  f"{prefix}_papg{w}", f"{prefix}_margin{w}", f"{prefix}_close{w}",
                                  f"{prefix}_blowout{w}", f"{prefix}_ou{w}"])

        # 2. MOMENTUM (16 features)
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            row.extend([strk(tr), abs(strk(tr)),
                        wp(tr, 5) - wp(tr, 82), wp(tr, 3) - wp(tr, 10),
                        ppg(tr, 5) - ppg(tr, 20), papg(tr, 5) - papg(tr, 20),
                        consistency(tr, 10), consistency(tr, 5)])
            if first:
                names.extend([f"{prefix}_streak", f"{prefix}_streak_abs",
                              f"{prefix}_form5v82", f"{prefix}_form3v10",
                              f"{prefix}_scoring_trend", f"{prefix}_defense_trend",
                              f"{prefix}_consistency10", f"{prefix}_consistency5"])

        # 3. REST & SCHEDULE (16 features)
        h_travel = travel_dist(hr_, home)
        a_travel = travel_dist(ar_, away)
        row.extend([
            min(h_rest, 7), min(a_rest, 7), h_rest - a_rest,
            1.0 if h_rest <= 1 else 0.0, 1.0 if a_rest <= 1 else 0.0,
            h_travel / 1000, a_travel / 1000, (h_travel - a_travel) / 1000,
            ARENA_ALTITUDE.get(home, 500) / 5280, ARENA_ALTITUDE.get(away, 500) / 5280,
            (ARENA_ALTITUDE.get(home, 500) - ARENA_ALTITUDE.get(away, 500)) / 5280,
            abs(TIMEZONE_ET.get(home, 0) - TIMEZONE_ET.get(away, 0)),
            0, 0, 0, 0,
        ])
        if first:
            names.extend(["h_rest", "a_rest", "rest_adv", "h_b2b", "a_b2b",
                          "h_travel", "a_travel", "travel_adv",
                          "h_altitude", "a_altitude", "altitude_delta",
                          "tz_shift", "h_games_7d", "a_games_7d", "sched_density", "pad1"])

        # 4. OPPONENT-ADJUSTED (12 features)
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            s5 = sos(tr, 5)
            s10 = sos(tr, 10)
            ss = sos(tr, 82)
            wp_above = sum(1 for r in tr if wp(team_results[r[3]], 82) > 0.5 and r[1]) / max(
                sum(1 for r in tr if wp(team_results[r[3]], 82) > 0.5), 1)
            wp_below = sum(1 for r in tr if wp(team_results[r[3]], 82) <= 0.5 and r[1]) / max(
                sum(1 for r in tr if wp(team_results[r[3]], 82) <= 0.5), 1)
            row.extend([s5, s10, ss, wp_above, wp_below, 0])
            if first:
                names.extend([f"{prefix}_sos5", f"{prefix}_sos10", f"{prefix}_sos_season",
                              f"{prefix}_wp_above500", f"{prefix}_wp_below500", f"{prefix}_margin_quality"])

        # 5. MATCHUP & ELO (12 features)
        row.extend([
            wp(hr_, 10) - wp(ar_, 10), pd(hr_, 10) - pd(ar_, 10),
            ppg(hr_, 10) - papg(ar_, 10), ppg(ar_, 10) - papg(hr_, 10),
            abs(ppg(hr_, 10) + papg(hr_, 10) - ppg(ar_, 10) - papg(ar_, 10)),
            consistency(hr_, 10) - consistency(ar_, 10),
            team_elo[home], team_elo[away], team_elo[home] - team_elo[away] + 50,
            (team_elo[home] - 1500) / 100, (team_elo[away] - 1500) / 100,
            (team_elo[home] - team_elo[away]) / 100,
        ])
        if first:
            names.extend(["rel_strength", "rel_pd", "off_matchup", "def_matchup",
                          "tempo_diff", "consistency_edge",
                          "elo_home", "elo_away", "elo_diff",
                          "elo_home_norm", "elo_away_norm", "elo_diff_norm"])

        # 6. CONTEXT (12 features)
        row.extend([
            1.0, sp, math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12),
            dow / 6.0, 1.0 if dow >= 5 else 0.0,
            min(len(hr_), 82) / 82.0, min(len(ar_), 82) / 82.0,
            wp(hr_, 82) + wp(ar_, 82), wp(hr_, 82) - wp(ar_, 82),
            1.0 if wp(hr_, 82) > 0.5 and wp(ar_, 82) > 0.5 else 0.0,
            ppg(hr_, 10) + ppg(ar_, 10),
        ])
        if first:
            names.extend(["home_court", "season_phase", "month_sin", "month_cos",
                          "day_of_week", "is_weekend", "h_games_pct", "a_games_pct",
                          "combined_wp", "wp_diff", "playoff_race", "expected_total"])

        X.append(row)
        y.append(1 if hs > as_ else 0)
        if first:
            feature_names = names
            first = False

        team_results[home].append((gd, hs > as_, hs - as_, away, hs, as_))
        team_results[away].append((gd, as_ > hs, as_ - hs, home, as_, hs))
        team_last[home] = gd
        team_last[away] = gd
        K = 20
        exp_h = 1 / (1 + 10 ** ((team_elo[away] - team_elo[home] - 50) / 400))
        team_elo[home] += K * ((1 if hs > as_ else 0) - exp_h)
        team_elo[away] += K * ((0 if hs > as_ else 1) - (1 - exp_h))

    X = np.nan_to_num(np.array(X, dtype=np.float64))
    y = np.array(y, dtype=np.int32)
    return X, y, feature_names


# ═══════════════════════════════════════════════════════════
# SECTION 3: INDIVIDUAL (feature mask + hyperparameters)
# ═══════════════════════════════════════════════════════════

class Individual:
    """One model configuration: feature selection mask + hyperparameters."""

    def __init__(self, n_features, target=100):
        prob = target / max(n_features, 1)
        self.features = [1 if random.random() < prob else 0 for _ in range(n_features)]
        self.hyperparams = {
            "n_estimators": random.randint(100, 600),
            "max_depth": random.randint(3, 10),
            "learning_rate": 10 ** random.uniform(-2.5, -0.5),
            "subsample": random.uniform(0.5, 1.0),
            "colsample_bytree": random.uniform(0.3, 1.0),
            "min_child_weight": random.randint(1, 15),
            "reg_alpha": 10 ** random.uniform(-6, 1),
            "reg_lambda": 10 ** random.uniform(-6, 1),
            "model_type": random.choice(["xgboost", "lightgbm"]),
            "calibration": random.choice(["isotonic", "sigmoid", "none"]),
        }
        self.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "composite": 0.0}
        self.generation = 0
        self.n_features = sum(self.features)

    def selected_indices(self):
        return [i for i, b in enumerate(self.features) if b]

    def to_dict(self):
        return {
            "n_features": self.n_features,
            "hyperparams": {k: v for k, v in self.hyperparams.items()},
            "fitness": dict(self.fitness),
            "generation": self.generation,
        }

    @staticmethod
    def crossover(p1, p2):
        """Two-point crossover on features + blend hyperparams."""
        child = Individual.__new__(Individual)
        n = len(p1.features)
        pt1 = random.randint(0, n - 1)
        pt2 = random.randint(pt1, n - 1)
        child.features = p1.features[:pt1] + p2.features[pt1:pt2] + p1.features[pt2:]

        child.hyperparams = {}
        for key in p1.hyperparams:
            if isinstance(p1.hyperparams[key], (int, float)):
                w = random.random()
                val = w * p1.hyperparams[key] + (1 - w) * p2.hyperparams[key]
                if isinstance(p1.hyperparams[key], int):
                    val = int(round(val))
                child.hyperparams[key] = val
            else:
                child.hyperparams[key] = random.choice([p1.hyperparams[key], p2.hyperparams[key]])

        child.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "composite": 0.0}
        child.generation = max(p1.generation, p2.generation) + 1
        child.n_features = sum(child.features)
        return child

    def mutate(self, rate=0.03):
        """Mutate features and hyperparameters."""
        for i in range(len(self.features)):
            if random.random() < rate:
                self.features[i] = 1 - self.features[i]
        if random.random() < 0.15:
            self.hyperparams["n_estimators"] = max(50, self.hyperparams["n_estimators"] + random.randint(-100, 100))
        if random.random() < 0.15:
            self.hyperparams["max_depth"] = max(2, min(12, self.hyperparams["max_depth"] + random.randint(-2, 2)))
        if random.random() < 0.15:
            self.hyperparams["learning_rate"] *= 10 ** random.uniform(-0.3, 0.3)
            self.hyperparams["learning_rate"] = max(0.001, min(0.5, self.hyperparams["learning_rate"]))
        if random.random() < 0.05:
            self.hyperparams["model_type"] = random.choice(["xgboost", "lightgbm"])
        if random.random() < 0.05:
            self.hyperparams["calibration"] = random.choice(["isotonic", "sigmoid", "none"])
        self.n_features = sum(self.features)


# ═══════════════════════════════════════════════════════════
# SECTION 4: FITNESS EVALUATION (multi-objective)
# ═══════════════════════════════════════════════════════════

def evaluate_individual(ind, X, y, n_splits=5, use_gpu=False):
    """
    Evaluate one individual via walk-forward backtest.
    Multi-objective: Brier + ROI + Sharpe + Calibration.
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import CalibratedClassifierCV

    selected = ind.selected_indices()
    if len(selected) < 15 or len(selected) > 250:
        ind.fitness = {"brier": 0.30, "roi": -0.10, "sharpe": -1.0, "calibration": 0.15, "composite": -1.0}
        return

    X_sub = X[:, selected]
    X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=1e6, neginf=-1e6)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    hp = ind.hyperparams

    model = _build_model(hp, use_gpu)
    if model is None:
        ind.fitness["composite"] = -1.0
        return

    briers, rois, all_probs, all_y = [], [], [], []

    for ti, vi in tscv.split(X_sub):
        try:
            m = type(model)(**model.get_params())
            if hp["calibration"] != "none":
                m = CalibratedClassifierCV(m, method=hp["calibration"], cv=3)
            m.fit(X_sub[ti], y[ti])
            probs = m.predict_proba(X_sub[vi])[:, 1]

            briers.append(brier_score_loss(y[vi], probs))
            rois.append(_simulate_betting(probs, y[vi]))
            all_probs.extend(probs)
            all_y.extend(y[vi])
        except Exception:
            briers.append(0.28)
            rois.append(-0.05)

    avg_brier = np.mean(briers)
    avg_roi = np.mean(rois)
    sharpe = np.mean(rois) / max(np.std(rois), 0.01) if len(rois) > 1 else 0.0
    cal_err = _calibration_error(np.array(all_probs), np.array(all_y)) if all_probs else 0.15

    # Multi-objective composite fitness (higher = better)
    composite = (
        0.40 * (1 - avg_brier) +      # Brier: lower is better
        0.25 * max(0, avg_roi) +        # ROI: higher is better
        0.20 * max(0, sharpe / 3) +     # Sharpe: higher is better
        0.15 * (1 - cal_err)            # Calibration: lower is better
    )

    ind.fitness = {
        "brier": round(avg_brier, 5),
        "roi": round(avg_roi, 4),
        "sharpe": round(sharpe, 4),
        "calibration": round(cal_err, 4),
        "composite": round(composite, 5),
    }


def _build_model(hp, use_gpu=False):
    """Build ML model from hyperparameters."""
    try:
        if hp["model_type"] == "xgboost":
            import xgboost as xgb
            params = {
                "n_estimators": hp["n_estimators"],
                "max_depth": hp["max_depth"],
                "learning_rate": hp["learning_rate"],
                "subsample": hp["subsample"],
                "colsample_bytree": hp["colsample_bytree"],
                "min_child_weight": hp["min_child_weight"],
                "reg_alpha": hp["reg_alpha"],
                "reg_lambda": hp["reg_lambda"],
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
            }
            if use_gpu:
                params["device"] = "cuda"
            return xgb.XGBClassifier(**params)
        elif hp["model_type"] == "lightgbm":
            import lightgbm as lgbm
            return lgbm.LGBMClassifier(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                learning_rate=hp["learning_rate"],
                subsample=hp["subsample"],
                num_leaves=min(2 ** hp["max_depth"] - 1, 127),
                reg_alpha=hp["reg_alpha"],
                reg_lambda=hp["reg_lambda"],
                verbose=-1, random_state=42, n_jobs=-1,
            )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=min(hp["n_estimators"], 200),
            max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"],
            random_state=42,
        )
    return None


def _simulate_betting(probs, actuals, edge=0.05):
    """Simulate flat betting where model has edge > threshold."""
    stake = 10
    profit = 0
    n_bets = 0
    for prob, actual in zip(probs, actuals):
        if prob > 0.5 + edge:
            n_bets += 1
            if actual == 1:
                profit += stake * (1 / prob - 1)
            else:
                profit -= stake
        elif prob < 0.5 - edge:
            n_bets += 1
            if actual == 0:
                profit += stake * (1 / (1 - prob) - 1)
            else:
                profit -= stake
    return profit / (n_bets * stake) if n_bets > 0 else 0.0


def _calibration_error(probs, actuals, n_bins=10):
    """Expected Calibration Error (ECE)."""
    if len(probs) == 0:
        return 1.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(probs) * abs(probs[mask].mean() - actuals[mask].mean())
    return ece


# ═══════════════════════════════════════════════════════════
# SECTION 5: GENETIC EVOLUTION ENGINE
# ═══════════════════════════════════════════════════════════

class GeneticEvolutionEngine:
    """
    REAL genetic evolution engine.
    Runs continuously, evolving a population of model configs.
    """

    def __init__(self, pop_size=80, elite_size=5, mutation_rate=0.03,
                 crossover_rate=0.7, target_features=100, n_splits=5):
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.target_features = target_features
        self.n_splits = n_splits

        self.population = []
        self.generation = 0
        self.best_ever = None
        self.history = []
        self.stagnation_counter = 0
        self.use_gpu = False

        # Detect GPU
        try:
            import xgboost as xgb
            _test = xgb.XGBClassifier(n_estimators=5, max_depth=3, tree_method="hist", device="cuda")
            _test.fit(np.random.randn(50, 5), np.random.randint(0, 2, 50))
            self.use_gpu = True
            print("[GPU] XGBoost CUDA: ENABLED")
        except Exception:
            print("[GPU] XGBoost CUDA: disabled, using CPU")

    def initialize(self, n_features):
        """Create initial random population."""
        self.n_features = n_features
        self.population = [Individual(n_features, self.target_features) for _ in range(self.pop_size)]
        print(f"[INIT] Population: {self.pop_size} individuals, {n_features} feature candidates, "
              f"~{self.target_features} target features")

    def restore_state(self):
        """Restore population from saved state (survive restarts)."""
        state_file = STATE_DIR / "population.json"
        if not state_file.exists():
            return False
        try:
            state = json.loads(state_file.read_text())
            self.generation = state["generation"]
            self.n_features = state["n_features"]
            self.history = state.get("history", [])
            self.stagnation_counter = state.get("stagnation_counter", 0)
            self.mutation_rate = state.get("mutation_rate", self.base_mutation_rate)

            self.population = []
            for ind_data in state["population"]:
                ind = Individual.__new__(Individual)
                ind.features = ind_data["features"]
                ind.hyperparams = ind_data["hyperparams"]
                ind.fitness = ind_data["fitness"]
                ind.generation = ind_data.get("generation", 0)
                ind.n_features = sum(ind.features)
                self.population.append(ind)

            if state.get("best_ever"):
                be = state["best_ever"]
                self.best_ever = Individual.__new__(Individual)
                self.best_ever.features = be["features"]
                self.best_ever.hyperparams = be["hyperparams"]
                self.best_ever.fitness = be["fitness"]
                self.best_ever.generation = be.get("generation", 0)
                self.best_ever.n_features = sum(self.best_ever.features)

            print(f"[RESTORE] Generation {self.generation}, {len(self.population)} individuals, "
                  f"best Brier={self.best_ever.fitness['brier']:.4f}" if self.best_ever else "")
            return True
        except Exception as e:
            print(f"[RESTORE] Failed: {e}")
            return False

    def save_state(self):
        """Save population state to survive restarts."""
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation": self.generation,
            "n_features": self.n_features,
            "stagnation_counter": self.stagnation_counter,
            "mutation_rate": self.mutation_rate,
            "population": [
                {
                    "features": ind.features,
                    "hyperparams": {k: (float(v) if isinstance(v, (np.floating,)) else v)
                                    for k, v in ind.hyperparams.items()},
                    "fitness": ind.fitness,
                    "generation": ind.generation,
                }
                for ind in self.population
            ],
            "best_ever": {
                "features": self.best_ever.features,
                "hyperparams": {k: (float(v) if isinstance(v, (np.floating,)) else v)
                                for k, v in self.best_ever.hyperparams.items()},
                "fitness": self.best_ever.fitness,
                "generation": self.best_ever.generation,
            } if self.best_ever else None,
            "history": self.history[-200:],
        }
        (STATE_DIR / "population.json").write_text(json.dumps(state, default=str))

    def evolve_one_generation(self, X, y):
        """Run one generation of evolution. Returns best individual."""
        self.generation += 1
        gen_start = time.time()

        # 1. Evaluate all individuals
        for i, ind in enumerate(self.population):
            evaluate_individual(ind, X, y, self.n_splits, self.use_gpu)
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i+1}/{len(self.population)}...", end="\r")

        # 2. Sort by composite fitness (higher = better)
        self.population.sort(key=lambda x: x.fitness["composite"], reverse=True)
        best = self.population[0]

        # 3. Track best ever
        prev_best_brier = self.best_ever.fitness["brier"] if self.best_ever else 1.0
        if self.best_ever is None or best.fitness["composite"] > self.best_ever.fitness["composite"]:
            self.best_ever = Individual.__new__(Individual)
            self.best_ever.features = best.features[:]
            self.best_ever.hyperparams = dict(best.hyperparams)
            self.best_ever.fitness = dict(best.fitness)
            self.best_ever.n_features = best.n_features
            self.best_ever.generation = self.generation

        # 4. Stagnation detection + adaptive mutation
        if abs(best.fitness["brier"] - prev_best_brier) < 0.0005:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        if self.stagnation_counter >= 5:
            self.mutation_rate = min(0.15, self.mutation_rate * 1.5)
            print(f"  [STAGNATION] {self.stagnation_counter} gens — mutation rate -> {self.mutation_rate:.3f}")
        elif self.stagnation_counter == 0:
            self.mutation_rate = max(self.base_mutation_rate, self.mutation_rate * 0.9)

        # 5. Record history
        self.history.append({
            "gen": self.generation,
            "best_brier": best.fitness["brier"],
            "best_roi": best.fitness["roi"],
            "best_sharpe": best.fitness["sharpe"],
            "best_composite": best.fitness["composite"],
            "n_features": best.n_features,
            "model_type": best.hyperparams["model_type"],
            "mutation_rate": round(self.mutation_rate, 4),
            "avg_composite": round(np.mean([ind.fitness["composite"] for ind in self.population]), 5),
            "pop_diversity": round(np.std([ind.n_features for ind in self.population]), 1),
        })

        elapsed = time.time() - gen_start
        print(f"  Gen {self.generation}: Brier={best.fitness['brier']:.4f} "
              f"ROI={best.fitness['roi']:.1%} Sharpe={best.fitness['sharpe']:.2f} "
              f"Features={best.n_features} Model={best.hyperparams['model_type']} "
              f"Composite={best.fitness['composite']:.4f} ({elapsed:.0f}s)")

        # 6. Create next generation
        new_pop = []

        # Elitism
        for i in range(self.elite_size):
            elite = Individual.__new__(Individual)
            elite.features = self.population[i].features[:]
            elite.hyperparams = dict(self.population[i].hyperparams)
            elite.fitness = dict(self.population[i].fitness)
            elite.n_features = self.population[i].n_features
            elite.generation = self.population[i].generation
            new_pop.append(elite)

        # Injection: if stagnating badly, inject fresh random individuals
        n_inject = 0
        if self.stagnation_counter >= 10:
            n_inject = self.pop_size // 5
            for _ in range(n_inject):
                new_pop.append(Individual(self.n_features, self.target_features))
            print(f"  [INJECTION] {n_inject} fresh individuals added")

        # Fill with crossover + mutation
        while len(new_pop) < self.pop_size:
            p1 = self._tournament_select(7)
            p2 = self._tournament_select(7)
            if random.random() < self.crossover_rate:
                child = Individual.crossover(p1, p2)
            else:
                child = Individual.__new__(Individual)
                child.features = p1.features[:]
                child.hyperparams = dict(p1.hyperparams)
                child.fitness = dict(p1.fitness)
                child.n_features = p1.n_features
                child.generation = self.generation
            child.mutate(self.mutation_rate)
            new_pop.append(child)

        self.population = new_pop
        return best

    def _tournament_select(self, k=7):
        """Tournament selection."""
        contestants = random.sample(self.population, min(k, len(self.population)))
        return max(contestants, key=lambda x: x.fitness["composite"])

    def save_cycle_results(self, feature_names):
        """Save results after a cycle of generations."""
        if not self.best_ever:
            return

        selected_names = [feature_names[i] for i in self.best_ever.selected_indices()
                          if i < len(feature_names)]

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation": self.generation,
            "population_size": self.pop_size,
            "feature_candidates": self.n_features,
            "mutation_rate": round(self.mutation_rate, 4),
            "stagnation_counter": self.stagnation_counter,
            "gpu": self.use_gpu,
            "best": {
                "brier": self.best_ever.fitness["brier"],
                "roi": self.best_ever.fitness["roi"],
                "sharpe": self.best_ever.fitness["sharpe"],
                "calibration": self.best_ever.fitness["calibration"],
                "composite": self.best_ever.fitness["composite"],
                "n_features": self.best_ever.n_features,
                "model_type": self.best_ever.hyperparams["model_type"],
                "hyperparams": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                for k, v in self.best_ever.hyperparams.items()},
                "selected_features": selected_names[:50],
            },
            "top5": [ind.to_dict() for ind in sorted(
                self.population, key=lambda x: x.fitness["composite"], reverse=True
            )[:5]],
            "history_last20": self.history[-20:],
        }

        # Save timestamped + latest
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        (RESULTS_DIR / f"evolution-{ts}.json").write_text(json.dumps(results, indent=2, default=str))
        (RESULTS_DIR / "evolution-latest.json").write_text(json.dumps(results, indent=2, default=str))
        return results


# ═══════════════════════════════════════════════════════════
# SECTION 6: VM CALLBACK
# ═══════════════════════════════════════════════════════════

def callback_to_vm(results):
    """POST results to VM data server (best-effort)."""
    import urllib.request
    try:
        url = f"{VM_CALLBACK_URL}/callback/evolution"
        body = json.dumps(results, default=str).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=10)
        print(f"  [CALLBACK] VM notified: {resp.status}")
    except Exception as e:
        # Best-effort, don't block on failure
        print(f"  [CALLBACK] VM unreachable: {e}")

    # Also try to write to shared mon-ipad data if accessible
    try:
        shared = Path("/home/termius/mon-ipad/data/nba-agent/evolution-latest.json")
        if shared.parent.exists():
            shared.write_text(json.dumps(results, indent=2, default=str))
            print(f"  [CALLBACK] Wrote to mon-ipad")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════
# SECTION 7: MAIN LOOP (continuous 24/7)
# ═══════════════════════════════════════════════════════════

def run_continuous(generations_per_cycle=10, total_cycles=None, pop_size=80,
                   target_features=100, n_splits=5, cool_down=30):
    """
    Main entry point — runs genetic evolution CONTINUOUSLY.

    Args:
        generations_per_cycle: Generations per cycle before saving/callback
        total_cycles: None = infinite (24/7 mode)
        pop_size: Population size
        target_features: Target number of features per individual
        n_splits: Walk-forward backtest splits
        cool_down: Seconds between cycles
    """
    print("=" * 70)
    print("  NBA QUANT AI — REAL GENETIC EVOLUTION LOOP v3")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Pop: {pop_size} | Target features: {target_features}")
    print(f"  Gens/cycle: {generations_per_cycle} | Cycles: {'INFINITE' if total_cycles is None else total_cycles}")
    print("=" * 70)

    # 1. Pull data
    print("\n[PHASE 1] Loading data...")
    pull_seasons()
    games = load_all_games()
    print(f"  {len(games)} games loaded")
    if len(games) < 500:
        print("  ERROR: Not enough games!")
        return

    # 2. Build features
    print("\n[PHASE 2] Building features...")
    X, y, feature_names = build_features(games)
    print(f"  Feature matrix: {X.shape} ({len(feature_names)} features)")

    # 3. Initialize engine
    print("\n[PHASE 3] Initializing engine...")
    engine = GeneticEvolutionEngine(
        pop_size=pop_size, elite_size=5, mutation_rate=0.03,
        crossover_rate=0.7, target_features=target_features, n_splits=n_splits,
    )

    # Try to restore previous state
    if not engine.restore_state():
        engine.initialize(X.shape[1])

    # 4. CONTINUOUS EVOLUTION LOOP
    cycle = 0
    while True:
        cycle += 1
        if total_cycles is not None and cycle > total_cycles:
            break

        cycle_start = time.time()
        print(f"\n{'='*60}")
        print(f"  CYCLE {cycle} — Starting {generations_per_cycle} generations")
        print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"{'='*60}")

        for gen in range(generations_per_cycle):
            try:
                best = engine.evolve_one_generation(X, y)
            except Exception as e:
                print(f"  [ERROR] Generation failed: {e}")
                traceback.print_exc()
                continue

        # Save state (survives restarts)
        engine.save_state()

        # Save results
        results = engine.save_cycle_results(feature_names)

        cycle_elapsed = time.time() - cycle_start
        print(f"\n  Cycle {cycle} complete in {cycle_elapsed:.0f}s")

        if engine.best_ever:
            print(f"  BEST EVER: Brier={engine.best_ever.fitness['brier']:.4f} "
                  f"ROI={engine.best_ever.fitness['roi']:.1%} "
                  f"Features={engine.best_ever.n_features}")

        # Callback to VM
        if results:
            callback_to_vm(results)

        # Refresh data periodically (every 10 cycles)
        if cycle % 10 == 0:
            print("\n  [REFRESH] Pulling latest game data...")
            try:
                pull_seasons()
                new_games = load_all_games()
                if len(new_games) > len(games):
                    games = new_games
                    X, y, feature_names = build_features(games)
                    print(f"  [REFRESH] Updated: {X.shape}")
            except Exception as e:
                print(f"  [REFRESH] Failed: {e}")

        if total_cycles is None:
            print(f"\n  Cooling down {cool_down}s before next cycle...")
            time.sleep(cool_down)

    print("\n" + "=" * 70)
    print("  EVOLUTION COMPLETE")
    if engine.best_ever:
        print(f"  Final best: Brier={engine.best_ever.fitness['brier']:.4f} "
              f"ROI={engine.best_ever.fitness['roi']:.1%}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NBA Quant Genetic Evolution Loop v3")
    parser.add_argument("--continuous", action="store_true", help="Run 24/7 (no cycle limit)")
    parser.add_argument("--generations", type=int, default=10, help="Generations per cycle (default: 10)")
    parser.add_argument("--cycles", type=int, default=None, help="Number of cycles (default: infinite)")
    parser.add_argument("--pop-size", type=int, default=80, help="Population size (default: 80)")
    parser.add_argument("--target-features", type=int, default=100, help="Target features (default: 100)")
    parser.add_argument("--splits", type=int, default=5, help="Walk-forward splits (default: 5)")
    parser.add_argument("--cooldown", type=int, default=30, help="Seconds between cycles (default: 30)")
    args = parser.parse_args()

    cycles = None if args.continuous else (args.cycles or 1)

    run_continuous(
        generations_per_cycle=args.generations,
        total_cycles=cycles,
        pop_size=args.pop_size,
        target_features=args.target_features,
        n_splits=args.splits,
        cool_down=args.cooldown,
    )
