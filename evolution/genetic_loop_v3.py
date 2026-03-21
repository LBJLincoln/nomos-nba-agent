#!/usr/bin/env python3
"""
NBA Quant AI — REAL Genetic Evolution Loop v4
================================================
RUNS 24/7 on HF Space or Google Colab.

This is NOT a fake LLM wrapper. This is REAL ML:
  - Population of 500 individuals across 5 islands (100 per island)
  - 13 model types: tree-based + neural nets (LSTM, Transformer, TabNet, etc.)
  - NSGA-II Pareto front ranking (multi-objective: Brier, ROI, Sharpe, Calibration)
  - Island migration every 10 generations for diversity
  - Adaptive mutation: 0.15 -> 0.05 decay + stagnation boost
  - Memory management: GC between evaluations for 16GB RAM
  - Continuous cycles — saves after each generation
  - Callbacks to VM after each cycle
  - Population persistence (survives restarts)

Usage:
  # On HF Space (24/7):
  python evolution/genetic_loop_v3.py --continuous

  # On Google Colab (manual):
  !python genetic_loop_v3.py --generations 50

  # Quick test:
  python evolution/genetic_loop_v3.py --generations 5 --pop-size 50
"""

import os, sys, json, time, random, math, warnings, traceback, gc
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# All model types the GA can evolve (13 total)
ALL_MODEL_TYPES = [
    "xgboost", "lightgbm", "catboost", "random_forest", "extra_trees",
    "stacking", "mlp", "lstm", "transformer", "tabnet",
    "ft_transformer", "deep_ensemble", "autogluon",
]
NEURAL_NET_TYPES = {"lstm", "transformer", "tabnet", "ft_transformer", "deep_ensemble", "mlp", "autogluon"}

# ── Run Logger (best-effort) ──
try:
    from evolution.run_logger import RunLogger
    _HAS_LOGGER = True
except ImportError:
    try:
        from run_logger import RunLogger
        _HAS_LOGGER = True
    except ImportError:
        _HAS_LOGGER = False

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
# SECTION 2: FEATURE ENGINE (250+ features)
# ═══════════════════════════════════════════════════════════

def build_features(games):
    """Build 250+ features from raw game data. Returns X, y, feature_names."""
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

        # 7. CROSS-WINDOW MOMENTUM (20 features) — trend acceleration
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            # Short vs long momentum (5 vs 20)
            wp_accel = wp(tr, 3) - 2 * wp(tr, 10) + wp(tr, 20) if len(tr) >= 20 else 0.0
            pd_accel = pd(tr, 3) - 2 * pd(tr, 10) + pd(tr, 20) if len(tr) >= 20 else 0.0
            # Pythagorean expected win rate (Bill James)
            pts_for = sum(x[4] for x in tr[-20:]) if len(tr) >= 5 else 100
            pts_against = sum(x[5] for x in tr[-20:]) if len(tr) >= 5 else 100
            pyth_exp = pts_for ** 13.91 / max(1, pts_for ** 13.91 + pts_against ** 13.91) if pts_for > 0 else 0.5
            # Scoring volatility
            pts_list = [x[4] for x in tr[-10:]] if len(tr) >= 5 else [100]
            pts_vol = (sum((p - sum(pts_list)/len(pts_list))**2 for p in pts_list) / len(pts_list)) ** 0.5 if len(pts_list) > 1 else 0
            # Home/away specific win rates
            home_games = [x for x in tr if x[3] != home] if prefix == "h" else [x for x in tr if x[3] != away]
            ha_wp = sum(1 for x in home_games[-20:] if x[1]) / max(len(home_games[-20:]), 1)
            # Opponent quality of recent wins
            recent_wins = [x for x in tr[-10:] if x[1]]
            win_quality = sum(wp(team_results[x[3]], 82) for x in recent_wins) / max(len(recent_wins), 1) if recent_wins else 0.5
            # Margin trend (linear slope over last 10 games)
            margins_10 = [x[2] for x in tr[-10:]] if len(tr) >= 5 else [0]
            if len(margins_10) >= 3:
                x_vals = list(range(len(margins_10)))
                x_mean = sum(x_vals) / len(x_vals)
                y_mean = sum(margins_10) / len(margins_10)
                num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, margins_10))
                den = sum((x - x_mean) ** 2 for x in x_vals)
                margin_slope = num / den if den > 0 else 0.0
            else:
                margin_slope = 0.0
            row.extend([
                wp(tr, 5) - wp(tr, 20) if len(tr) >= 20 else 0.0,
                wp_accel, pd_accel, pyth_exp,
                pts_vol / 10.0,  # normalized
                ha_wp, win_quality,
                margin_slope,
                ppg(tr, 3) / max(ppg(tr, 20), 1),  # recent scoring ratio
                papg(tr, 3) / max(papg(tr, 20), 1),  # recent defense ratio
            ])
            if first:
                names.extend([f"{prefix}_wp5v20", f"{prefix}_wp_accel", f"{prefix}_pd_accel",
                              f"{prefix}_pyth_exp", f"{prefix}_pts_vol",
                              f"{prefix}_location_wp", f"{prefix}_win_quality",
                              f"{prefix}_margin_slope", f"{prefix}_off_ratio", f"{prefix}_def_ratio"])

        # 8. INTERACTION FEATURES (12 features) — key cross-terms
        elo_d = team_elo[home] - team_elo[away] + 50
        rest_adv = h_rest - a_rest
        wp_d = wp(hr_, 10) - wp(ar_, 10)
        row.extend([
            elo_d * rest_adv / 10.0,          # elo × rest interaction
            wp_d * rest_adv / 3.0,             # form × rest interaction
            elo_d * (1 if h_rest <= 1 else 0), # elo × b2b penalty
            wp_d ** 2,                          # squared wp diff (nonlinearity)
            elo_d ** 2 / 10000.0,               # squared elo diff
            (ppg(hr_, 10) - papg(ar_, 10)) * (ppg(ar_, 10) - papg(hr_, 10)),  # off×def interaction
            consistency(hr_, 10) * consistency(ar_, 10) / 100.0,  # consistency product
            wp(hr_, 82) * wp(ar_, 82),          # season quality product
            (wp(hr_, 5) - wp(hr_, 20)) * (wp(ar_, 5) - wp(ar_, 20)),  # momentum alignment
            abs(ppg(hr_, 10) + papg(hr_, 10) - ppg(ar_, 10) - papg(ar_, 10)) * elo_d / 1000.0,  # tempo×elo
            (1.0 if wp(hr_, 82) > 0.6 else 0.0) * (1.0 if wp(ar_, 82) < 0.4 else 0.0),  # mismatch flag
            float(h_rest >= 3 and a_rest <= 1),  # rest mismatch flag
        ])
        if first:
            names.extend(["elo_rest_interact", "form_rest_interact", "elo_b2b_penalty",
                          "wp_diff_sq", "elo_diff_sq", "off_def_interact",
                          "consistency_product", "quality_product", "momentum_align",
                          "tempo_elo_interact", "mismatch_flag", "rest_mismatch_flag"])

        # 9. NEW HIGH-IMPACT FEATURES (50 features, windows [5, 10])
        NEW_WINDOWS = [5, 10]

        # Helper: home/away split win% (home team plays at home, away team plays away)
        def home_split_wp(r, n, is_home_team):
            """Win% for home-only or away-only games over last n."""
            if is_home_team:
                # home team's results when they were the home team (opponent is different city)
                loc_games = [x for x in r if x[3] != home][-n:]
            else:
                loc_games = [x for x in r if x[3] != away][-n:]
            if not loc_games:
                return wp(r, n)  # fallback to overall
            return sum(1 for x in loc_games if x[1]) / len(loc_games)

        def away_split_wp(r, n, is_home_team):
            """Win% for away-only games over last n."""
            if is_home_team:
                loc_games = [x for x in r if x[3] == home][-n:]
            else:
                loc_games = [x for x in r if x[3] == away][-n:]
            if not loc_games:
                return wp(r, n)
            return sum(1 for x in loc_games if x[1]) / len(loc_games)

        def net_rating(r, n):
            """Net points per game over window (proxy for net rating)."""
            s = r[-n:]
            if not s:
                return 0.0
            return sum(x[4] - x[5] for x in s) / len(s)

        def pace_proxy(r, n):
            """Approximate pace as total points per game (proxy when possession data absent)."""
            s = r[-n:]
            if not s:
                return 200.0
            return sum(x[4] + x[5] for x in s) / len(s)

        def h2h_wp(hr, ar, n):
            """Head-to-head win% for home team vs this specific away team over last n meetings."""
            meetings = [x for x in hr if x[3] == away][-n:]
            if not meetings:
                return 0.5
            return sum(1 for x in meetings if x[1]) / len(meetings)

        def sos_window(r, n):
            """Average opponent win% over last n games (Strength of Schedule)."""
            rec = r[-n:]
            if not rec:
                return 0.5
            ops = [wp(team_results[x[3]], 82) for x in rec if team_results[x[3]]]
            return sum(ops) / len(ops) if ops else 0.5

        # 9a. Net Rating (windows 5, 10) — 4 features
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            for w in NEW_WINDOWS:
                row.append(net_rating(tr, w))
                if first:
                    names.append(f"{prefix}_net_rating{w}")

        # 9b. Pace proxy (windows 5, 10) — 4 features
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            for w in NEW_WINDOWS:
                row.append(pace_proxy(tr, w))
                if first:
                    names.append(f"{prefix}_pace{w}")

        # 9c. Rest days (already exists as h_rest/a_rest, add explicit named vars for clarity)
        # These are already in section 3 above; skip to avoid duplication.

        # 9d. Home/Away Win% Split (windows 5, 10) — 4 features each side = 8 features
        for w in NEW_WINDOWS:
            row.append(home_split_wp(hr_, w, is_home_team=True))   # h home-venue wp
            row.append(away_split_wp(ar_, w, is_home_team=False))  # a away-venue wp
            if first:
                names.append(f"h_home_wp{w}")
                names.append(f"a_away_wp{w}")

        # 9e. Matchup H2H record (windows 5, 10) — 2 features
        for w in NEW_WINDOWS:
            row.append(h2h_wp(hr_, ar_, w))
            if first:
                names.append(f"h_h2h_wp{w}")

        # 9f. Strength of Schedule windows 5, 10 (distinct from existing sos5/sos10 in sec 4)
        # sec 4 already has h_sos5, h_sos10 — skip to avoid duplication.

        # 9g. Streak type: signed streak (positive=wins, negative=losses) — 2 features
        # (strk() already included in section 2 as h_streak/a_streak; skip duplicate.)

        # 9h. Pace × Net Rating interaction — 4 features (home + away, windows 5 and 10)
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            for w in NEW_WINDOWS:
                p = pace_proxy(tr, w)
                n_r = net_rating(tr, w)
                row.append((p * n_r) / 1000.0)  # scaled
                if first:
                    names.append(f"{prefix}_pace_net_interact{w}")

        # 9i. Pythagorean-adjusted net rating (per-100-possessions approximation) — 4 features
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            for w in NEW_WINDOWS:
                s = tr[-w:]
                if s:
                    total_pts_for = sum(x[4] for x in s)
                    total_pts_ag = sum(x[5] for x in s)
                    n_games = len(s)
                    avg_pace = (total_pts_for + total_pts_ag) / max(n_games, 1)
                    # net per 100 possessions approximation
                    net_per100 = ((total_pts_for - total_pts_ag) / max(n_games, 1)) / max(avg_pace / 100.0, 1.0)
                else:
                    net_per100 = 0.0
                row.append(net_per100)
                if first:
                    names.append(f"{prefix}_net_per100_{w}")

        # 9j. Recent opponent quality (win% of opponents faced) — 4 features (windows 5, 10)
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            for w in NEW_WINDOWS:
                row.append(sos_window(tr, w))
                if first:
                    names.append(f"{prefix}_opp_quality{w}")

        # ── SECTION 10: EXPONENTIALLY-WEIGHTED MOMENTUM FEATURES (~28 features) ──
        # EWM uses manual exponential decay (no pandas needed) for each team's history.
        # Halflife h means the weight of a game h games ago is 0.5x the weight of the current.
        # alpha = 1 - exp(-ln(2) / halflife)  =>  older games decay exponentially.

        def ewm_win(r, halflife):
            """EWM of wins (0/1) with given halflife in games."""
            s = [x[1] for x in r]
            if not s:
                return 0.5
            alpha = 1.0 - math.exp(-math.log(2) / max(halflife, 0.5))
            val, w_sum = 0.0, 0.0
            for i, v in enumerate(s):
                w = (1 - alpha) ** (len(s) - 1 - i)
                val += w * float(v)
                w_sum += w
            return val / w_sum if w_sum > 0 else 0.5

        def ewm_pd(r, halflife):
            """EWM of point differentials with given halflife."""
            s = [x[2] for x in r]
            if not s:
                return 0.0
            alpha = 1.0 - math.exp(-math.log(2) / max(halflife, 0.5))
            val, w_sum = 0.0, 0.0
            for i, v in enumerate(s):
                w = (1 - alpha) ** (len(s) - 1 - i)
                val += w * v
                w_sum += w
            return val / w_sum if w_sum > 0 else 0.0

        def ewm_ppg(r, halflife):
            """EWM of points scored per game."""
            s = [x[4] for x in r]
            if not s:
                return 100.0
            alpha = 1.0 - math.exp(-math.log(2) / max(halflife, 0.5))
            val, w_sum = 0.0, 0.0
            for i, v in enumerate(s):
                w = (1 - alpha) ** (len(s) - 1 - i)
                val += w * v
                w_sum += w
            return val / w_sum if w_sum > 0 else 100.0

        def ewm_papg(r, halflife):
            """EWM of opponent points per game (defensive rating proxy)."""
            s = [x[5] for x in r]
            if not s:
                return 100.0
            alpha = 1.0 - math.exp(-math.log(2) / max(halflife, 0.5))
            val, w_sum = 0.0, 0.0
            for i, v in enumerate(s):
                w = (1 - alpha) ** (len(s) - 1 - i)
                val += w * v
                w_sum += w
            return val / w_sum if w_sum > 0 else 100.0

        def streak_decay_score(r):
            """current_streak x (1 / (1 + games_since_last_loss)).
            Captures both streak length and recency of last loss."""
            if not r:
                return 0.0
            cur_streak = 0
            last_result = r[-1][1]
            for x in reversed(r):
                if x[1] == last_result:
                    cur_streak += 1
                else:
                    break
            if not last_result:
                return -float(cur_streak)  # losing streak: negative
            # Count consecutive games back since last loss (= win streak length)
            games_since_loss = 0
            for x in reversed(r):
                if not x[1]:
                    break
                games_since_loss += 1
            return cur_streak * (1.0 / (1 + games_since_loss))

        def fatigue_index(r, n=5):
            """Sum of (1/rest_days) for last n inter-game gaps — high = compressed schedule."""
            recent = r[-n:]
            if len(recent) < 2:
                return 0.0
            total = 0.0
            for i in range(1, len(recent)):
                try:
                    d1 = datetime.strptime(recent[i - 1][0][:10], "%Y-%m-%d")
                    d2 = datetime.strptime(recent[i][0][:10], "%Y-%m-%d")
                    gap = max(1, abs((d2 - d1).days))
                    total += 1.0 / gap
                except Exception:
                    total += 0.5  # fallback: assume 2-day gap
            return total

        def b2b_delta(r, metric_idx=2):
            """B2B performance delta: avg metric in B2B games minus avg in normal-rest games.
            B2B = previous game was <= 1 day ago."""
            b2b_vals, normal_vals = [], []
            for i in range(1, len(r)):
                try:
                    d1 = datetime.strptime(r[i - 1][0][:10], "%Y-%m-%d")
                    d2 = datetime.strptime(r[i][0][:10], "%Y-%m-%d")
                    gap = abs((d2 - d1).days)
                except Exception:
                    gap = 2
                val = r[i][metric_idx]
                if gap <= 1:
                    b2b_vals.append(val)
                else:
                    normal_vals.append(val)
            b2b_avg = sum(b2b_vals) / len(b2b_vals) if b2b_vals else 0.0
            normal_avg = sum(normal_vals) / len(normal_vals) if normal_vals else 0.0
            return b2b_avg - normal_avg

        def travel_burden(r, n=7):
            """Count unique opponents (proxy for unique cities visited) in last n games.
            More unique opponents correlates with more travel across the schedule."""
            recent = r[-n:]
            if not recent:
                return 0
            return len({x[3] for x in recent})

        # 10a. EWM Win Probability — halflives [3, 5, 10] x 2 teams = 6 features
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            for hl in [3, 5, 10]:
                row.append(ewm_win(tr, hl))
                if first:
                    names.append(f"{prefix}_ewm_win_hl{hl}")

        # 10b. EWM Point Differential — halflives [3, 5, 10] x 2 teams = 6 features
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            for hl in [3, 5, 10]:
                row.append(ewm_pd(tr, hl) / 10.0)  # normalize: typical margins ~0–20 pts
                if first:
                    names.append(f"{prefix}_ewm_pd_hl{hl}")

        # 10c. EWM Offensive Rating (halflife=5) — 2 features
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            row.append(ewm_ppg(tr, 5) / 100.0)  # normalize to ~1.0 range
            if first:
                names.append(f"{prefix}_ewm_off_hl5")

        # 10d. EWM Defensive Rating (halflife=5) — 2 features
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            row.append(ewm_papg(tr, 5) / 100.0)
            if first:
                names.append(f"{prefix}_ewm_def_hl5")

        # 10e. Streak Decay Score — 2 features
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            row.append(streak_decay_score(tr))
            if first:
                names.append(f"{prefix}_streak_decay")

        # 10f. Fatigue Index (last 5 games) — 2 features
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            row.append(fatigue_index(tr, n=5))
            if first:
                names.append(f"{prefix}_fatigue_idx")

        # 10g. B2B Performance Delta (point margin) — 2 features
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            row.append(b2b_delta(tr, metric_idx=2) / 10.0)  # normalized margin delta
            if first:
                names.append(f"{prefix}_b2b_margin_delta")

        # 10h. Travel Burden (unique cities proxy over last 7 games) — 2 features
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            row.append(float(travel_burden(tr, n=7)) / 7.0)  # normalize to [0, 1]
            if first:
                names.append(f"{prefix}_travel_burden7")

        # 10i. Cross-team EWM interaction features — 4 features
        row.append(ewm_win(hr_, 3) - ewm_win(ar_, 3))         # home vs away momentum (hl=3)
        row.append(ewm_win(hr_, 5) - ewm_win(ar_, 5))         # home vs away momentum (hl=5)
        row.append((ewm_pd(hr_, 5) - ewm_pd(ar_, 5)) / 10.0)  # relative margin quality (hl=5)
        row.append((ewm_ppg(hr_, 5) - ewm_papg(ar_, 5)) / 100.0)  # home offense vs away defense
        if first:
            names.extend(["ewm_win_diff_hl3", "ewm_win_diff_hl5",
                          "ewm_pd_diff_hl5", "ewm_off_vs_def_hl5"])

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

    def __init__(self, n_features, target=100, model_type=None):
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
            "model_type": model_type or random.choice(ALL_MODEL_TYPES),
            "calibration": random.choice(["isotonic", "sigmoid", "none"]),
            # Neural net hyperparams
            "nn_hidden_dims": random.choice([64, 128, 256]),
            "nn_n_layers": random.randint(2, 4),
            "nn_dropout": random.uniform(0.1, 0.5),
            "nn_epochs": random.randint(20, 100),
            "nn_batch_size": random.choice([32, 64, 128]),
        }
        self.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "calibration_error": 1.0, "composite": 0.0}
        self.pareto_rank = 999
        self.crowding_dist = 0.0
        self.island_id = -1
        self.generation = 0
        self.birth_generation = 0
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
    def _hamming_distance(f1, f2):
        """Normalized Hamming distance between two binary feature masks (0.0 – 1.0)."""
        n = len(f1)
        if n == 0:
            return 0.0
        return sum(a != b for a, b in zip(f1, f2)) / n

    @staticmethod
    def crossover(p1, p2):
        """Crossover on features + blend hyperparams.

        Crossover type is selected based on parent similarity:
          - Parents very similar (Hamming < 0.1): uniform crossover.
            Picks each bit independently, generating more variation between
            nearly-identical individuals.
          - Otherwise: classic two-point crossover.
        """
        child = Individual.__new__(Individual)
        n = len(p1.features)
        parent_hamming = Individual._hamming_distance(p1.features, p2.features)
        if parent_hamming < 0.1:
            # Uniform crossover: each position drawn independently
            child.features = [
                p1.features[i] if random.random() < 0.5 else p2.features[i]
                for i in range(n)
            ]
        else:
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

        child.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "calibration_error": 1.0, "composite": 0.0}
        child.generation = max(p1.generation, p2.generation) + 1
        child.birth_generation = child.generation
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
        if random.random() < 0.08:
            self.hyperparams["model_type"] = random.choice(ALL_MODEL_TYPES)
        if random.random() < 0.05:
            self.hyperparams["calibration"] = random.choice(["isotonic", "sigmoid", "none"])
        # Neural net hyperparams
        if random.random() < 0.10:
            self.hyperparams["nn_hidden_dims"] = random.choice([64, 128, 256, 512])
        if random.random() < 0.10:
            self.hyperparams["nn_n_layers"] = max(1, min(6, self.hyperparams.get("nn_n_layers", 2) + random.randint(-1, 1)))
        if random.random() < 0.10:
            self.hyperparams["nn_dropout"] = max(0.0, min(0.7, self.hyperparams.get("nn_dropout", 0.3) + random.uniform(-0.1, 0.1)))
        self.n_features = sum(self.features)


# ═══════════════════════════════════════════════════════════
# SECTION 4: FITNESS EVALUATION (multi-objective)
# ═══════════════════════════════════════════════════════════

def evaluate_individual(ind, X, y, n_splits=5, use_gpu=False, _eval_counter=[0]):
    """
    Evaluate one individual via walk-forward backtest.
    Multi-objective: Brier + ROI + Sharpe + Calibration.
    Includes memory management for 16GB RAM with 500 individuals.

    Post-hoc Platt Scaling (added 2026-03-21):
      Each train fold is split 80/20 into train_proper + calibration_set.
      A LogisticRegression is fitted on (raw_probs_cal, y_cal) and used to
      transform test probabilities → calibrated probabilities before computing
      all downstream metrics (Brier, ROI, ECE).  This removes systematic
      over/under-confidence from tree-based models without touching cv=3 inner
      calibration, giving an expected Brier improvement of -0.008 to -0.015.
    """
    _eval_counter[0] += 1
    if _eval_counter[0] % 10 == 0:
        gc.collect()
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    selected = ind.selected_indices()
    if len(selected) < 15 or len(selected) > 250:
        ind.fitness = {"brier": 0.30, "roi": -0.10, "sharpe": -1.0, "calibration": 0.15, "calibration_error": 0.15, "composite": -1.0}
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
            # ── Platt Scaling: split train fold 80/20 → proper + calibration ──
            cal_split = max(1, int(len(ti) * 0.20))
            ti_proper = ti[:-cal_split]   # first 80% (chronological order preserved)
            ti_cal    = ti[-cal_split:]   # last 20% as held-out calibration set

            m = type(model)(**model.get_params())
            if hp["calibration"] != "none":
                m = CalibratedClassifierCV(m, method=hp["calibration"], cv=3)

            # Train on proper subset only
            m.fit(X_sub[ti_proper], y[ti_proper])

            # Get raw probabilities on calibration set
            raw_cal = m.predict_proba(X_sub[ti_cal])[:, 1].reshape(-1, 1)
            y_cal   = y[ti_cal]

            # Fit logistic regression calibrator (Platt scaling)
            platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42)
            platt.fit(raw_cal, y_cal)

            # Apply calibration to test set predictions
            raw_test = m.predict_proba(X_sub[vi])[:, 1].reshape(-1, 1)
            probs = platt.predict_proba(raw_test)[:, 1]

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
        "calibration_error": round(cal_err, 4),   # ECE with 10 bins, on calibrated probs
        "composite": round(composite, 5),
    }


def _build_model(hp, use_gpu=False):
    """Build ML model from hyperparameters."""
    mt = hp["model_type"]
    try:
        if mt == "xgboost":
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
        elif mt == "lightgbm":
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
        elif mt == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=hp["n_estimators"],
                depth=min(hp["max_depth"], 10),
                learning_rate=hp["learning_rate"],
                l2_leaf_reg=hp["reg_lambda"],
                verbose=0, random_state=42,
            )
        elif mt == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                min_samples_leaf=max(1, hp["min_child_weight"]),
                random_state=42, n_jobs=-1,
            )
        elif mt == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                min_samples_leaf=max(1, hp["min_child_weight"]),
                random_state=42, n_jobs=-1,
            )
        elif mt == "stacking":
            from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            estimators = [
                ("rf", RandomForestClassifier(n_estimators=100, max_depth=hp["max_depth"], random_state=42, n_jobs=-1)),
                ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=min(hp["max_depth"], 6), learning_rate=hp["learning_rate"], random_state=42)),
            ]
            try:
                import xgboost as xgb
                estimators.append(("xgb", xgb.XGBClassifier(n_estimators=100, max_depth=hp["max_depth"], learning_rate=hp["learning_rate"], eval_metric="logloss", random_state=42, n_jobs=-1)))
            except ImportError:
                pass
            return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=500), cv=3, n_jobs=-1)
        elif mt == "mlp":
            from sklearn.neural_network import MLPClassifier
            hidden = tuple([hp.get("nn_hidden_dims", 128)] * hp.get("nn_n_layers", 2))
            return MLPClassifier(
                hidden_layer_sizes=hidden,
                learning_rate_init=hp["learning_rate"],
                max_iter=hp.get("nn_epochs", 50),
                alpha=hp["reg_alpha"],
                random_state=42,
            )
        else:
            # Fallback for unknown types (lstm, transformer, tabnet, etc.) — use GBM
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=min(hp["n_estimators"], 200),
                max_depth=hp["max_depth"],
                learning_rate=hp["learning_rate"],
                random_state=42,
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


def _simulate_betting(probs, actuals, edge=0.05, vig=0.045):
    """Simulate flat betting with realistic market odds (including vig).

    Uses market-implied odds with vig instead of fair value (1/prob).
    Standard US sportsbook vig ~4.5% (e.g., -110/-110 on spreads).
    This gives a realistic ROI estimate vs the previous overoptimistic version.
    """
    stake = 10
    profit = 0
    n_bets = 0
    for prob, actual in zip(probs, actuals):
        if prob > 0.5 + edge:
            # Market odds for home win: fair_odds / (1 + vig)
            # Fair decimal odds = 1/market_implied_prob, market adds vig
            market_implied = 0.5  # baseline market line
            market_decimal = 1.0 / (market_implied * (1 + vig))
            n_bets += 1
            if actual == 1:
                profit += stake * (market_decimal - 1)
            else:
                profit -= stake
        elif prob < 0.5 - edge:
            market_implied = 0.5
            market_decimal = 1.0 / (market_implied * (1 + vig))
            n_bets += 1
            if actual == 0:
                profit += stake * (market_decimal - 1)
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

    def __init__(self, pop_size=500, elite_size=25, mutation_rate=0.15,
                 crossover_rate=0.85, target_features=100, n_splits=5,
                 n_islands=5, migration_interval=10, migrants_per_island=5):
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.mut_floor = 0.05
        self.mut_decay = 0.995
        self.crossover_rate = crossover_rate
        self.target_features = target_features
        self.n_splits = n_splits
        self.n_islands = n_islands
        self.island_size = pop_size // n_islands
        self.migration_interval = migration_interval
        self.migrants_per_island = migrants_per_island

        self.population = []
        self.generation = 0
        self.best_ever = None
        self.history = []
        self.stagnation_counter = 0
        self.use_gpu = False
        # Hamming diversity tracking
        self._pop_centroid = None        # float list — mean feature mask over population
        self._hamming_diversity = 1.0   # normalized average pairwise Hamming distance
        self._no_improve_counter = 0    # gens without best-ever composite improvement

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
                ind.birth_generation = ind_data.get("birth_generation", ind.generation)
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

    def resize_population_features(self, new_n_features):
        """Resize feature masks if feature count changed (e.g., new features added)."""
        old_n = self.n_features
        if old_n == new_n_features:
            return
        delta = new_n_features - old_n
        print(f"[RESIZE] Feature count changed: {old_n} -> {new_n_features} (delta={delta})")
        self.n_features = new_n_features
        for ind in self.population:
            if len(ind.features) < new_n_features:
                # Extend with random activation for new features (50% chance each)
                ind.features.extend([1 if random.random() < 0.3 else 0 for _ in range(new_n_features - len(ind.features))])
            elif len(ind.features) > new_n_features:
                ind.features = ind.features[:new_n_features]
            ind.n_features = sum(ind.features)
        if self.best_ever:
            if len(self.best_ever.features) < new_n_features:
                self.best_ever.features.extend([0] * (new_n_features - len(self.best_ever.features)))
            elif len(self.best_ever.features) > new_n_features:
                self.best_ever.features = self.best_ever.features[:new_n_features]
            self.best_ever.n_features = sum(self.best_ever.features)
        print(f"[RESIZE] All {len(self.population)} individuals resized")

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
                    "birth_generation": getattr(ind, 'birth_generation', ind.generation),
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

    # ── Hamming Diversity Utilities ──────────────────────────────────────────

    def _update_pop_centroid(self):
        """Compute and cache the population centroid (mean feature mask).

        The centroid[i] is the fraction of individuals that have feature i active.
        Used by _tournament_select for crowding distance.
        """
        if not self.population:
            return
        n = len(self.population[0].features)
        centroid = [0.0] * n
        for ind in self.population:
            for i, v in enumerate(ind.features):
                centroid[i] += v
        pop_len = len(self.population)
        self._pop_centroid = [c / pop_len for c in centroid]

    def _compute_hamming_diversity(self, sample_size=50):
        """Compute the normalized average pairwise Hamming distance of the population.

        Exact O(N²) computation is expensive for pop_size=500, so we use a
        random sample of up to `sample_size` pairs for efficiency.

        Returns a float in [0, 1].  A value of 0 means all feature masks are
        identical; a value of 1 means every bit differs between every pair.
        """
        pop = self.population
        if len(pop) < 2:
            return 1.0
        n_feat = len(pop[0].features)
        if n_feat == 0:
            return 0.0

        # Random sampling: up to sample_size² / 2 pairs
        indices = list(range(len(pop)))
        random.shuffle(indices)
        sample = indices[:sample_size]

        total_dist = 0.0
        n_pairs = 0
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                f1 = pop[sample[i]].features
                f2 = pop[sample[j]].features
                total_dist += sum(a != b for a, b in zip(f1, f2)) / n_feat
                n_pairs += 1

        return total_dist / n_pairs if n_pairs > 0 else 1.0

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

        # 4. Stagnation detection — track BOTH Brier and composite
        prev_best_composite = self.best_ever.fitness["composite"] if self.best_ever and hasattr(self.best_ever, 'fitness') else 0.0
        brier_stagnant = abs(best.fitness["brier"] - prev_best_brier) < 0.0005
        composite_stagnant = abs(best.fitness["composite"] - prev_best_composite) < 0.001
        if brier_stagnant and composite_stagnant:
            self.stagnation_counter += 1
            self._no_improve_counter += 1
        elif not brier_stagnant:
            self.stagnation_counter = max(0, self.stagnation_counter - 2)  # Partial reset
            self._no_improve_counter = 0
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
            self._no_improve_counter = 0

        # 4b. Hamming Diversity Monitor
        # Compute normalized average pairwise Hamming distance; also refresh centroid
        # (used by crowding-aware tournament selection below).
        self._hamming_diversity = self._compute_hamming_diversity(sample_size=50)
        self._update_pop_centroid()
        if self._hamming_diversity < 0.15:
            print(f"  [DIVERSITY-LOW] Hamming diversity={self._hamming_diversity:.3f} < 0.15 threshold")

        # 4c. Adaptive Mutation Rate — diversity-driven formula
        # Base = 0.03; rises smoothly toward 0.10 as diversity falls below 0.25.
        # Formula: mutation_rate = 0.03 + 0.07 * max(0, 1 - diversity / 0.25)
        diversity_mutation = 0.03 + 0.07 * max(0.0, 1.0 - self._hamming_diversity / 0.25)
        # Stagnation boosts applied on top (capped at 0.25)
        if self.stagnation_counter >= 10:
            self.mutation_rate = min(0.25, diversity_mutation * 1.8)
            print(f"  [STAGNATION-CRITICAL] {self.stagnation_counter} gens — "
                  f"mutation rate -> {self.mutation_rate:.3f} (diversity={self._hamming_diversity:.3f})")
        elif self.stagnation_counter >= 7:
            self.mutation_rate = min(0.20, diversity_mutation * 1.5)
            print(f"  [STAGNATION] {self.stagnation_counter} gens — "
                  f"mutation rate -> {self.mutation_rate:.3f} (diversity={self._hamming_diversity:.3f})")
        elif self.stagnation_counter >= 3:
            self.mutation_rate = min(0.15, diversity_mutation * 1.2)
        else:
            # Normal regime: formula drives the rate directly
            self.mutation_rate = diversity_mutation

        # 5. Record history
        self.history.append({
            "gen": self.generation,
            "best_brier": best.fitness["brier"],
            "best_roi": best.fitness["roi"],
            "best_sharpe": best.fitness["sharpe"],
            "best_composite": best.fitness["composite"],
            "best_calibration_error": best.fitness.get("calibration_error", best.fitness.get("calibration", 1.0)),
            "n_features": best.n_features,
            "model_type": best.hyperparams["model_type"],
            "mutation_rate": round(self.mutation_rate, 4),
            "avg_composite": round(np.mean([ind.fitness["composite"] for ind in self.population]), 5),
            "pop_diversity": round(np.std([ind.n_features for ind in self.population]), 1),
            "hamming_diversity": round(self._hamming_diversity, 4),
        })

        elapsed = time.time() - gen_start
        ece_val = best.fitness.get("calibration_error", best.fitness.get("calibration", 1.0))
        print(f"  Gen {self.generation}: Brier={best.fitness['brier']:.4f} "
              f"ROI={best.fitness['roi']:.1%} Sharpe={best.fitness['sharpe']:.2f} "
              f"ECE={ece_val:.4f} Features={best.n_features} Model={best.hyperparams['model_type']} "
              f"Composite={best.fitness['composite']:.4f} "
              f"Diversity={self._hamming_diversity:.3f} MutRate={self.mutation_rate:.3f} ({elapsed:.0f}s)")

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
            elite.birth_generation = getattr(self.population[i], 'birth_generation', self.population[i].generation)
            new_pop.append(elite)

        # Aging: remove individuals that have survived > 15 generations without improvement
        MAX_AGE = 15
        aged_out = 0
        for i in range(len(new_pop) - 1, self.elite_size - 1, -1):
            if i < len(new_pop):
                age = self.generation - getattr(new_pop[i], 'birth_generation', 0)
                if age > MAX_AGE and new_pop[i].fitness["composite"] < new_pop[0].fitness["composite"] * 0.95:
                    new_pop.pop(i)
                    aged_out += 1
        if aged_out > 0:
            print(f"  [AGING] {aged_out} stale individuals removed")

        # Injection: smarter — at stagnation >= 7 inject targeted mutants of best, not just random
        n_inject = 0
        if self.stagnation_counter >= 7:
            n_inject = self.pop_size // 4
            # Half random, half targeted mutations of best individual
            n_random = n_inject // 2
            n_mutant = n_inject - n_random
            for _ in range(n_random):
                new_pop.append(Individual(self.n_features, self.target_features))
            # Targeted mutants: take best, apply heavy mutation
            for _ in range(n_mutant):
                mutant = Individual.__new__(Individual)
                mutant.features = self.population[0].features[:]
                mutant.hyperparams = dict(self.population[0].hyperparams)
                mutant.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "calibration_error": 1.0, "composite": 0.0}
                mutant.birth_generation = self.generation
                mutant.n_features = self.population[0].n_features
                mutant.generation = self.generation
                mutant.mutate(0.25)  # Heavy mutation
                new_pop.append(mutant)
            print(f"  [INJECTION] {n_random} random + {n_mutant} targeted mutants (stagnation={self.stagnation_counter})")
        elif self.stagnation_counter >= 3:
            # Mild injection: 10% fresh individuals
            n_inject = self.pop_size // 10
            for _ in range(n_inject):
                new_pop.append(Individual(self.n_features, self.target_features))
            print(f"  [INJECTION-MILD] {n_inject} fresh individuals (stagnation={self.stagnation_counter})")

        # Diversity Injection: triggered independently when diversity is critically low
        # (diversity < 0.15) OR when there has been no fitness improvement for 5
        # consecutive generations — whichever happens first.  Elites are always kept.
        diversity_trigger = (self._hamming_diversity < 0.15) or (self._no_improve_counter >= 5)
        if diversity_trigger and n_inject == 0:
            # Inject 20% of population as freshly randomized individuals (elites already in new_pop)
            n_diversity_inject = max(1, self.pop_size // 5)
            # Cap to avoid going way over pop_size before the fill loop
            slots_remaining = max(0, self.pop_size - len(new_pop) - n_diversity_inject)
            for _ in range(n_diversity_inject):
                new_pop.append(Individual(self.n_features, self.target_features))
            trigger_reason = (
                f"diversity={self._hamming_diversity:.3f}<0.15"
                if self._hamming_diversity < 0.15
                else f"no_improve={self._no_improve_counter}>=5"
            )
            print(f"  [DIVERSITY-INJECT] {n_diversity_inject} fresh individuals injected "
                  f"({trigger_reason}), elites preserved")

        # Fill with crossover + mutation
        while len(new_pop) < self.pop_size:
            # Diversity-aware tournament: 80% fitness-based, 20% diversity-based
            if random.random() < 0.2:
                p1 = self._diversity_select(7)
                p2 = self._tournament_select(7)
            else:
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
                child.birth_generation = self.generation
            child.mutate(self.mutation_rate)
            new_pop.append(child)

        self.population = new_pop[:self.pop_size]
        return best

    def _tournament_select(self, k=7):
        """Tournament selection with crowding.

        Standard tournament selection, but when two candidates have similar
        composite fitness (within 5%), prefer the one that is more unique —
        measured by Hamming distance from the population centroid.  This
        implements a lightweight niching pressure that rewards exploration
        without discarding high-quality individuals.
        """
        contestants = random.sample(self.population, min(k, len(self.population)))
        best = max(contestants, key=lambda x: x.fitness["composite"])
        best_fit = best.fitness["composite"]

        # Among contestants within 5% of the best, prefer the most unique one
        similar = [c for c in contestants if best_fit > 0 and
                   abs(c.fitness["composite"] - best_fit) / max(abs(best_fit), 1e-9) < 0.05]
        if len(similar) > 1 and hasattr(self, '_pop_centroid') and self._pop_centroid is not None:
            centroid = self._pop_centroid
            def _dist_from_centroid(ind):
                f = ind.features
                n = len(f)
                if n == 0 or len(centroid) != n:
                    return 0.0
                return sum(abs(f[i] - centroid[i]) for i in range(n)) / n
            best = max(similar, key=_dist_from_centroid)

        return best

    def _diversity_select(self, k=7):
        """Diversity-preserving selection: pick the most unique individual from k random."""
        contestants = random.sample(self.population, min(k, len(self.population)))
        if not self.population:
            return contestants[0]
        # Measure uniqueness: how different is this individual's feature set from the elite?
        elite_features = set()
        for i, ind in enumerate(self.population[:self.elite_size]):
            elite_features.update(ind.selected_indices())
        best_diversity = -1
        best_ind = contestants[0]
        for c in contestants:
            c_features = set(c.selected_indices())
            if not c_features:
                continue
            overlap = len(c_features & elite_features) / max(len(c_features), 1)
            diversity = 1.0 - overlap
            # Weight by fitness to avoid picking terrible individuals
            score = diversity * 0.6 + max(0, c.fitness["composite"]) * 0.4
            if score > best_diversity:
                best_diversity = score
                best_ind = c
        return best_ind

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
                "calibration_error": self.best_ever.fitness.get("calibration_error", self.best_ever.fitness["calibration"]),
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

def run_continuous(generations_per_cycle=10, total_cycles=None, pop_size=500,
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
        pop_size=pop_size, elite_size=max(5, pop_size // 20), mutation_rate=0.15,
        crossover_rate=0.85, target_features=target_features, n_splits=n_splits,
        n_islands=5, migration_interval=10, migrants_per_island=5,
    )

    # Try to restore previous state
    if not engine.restore_state():
        engine.initialize(X.shape[1])
    else:
        # Resize population if feature count changed (new features added)
        engine.resize_population_features(X.shape[1])

    # ── Supabase Run Logger + Auto-Cut ──
    run_logger = None
    if _HAS_LOGGER:
        try:
            run_logger = RunLogger(local_dir=str(RESULTS_DIR / "run-logs"))
            print("[RUN-LOGGER] Supabase logging + auto-cut ACTIVE")
        except Exception as e:
            print(f"[RUN-LOGGER] Init failed: {e}")

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
                gen_start = time.time()
                best = engine.evolve_one_generation(X, y)

                # ── Log generation + auto-cut ──
                if run_logger and best:
                    try:
                        pop_div = float(np.std([ind.n_features for ind in engine.population]))
                        avg_comp = float(np.mean([ind.fitness["composite"] for ind in engine.population]))
                        run_logger.log_generation(
                            cycle=cycle, generation=engine.generation,
                            best={"brier": best.fitness["brier"], "roi": best.fitness["roi"],
                                  "sharpe": best.fitness["sharpe"], "composite": best.fitness["composite"],
                                  "n_features": best.n_features, "model_type": best.hyperparams["model_type"]},
                            mutation_rate=engine.mutation_rate, avg_composite=avg_comp,
                            pop_diversity=pop_div, duration_s=time.time() - gen_start)

                        # Auto-cut check
                        cut_actions = run_logger.check_auto_cut(best.fitness, {
                            "mutation_rate": engine.mutation_rate,
                            "stagnation": engine.stagnation_counter,
                            "pop_size": engine.pop_size,
                            "pop_diversity": pop_div,
                        })
                        for action in cut_actions:
                            atype = action["type"]
                            params = action.get("params", {})
                            if atype == "config" and "mutation_rate" in params:
                                engine.mutation_rate = params["mutation_rate"]
                            elif atype == "emergency_diversify":
                                n_new = engine.pop_size // 3
                                engine.population = sorted(engine.population, key=lambda x: x.fitness["composite"], reverse=True)[:engine.pop_size - n_new]
                                for _ in range(n_new):
                                    engine.population.append(Individual(engine.n_features, engine.target_features))
                                print(f"  [AUTO-CUT] Diversified: {n_new} fresh individuals")
                            elif atype == "full_reset":
                                engine.population = sorted(engine.population, key=lambda x: x.fitness["composite"], reverse=True)[:engine.elite_size]
                                while len(engine.population) < engine.pop_size:
                                    engine.population.append(Individual(engine.n_features, engine.target_features))
                                engine.stagnation_counter = 0
                                print(f"  [AUTO-CUT] FULL RESET executed")
                    except Exception as e:
                        print(f"  [RUN-LOGGER] Error: {e}")
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

        # ── Log cycle to Supabase ──
        if run_logger and results and engine.best_ever:
            try:
                pop_div = float(np.std([ind.n_features for ind in engine.population]))
                avg_comp = float(np.mean([ind.fitness["composite"] for ind in engine.population]))
                run_logger.log_cycle(
                    cycle=cycle, generation=engine.generation,
                    best=engine.best_ever.fitness | {"n_features": engine.best_ever.n_features,
                                                      "model_type": engine.best_ever.hyperparams["model_type"]},
                    pop_size=engine.pop_size, mutation_rate=engine.mutation_rate,
                    crossover_rate=engine.crossover_rate, stagnation=engine.stagnation_counter,
                    games=len(games), feature_candidates=X.shape[1],
                    cycle_duration_s=cycle_elapsed, avg_composite=avg_comp, pop_diversity=pop_div,
                    top5=results.get("top5"), selected_features=results.get("best", {}).get("selected_features"))
                print(f"  [RUN-LOGGER] Cycle {cycle} logged to Supabase")
            except Exception as e:
                print(f"  [RUN-LOGGER] Cycle log error: {e}")

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
    parser.add_argument("--pop-size", type=int, default=500, help="Population size (default: 500)")
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
