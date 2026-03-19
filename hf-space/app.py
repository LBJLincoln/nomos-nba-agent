#!/usr/bin/env python3
"""
NOMOS NBA QUANT AI — REAL Genetic Evolution (HF Space)
========================================================
RUNS 24/7 on HuggingFace Space (2 vCPU / 16GB RAM).

NOT a fake LLM wrapper. REAL ML:
  - Population of 60 individuals (feature mask + hyperparams)
  - Walk-forward backtest fitness (Brier + LogLoss + Sharpe + ECE)
  - Tournament selection, crossover, adaptive mutation
  - Continuous cycles — saves after each generation
  - Gradio dashboard showing live evolution progress
  - JSON API for crew agents on VM

Target: Brier < 0.20 | ROI > 5% | Sharpe > 1.0
"""

import os, sys, json, time, math, threading, warnings, random, traceback
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

# ── Env ──
for f in [Path(".env"), Path(".env.local")]:
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgbm

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ── Run Logger (Supabase logging + auto-cut) ──
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evolution.run_logger import RunLogger
    _HAS_LOGGER = True
except ImportError:
    _HAS_LOGGER = False
    print("[WARN] run_logger not available — logging disabled")

# ── Paths (use /data for HF Space persistent storage) ──
_persistent = Path("/data")
DATA_DIR = _persistent if _persistent.exists() else Path("data")
HIST_DIR = DATA_DIR / "historical"
RESULTS_DIR = DATA_DIR / "results"
STATE_DIR = DATA_DIR / "evolution-state"
for d in [DATA_DIR, HIST_DIR, RESULTS_DIR, STATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Copy bundled game data to persistent storage if not already there
_repo_hist = Path("data/historical")
if _repo_hist.exists() and DATA_DIR != Path("data"):
    import shutil
    for f in _repo_hist.glob("games-*.json"):
        dest = HIST_DIR / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            print(f"Copied {f.name} to persistent storage")

VM_URL = os.environ.get("VM_CALLBACK_URL", "http://34.136.180.66:8080")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

# ═══════════════════════════════════════════════════════
# LIVE STATE (shared between evolution thread + Gradio)
# ═══════════════════════════════════════════════════════

live = {
    "status": "STARTING",
    "cycle": 0,
    "generation": 0,
    "best_brier": 1.0,
    "best_roi": 0.0,
    "best_sharpe": 0.0,
    "best_features": 0,
    "best_model_type": "none",
    "pop_size": 0,
    "mutation_rate": 0.03,
    "stagnation": 0,
    "games": 0,
    "feature_candidates": 0,
    "gpu": False,
    "log": [],
    "history": [],
    "top5": [],
    "started_at": datetime.now(timezone.utc).isoformat(),
    "last_update": "never",
}


def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = f"[{ts}] [{level}] {msg}"
    print(entry, flush=True)
    live["log"].append(entry)
    if len(live["log"]) > 500:
        live["log"] = live["log"][-500:]
    live["last_update"] = datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════

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
ARENA_ALTITUDE = {
    "DEN": 5280, "UTA": 4226, "PHX": 1086, "OKC": 1201, "SAS": 650,
    "DAL": 430, "HOU": 43, "MEM": 337, "ATL": 1050, "CHA": 751,
    "IND": 715, "CHI": 594, "MIL": 617, "MIN": 830, "DET": 600,
    "CLE": 653, "BOS": 141, "NYK": 33, "BKN": 33, "PHI": 39,
    "WAS": 25, "MIA": 6, "ORL": 82, "NOP": 7, "TOR": 250,
    "POR": 50, "SAC": 30, "GSW": 12, "LAL": 305, "LAC": 305,
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
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def pull_seasons():
    existing = {f.stem.replace("games-", "") for f in HIST_DIR.glob("games-*.json")}
    log(f"Cached seasons: {sorted(existing) if existing else 'none'}")

    # Count total cached games
    total_cached = 0
    for f in HIST_DIR.glob("games-*.json"):
        try:
            total_cached += len(json.loads(f.read_text()))
        except:
            pass
    log(f"Total cached games: {total_cached}")

    # If we have enough data, skip NBA API entirely (it's slow and unreliable)
    if total_cached >= 500:
        log(f"Enough cached data ({total_cached} games). Skipping NBA API pull.")
        return

    try:
        from nba_api.stats.endpoints import leaguegamefinder
    except ImportError:
        log("nba_api not installed — using cached data only", "WARN")
        return

    targets = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    missing = [s for s in targets if s not in existing]
    if not missing:
        log("All seasons cached — skipping NBA API pull")
        return
    log(f"Missing seasons: {missing} — pulling from NBA API (timeout 20s each, max 3)")
    pulled = 0
    for season in missing[:3]:  # Max 3 pulls to avoid long startup
        log(f"Pulling {season}...")
        try:
            time.sleep(2)
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season, league_id_nullable="00",
                season_type_nullable="Regular Season", timeout=20)
            df = finder.get_data_frames()[0]
            if df.empty:
                log(f"  {season}: empty response", "WARN")
                continue
            pairs = {}
            for _, row in df.iterrows():
                gid = row["GAME_ID"]
                if gid not in pairs: pairs[gid] = []
                pairs[gid].append({"team_name": row.get("TEAM_NAME", ""),
                                   "matchup": row.get("MATCHUP", ""),
                                   "pts": int(row["PTS"]) if row.get("PTS") is not None else None,
                                   "game_date": row.get("GAME_DATE", "")})
            games = []
            for gid, teams in pairs.items():
                if len(teams) != 2: continue
                home = next((t for t in teams if " vs. " in str(t.get("matchup", ""))), None)
                away = next((t for t in teams if " @ " in str(t.get("matchup", ""))), None)
                if not home or not away or home["pts"] is None: continue
                games.append({"game_date": home["game_date"],
                              "home_team": home["team_name"], "away_team": away["team_name"],
                              "home": {"team_name": home["team_name"], "pts": home["pts"]},
                              "away": {"team_name": away["team_name"], "pts": away["pts"]}})
            if games:
                (HIST_DIR / f"games-{season}.json").write_text(json.dumps(games))
                log(f"  {len(games)} games for {season}")
                pulled += 1
        except Exception as e:
            log(f"  {season} failed (skipping): {e}", "WARN")
    if pulled == 0 and total_cached < 500:
        log("WARNING: No data available — evolution cannot start", "ERROR")


def load_all_games():
    games = []
    for f in sorted(HIST_DIR.glob("games-*.json")):
        data = json.loads(f.read_text())
        games.extend(data if isinstance(data, list) else data.get("games", []))
    games.sort(key=lambda g: g.get("game_date", g.get("date", "")))
    return games


# ═══════════════════════════════════════════════════════
# FEATURE ENGINE (164 features)
# ═══════════════════════════════════════════════════════

def build_features(games):
    """Build features using the full NBAFeatureEngine (2058 features)."""
    try:
        from features.engine import NBAFeatureEngine
        engine = NBAFeatureEngine()
        X, y, feature_names = engine.build(games)
        X = np.nan_to_num(np.array(X, dtype=np.float64))
        y = np.array(y, dtype=np.int32)
        return X, y, feature_names
    except Exception as e:
        print(f"[WARN] NBAFeatureEngine failed ({e}), falling back to inline features")
    # ── Fallback: inline 213 features (legacy) ──
    team_results = defaultdict(list)
    team_last = {}
    team_elo = defaultdict(lambda: 1500.0)
    X, y, feature_names = [], [], []
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
        if hs is None or as_ is None: continue
        hs, as_ = int(hs), int(as_)
        home, away = resolve(hr), resolve(ar)
        if not home or not away: continue
        gd = game.get("game_date", game.get("date", ""))[:10]
        hr_, ar_ = team_results[home], team_results[away]

        if len(hr_) < 5 or len(ar_) < 5:
            team_results[home].append((gd, hs > as_, hs - as_, away, hs, as_))
            team_results[away].append((gd, as_ > hs, as_ - hs, home, as_, hs))
            team_last[home] = gd; team_last[away] = gd
            K = 20; exp_h = 1 / (1 + 10 ** ((team_elo[away] - team_elo[home] - 50) / 400))
            team_elo[home] += K * ((1 if hs > as_ else 0) - exp_h)
            team_elo[away] += K * ((0 if hs > as_ else 1) - (1 - exp_h))
            continue

        def wp(r, n): s = r[-n:]; return sum(1 for x in s if x[1]) / len(s) if s else 0.5
        def pd(r, n): s = r[-n:]; return sum(x[2] for x in s) / len(s) if s else 0
        def ppg(r, n): s = r[-n:]; return sum(x[4] for x in s) / len(s) if s else 100
        def papg(r, n): s = r[-n:]; return sum(x[5] for x in s) / len(s) if s else 100
        def strk(r):
            if not r: return 0
            s, l = 0, r[-1][1]
            for x in reversed(r):
                if x[1] == l: s += 1
                else: break
            return s if l else -s
        def close_pct(r, n): s = r[-n:]; return sum(1 for x in s if abs(x[2]) <= 5) / len(s) if s else 0.5
        def blowout_pct(r, n): s = r[-n:]; return sum(1 for x in s if abs(x[2]) >= 15) / len(s) if s else 0
        def consistency(r, n):
            s = r[-n:]
            if len(s) < 3: return 0
            m = [x[2] for x in s]; avg = sum(m) / len(m)
            return (sum((v - avg) ** 2 for v in m) / len(m)) ** 0.5
        def rest(t):
            last = team_last.get(t)
            if not last or not gd: return 3
            try: return max(0, (datetime.strptime(gd[:10], "%Y-%m-%d") - datetime.strptime(last[:10], "%Y-%m-%d")).days)
            except: return 3
        def sos(r, n=10):
            rec = r[-n:]
            if not rec: return 0.5
            ops = [wp(team_results[x[3]], 82) for x in rec if team_results[x[3]]]
            return sum(ops) / len(ops) if ops else 0.5
        def travel_d(r, team):
            if not r: return 0
            lo = r[-1][3]
            if lo in ARENA_COORDS and team in ARENA_COORDS: return haversine(*ARENA_COORDS[lo], *ARENA_COORDS[team])
            return 0

        h_rest, a_rest = rest(home), rest(away)
        try: dt = datetime.strptime(gd, "%Y-%m-%d"); month, dow = dt.month, dt.weekday()
        except: month, dow = 1, 2
        sp = max(0, min(1, (month - 10) / 7)) if month >= 10 else max(0, min(1, (month + 2) / 7))
        row, names = [], []

        for prefix, tr in [("h", hr_), ("a", ar_)]:
            for w in WINDOWS:
                row.extend([wp(tr, w), pd(tr, w), ppg(tr, w), papg(tr, w),
                            ppg(tr, w) - papg(tr, w), close_pct(tr, w), blowout_pct(tr, w), ppg(tr, w) + papg(tr, w)])
                if first:
                    names.extend([f"{prefix}_wp{w}", f"{prefix}_pd{w}", f"{prefix}_ppg{w}", f"{prefix}_papg{w}",
                                  f"{prefix}_margin{w}", f"{prefix}_close{w}", f"{prefix}_blowout{w}", f"{prefix}_ou{w}"])

        for prefix, tr in [("h", hr_), ("a", ar_)]:
            row.extend([strk(tr), abs(strk(tr)), wp(tr, 5) - wp(tr, 82), wp(tr, 3) - wp(tr, 10),
                        ppg(tr, 5) - ppg(tr, 20), papg(tr, 5) - papg(tr, 20), consistency(tr, 10), consistency(tr, 5)])
            if first:
                names.extend([f"{prefix}_streak", f"{prefix}_streak_abs", f"{prefix}_form5v82", f"{prefix}_form3v10",
                              f"{prefix}_scoring_trend", f"{prefix}_defense_trend", f"{prefix}_consistency10", f"{prefix}_consistency5"])

        ht, at = travel_d(hr_, home), travel_d(ar_, away)
        row.extend([min(h_rest, 7), min(a_rest, 7), h_rest - a_rest, float(h_rest <= 1), float(a_rest <= 1),
                     ht / 1000, at / 1000, (ht - at) / 1000,
                     ARENA_ALTITUDE.get(home, 500) / 5280, ARENA_ALTITUDE.get(away, 500) / 5280,
                     (ARENA_ALTITUDE.get(home, 500) - ARENA_ALTITUDE.get(away, 500)) / 5280,
                     abs(TIMEZONE_ET.get(home, 0) - TIMEZONE_ET.get(away, 0)), 0, 0, 0, 0])
        if first:
            names.extend(["h_rest", "a_rest", "rest_adv", "h_b2b", "a_b2b", "h_travel", "a_travel", "travel_adv",
                          "h_alt", "a_alt", "alt_delta", "tz_shift", "h_g7d", "a_g7d", "sched_dens", "pad1"])

        for prefix, tr in [("h", hr_), ("a", ar_)]:
            s5, s10, ss = sos(tr, 5), sos(tr, 10), sos(tr, 82)
            wa = sum(1 for r in tr if wp(team_results[r[3]], 82) > 0.5 and r[1]) / max(sum(1 for r in tr if wp(team_results[r[3]], 82) > 0.5), 1)
            wb = sum(1 for r in tr if wp(team_results[r[3]], 82) <= 0.5 and r[1]) / max(sum(1 for r in tr if wp(team_results[r[3]], 82) <= 0.5), 1)
            row.extend([s5, s10, ss, wa, wb, 0])
            if first:
                names.extend([f"{prefix}_sos5", f"{prefix}_sos10", f"{prefix}_sos_season", f"{prefix}_wp_above500", f"{prefix}_wp_below500", f"{prefix}_mq"])

        row.extend([wp(hr_, 10) - wp(ar_, 10), pd(hr_, 10) - pd(ar_, 10), ppg(hr_, 10) - papg(ar_, 10),
                     ppg(ar_, 10) - papg(hr_, 10), abs(ppg(hr_, 10) + papg(hr_, 10) - ppg(ar_, 10) - papg(ar_, 10)),
                     consistency(hr_, 10) - consistency(ar_, 10),
                     team_elo[home], team_elo[away], team_elo[home] - team_elo[away] + 50,
                     (team_elo[home] - 1500) / 100, (team_elo[away] - 1500) / 100, (team_elo[home] - team_elo[away]) / 100])
        if first:
            names.extend(["rel_str", "rel_pd", "off_match", "def_match", "tempo_diff", "cons_edge",
                          "elo_h", "elo_a", "elo_diff", "elo_h_n", "elo_a_n", "elo_d_n"])

        h_gp, a_gp = min(len(hr_), 82) / 82.0, min(len(ar_), 82) / 82.0
        h_wp82, a_wp82 = wp(hr_, 82), wp(ar_, 82)
        row.extend([1.0, sp, math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12),
                     dow / 6.0, float(dow >= 5), h_gp, a_gp,
                     h_wp82 + a_wp82, h_wp82 - a_wp82,
                     float(h_wp82 > 0.5 and a_wp82 > 0.5), ppg(hr_, 10) + ppg(ar_, 10)])
        if first:
            names.extend(["home_ct", "season_ph", "month_sin", "month_cos", "dow", "weekend",
                          "h_gp", "a_gp", "comb_wp", "wp_diff", "playoff_race", "exp_total"])

        # ── ADDITIONAL FEATURES (v2 — tanking, season context, advanced) ──

        # Tanking detection: teams with <35% WP after ASB are likely tanking
        h_tanking = float(h_wp82 < 0.35 and h_gp > 0.5)
        a_tanking = float(a_wp82 < 0.35 and a_gp > 0.5)
        # Playoff motivation: teams between 40-60% WP fighting for spots
        h_playoff_fight = float(0.40 <= h_wp82 <= 0.60 and h_gp > 0.6)
        a_playoff_fight = float(0.40 <= a_wp82 <= 0.60 and a_gp > 0.6)
        # Elite team indicator (>65% WP and coasting?)
        h_elite = float(h_wp82 > 0.65)
        a_elite = float(a_wp82 > 0.65)
        # Season trajectory: recent form vs overall (improvement/decline)
        h_trajectory = wp(hr_, 10) - h_wp82
        a_trajectory = wp(ar_, 10) - a_wp82
        # Scoring volatility
        h_vol = consistency(hr_, 15) if len(hr_) >= 15 else consistency(hr_, 10)
        a_vol = consistency(ar_, 15) if len(ar_) >= 15 else consistency(ar_, 10)
        # Defensive rating proxy
        h_drtg = papg(hr_, 10)
        a_drtg = papg(ar_, 10)
        h_ortg = ppg(hr_, 10)
        a_ortg = ppg(ar_, 10)
        h_netrtg = h_ortg - h_drtg
        a_netrtg = a_ortg - a_drtg
        # Pace proxy (total points)
        h_pace = (ppg(hr_, 10) + papg(hr_, 10)) / 2
        a_pace = (ppg(ar_, 10) + papg(ar_, 10)) / 2
        pace_diff = h_pace - a_pace
        # Close game performance (clutch proxy)
        h_close_wp = sum(1 for x in hr_[-20:] if abs(x[2]) <= 5 and x[1]) / max(sum(1 for x in hr_[-20:] if abs(x[2]) <= 5), 1)
        a_close_wp = sum(1 for x in ar_[-20:] if abs(x[2]) <= 5 and x[1]) / max(sum(1 for x in ar_[-20:] if abs(x[2]) <= 5), 1)
        # Blowout tendency
        h_blowout_rate = blowout_pct(hr_, 20)
        a_blowout_rate = blowout_pct(ar_, 20)
        # Home/away specific performance
        h_home_games = [x for x in hr_ if True]  # All results are at home for home team
        a_away_games = [x for x in ar_ if True]
        h_home_wp = wp(hr_, 82)  # Proxy (would need H/A split)
        a_away_wp = wp(ar_, 82)
        # Strength of schedule recent (last 5 opponents)
        h_recent_sos = sos(hr_, 5)
        a_recent_sos = sos(ar_, 5)
        # Win percentage vs good/bad teams (already have but add differential)
        h_quality_diff = (sum(1 for r in hr_ if wp(team_results[r[3]], 82) > 0.5 and r[1]) / max(sum(1 for r in hr_ if wp(team_results[r[3]], 82) > 0.5), 1)) - (sum(1 for r in hr_ if wp(team_results[r[3]], 82) <= 0.5 and r[1]) / max(sum(1 for r in hr_ if wp(team_results[r[3]], 82) <= 0.5), 1))
        a_quality_diff = (sum(1 for r in ar_ if wp(team_results[r[3]], 82) > 0.5 and r[1]) / max(sum(1 for r in ar_ if wp(team_results[r[3]], 82) > 0.5), 1)) - (sum(1 for r in ar_ if wp(team_results[r[3]], 82) <= 0.5 and r[1]) / max(sum(1 for r in ar_ if wp(team_results[r[3]], 82) <= 0.5), 1))
        # Form acceleration (last 3 vs last 10)
        h_accel = wp(hr_, 3) - wp(hr_, 10)
        a_accel = wp(ar_, 3) - wp(ar_, 10)
        # Elo confidence (distance from 1500)
        h_elo_conf = abs(team_elo[home] - 1500) / 300
        a_elo_conf = abs(team_elo[away] - 1500) / 300
        # Combined metrics
        matchup_quality = (h_wp82 + a_wp82) / 2  # How good is this game?
        competitiveness = 1 - abs(h_wp82 - a_wp82)  # How close are teams?
        upset_potential = max(0, a_wp82 - h_wp82)  # Away team better?

        row.extend([
            h_tanking, a_tanking, h_playoff_fight, a_playoff_fight, h_elite, a_elite,
            h_trajectory, a_trajectory, h_trajectory - a_trajectory,
            h_vol, a_vol, abs(h_vol - a_vol),
            h_drtg, a_drtg, h_ortg, a_ortg, h_netrtg, a_netrtg, h_netrtg - a_netrtg,
            h_pace, a_pace, pace_diff,
            h_close_wp, a_close_wp, h_close_wp - a_close_wp,
            h_blowout_rate, a_blowout_rate,
            h_recent_sos, a_recent_sos, h_recent_sos - a_recent_sos,
            h_quality_diff, a_quality_diff,
            h_accel, a_accel, h_accel - a_accel,
            h_elo_conf, a_elo_conf,
            matchup_quality, competitiveness, upset_potential,
            # Interaction features
            h_wp82 * h_rest, a_wp82 * a_rest,  # quality × rest
            h_netrtg * float(h_rest >= 2), a_netrtg * float(a_rest >= 2),  # net rating when rested
            h_trajectory * h_gp, a_trajectory * a_gp,  # trajectory weighted by games played
            (team_elo[home] - team_elo[away]) * sp,  # Elo diff × season phase
            competitiveness * sp,  # Tighter games matter more late season
            h_close_wp * competitiveness,  # Clutch ability in close matchups
        ])
        if first:
            names.extend([
                "h_tanking", "a_tanking", "h_playoff_fight", "a_playoff_fight", "h_elite", "a_elite",
                "h_trajectory", "a_trajectory", "trajectory_diff",
                "h_volatility", "a_volatility", "volatility_diff",
                "h_drtg", "a_drtg", "h_ortg", "a_ortg", "h_netrtg", "a_netrtg", "netrtg_diff",
                "h_pace", "a_pace", "pace_diff",
                "h_clutch_wp", "a_clutch_wp", "clutch_diff",
                "h_blowout_rate", "a_blowout_rate",
                "h_recent_sos", "a_recent_sos", "recent_sos_diff",
                "h_quality_diff", "a_quality_diff",
                "h_form_accel", "a_form_accel", "accel_diff",
                "h_elo_confidence", "a_elo_confidence",
                "matchup_quality", "competitiveness", "upset_potential",
                "h_quality_rest", "a_quality_rest",
                "h_netrtg_rested", "a_netrtg_rested",
                "h_traj_weighted", "a_traj_weighted",
                "elo_season_interaction", "compete_season_interaction",
                "clutch_compete_interaction",
            ])

        X.append(row); y.append(1 if hs > as_ else 0)
        if first: feature_names = names; first = False
        team_results[home].append((gd, hs > as_, hs - as_, away, hs, as_))
        team_results[away].append((gd, as_ > hs, as_ - hs, home, as_, hs))
        team_last[home] = gd; team_last[away] = gd
        K = 20; exp_h = 1 / (1 + 10 ** ((team_elo[away] - team_elo[home] - 50) / 400))
        team_elo[home] += K * ((1 if hs > as_ else 0) - exp_h)
        team_elo[away] += K * ((0 if hs > as_ else 1) - (1 - exp_h))

    return np.nan_to_num(np.array(X, dtype=np.float64)), np.array(y, dtype=np.int32), feature_names


# ═══════════════════════════════════════════════════════
# GENETIC INDIVIDUAL
# ═══════════════════════════════════════════════════════

class Individual:
    def __init__(self, n_features, target=100):
        prob = target / max(n_features, 1)
        self.features = [1 if random.random() < prob else 0 for _ in range(n_features)]
        self.hyperparams = {
            "n_estimators": random.randint(50, 300),
            "max_depth": random.randint(3, 10),
            "learning_rate": 10 ** random.uniform(-2.5, -0.5),
            "subsample": random.uniform(0.5, 1.0),
            "colsample_bytree": random.uniform(0.3, 1.0),
            "min_child_weight": random.randint(1, 15),
            "reg_alpha": 10 ** random.uniform(-6, 1),
            "reg_lambda": 10 ** random.uniform(-6, 1),
            "model_type": random.choice(["xgboost", "lightgbm", "catboost", "random_forest", "extra_trees", "stacking", "mlp"]),
            "calibration": random.choice(["isotonic", "sigmoid", "none"]),
        }
        self.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "composite": 0.0}
        self.generation = 0
        self.n_features = sum(self.features)

    def selected_indices(self):
        return [i for i, b in enumerate(self.features) if b]

    def to_dict(self):
        return {"n_features": self.n_features, "hyperparams": dict(self.hyperparams),
                "fitness": dict(self.fitness), "generation": self.generation}

    @staticmethod
    def crossover(p1, p2):
        child = Individual.__new__(Individual)
        n = len(p1.features)
        pt1, pt2 = sorted(random.sample(range(n), 2))
        child.features = p1.features[:pt1] + p2.features[pt1:pt2] + p1.features[pt2:]
        child.hyperparams = {}
        for key in p1.hyperparams:
            if isinstance(p1.hyperparams[key], (int, float)):
                w = random.random()
                val = w * p1.hyperparams[key] + (1 - w) * p2.hyperparams[key]
                child.hyperparams[key] = int(round(val)) if isinstance(p1.hyperparams[key], int) else val
            else:
                child.hyperparams[key] = random.choice([p1.hyperparams[key], p2.hyperparams[key]])
        child.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "composite": 0.0}
        child.generation = max(p1.generation, p2.generation) + 1
        child.n_features = sum(child.features)
        return child

    def mutate(self, rate=0.03, feature_importance=None):
        """Directed mutation: bias towards features that top performers use."""
        for i in range(len(self.features)):
            if random.random() < rate:
                if feature_importance is not None and i < len(feature_importance):
                    # Directed: more likely to ADD features that top performers use
                    imp = feature_importance[i]
                    if self.features[i] == 0 and imp > 0.5:
                        self.features[i] = 1  # Add high-importance feature
                    elif self.features[i] == 1 and imp < 0.15:
                        self.features[i] = 0  # Drop low-importance feature
                    else:
                        self.features[i] = 1 - self.features[i]
                else:
                    self.features[i] = 1 - self.features[i]
        if random.random() < 0.15: self.hyperparams["n_estimators"] = max(50, min(200, self.hyperparams["n_estimators"] + random.randint(-50, 50)))
        if random.random() < 0.15: self.hyperparams["max_depth"] = max(2, min(8, self.hyperparams["max_depth"] + random.randint(-2, 2)))
        if random.random() < 0.15: self.hyperparams["learning_rate"] = max(0.001, min(0.3, self.hyperparams["learning_rate"] * 10 ** random.uniform(-0.3, 0.3)))
        if random.random() < 0.10: self.hyperparams["model_type"] = random.choice(["xgboost", "lightgbm", "catboost", "random_forest", "extra_trees", "stacking", "mlp"])
        if random.random() < 0.05: self.hyperparams["calibration"] = random.choice(["isotonic", "sigmoid", "none"])
        self.n_features = sum(self.features)


# ═══════════════════════════════════════════════════════
# FITNESS EVALUATION
# ═══════════════════════════════════════════════════════

def _evaluate_stacking(ind, X_sub, y_eval, hp_eval, n_splits, fast):
    """Stacking: XGBoost + LightGBM + CatBoost base models → LogisticRegression meta-learner."""
    splits = n_splits if fast else max(n_splits, 3)
    tscv = TimeSeriesSplit(n_splits=splits)
    # Cap base learner estimators at 80 for speed
    base_est = min(hp_eval.get("n_estimators", 80), 80)
    depth = min(hp_eval.get("max_depth", 6), 6)
    lr = hp_eval.get("learning_rate", 0.1)

    briers, rois, all_p, all_y = [], [], [], []
    for ti, vi in tscv.split(X_sub):
        try:
            X_tr, y_tr = X_sub[ti], y_eval[ti]
            X_val, y_val = X_sub[vi], y_eval[vi]

            # Build 3 base models (capped at 80 estimators each)
            base_models = []
            # XGBoost
            base_models.append(xgb.XGBClassifier(
                n_estimators=base_est, max_depth=depth, learning_rate=lr,
                subsample=hp_eval.get("subsample", 0.8),
                colsample_bytree=hp_eval.get("colsample_bytree", 0.8),
                eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist"))
            # LightGBM
            base_models.append(lgbm.LGBMClassifier(
                n_estimators=base_est, max_depth=depth, learning_rate=lr,
                subsample=hp_eval.get("subsample", 0.8),
                num_leaves=min(2 ** depth - 1, 31),
                verbose=-1, random_state=42, n_jobs=-1))
            # CatBoost
            try:
                from catboost import CatBoostClassifier
                base_models.append(CatBoostClassifier(
                    iterations=min(base_est, 80), depth=min(depth, 6),
                    learning_rate=lr, verbose=0, random_state=42, thread_count=-1))
            except ImportError:
                # Fallback: use RandomForest if CatBoost unavailable
                from sklearn.ensemble import RandomForestClassifier
                base_models.append(RandomForestClassifier(
                    n_estimators=base_est, max_depth=depth,
                    random_state=42, n_jobs=-1))

            # Get OOF predictions from each base model using inner CV
            inner_cv = TimeSeriesSplit(n_splits=2)
            oof_preds = np.zeros((len(X_tr), len(base_models)))
            for m_idx, bm in enumerate(base_models):
                try:
                    oof = cross_val_predict(bm, X_tr, y_tr, cv=inner_cv, method="predict_proba")[:, 1]
                    oof_preds[:, m_idx] = oof
                except Exception:
                    # Fallback: use 0.5 if a base model fails
                    oof_preds[:, m_idx] = 0.5

            # Train meta-learner on OOF predictions
            meta = LogisticRegression(C=1.0, max_iter=200, random_state=42)
            meta.fit(oof_preds, y_tr)

            # Train base models on full training fold for validation predictions
            val_preds = np.zeros((len(X_val), len(base_models)))
            for m_idx, bm in enumerate(base_models):
                try:
                    bm_fresh = type(bm)(**bm.get_params())
                    bm_fresh.fit(X_tr, y_tr)
                    val_preds[:, m_idx] = bm_fresh.predict_proba(X_val)[:, 1]
                except Exception:
                    val_preds[:, m_idx] = 0.5

            # Final stacked prediction
            p = meta.predict_proba(val_preds)[:, 1]
            briers.append(brier_score_loss(y_val, p))
            rois.append(_log_loss_score(p, y_val))
            all_p.extend(p); all_y.extend(y_val)
        except Exception:
            briers.append(0.28); rois.append(0.0)

    return briers, rois, all_p, all_y


def _prune_correlated_features(X_sub, threshold=0.95):
    """Drop redundant features: for any pair with |corr| > threshold, drop the one with lower variance.

    Args:
        X_sub: numpy array (n_samples, n_features) — already selected features
        threshold: correlation threshold (default 0.95)

    Returns:
        (X_pruned, keep_mask): pruned array and boolean mask of kept column indices
    """
    try:
        n_cols = X_sub.shape[1]
        if n_cols < 2:
            return X_sub, np.ones(n_cols, dtype=bool)

        # Compute pairwise Pearson correlation
        corr = np.corrcoef(X_sub, rowvar=False)

        # Handle NaN correlations (constant columns, etc.)
        corr = np.nan_to_num(corr, nan=0.0)

        # Compute variance for each column (used to decide which to drop)
        variances = np.var(X_sub, axis=0)

        # Greedy pruning: iterate upper triangle, mark lower-variance feature for removal
        to_drop = set()
        for i in range(n_cols):
            if i in to_drop:
                continue
            for j in range(i + 1, n_cols):
                if j in to_drop:
                    continue
                if abs(corr[i, j]) > threshold:
                    # Drop the feature with lower variance
                    if variances[i] < variances[j]:
                        to_drop.add(i)
                        break  # i is dropped, no need to check more pairs for i
                    else:
                        to_drop.add(j)

        keep_mask = np.ones(n_cols, dtype=bool)
        for idx in to_drop:
            keep_mask[idx] = False

        return X_sub[:, keep_mask], keep_mask
    except Exception:
        # If anything fails, skip pruning entirely
        return X_sub, np.ones(X_sub.shape[1], dtype=bool)


def evaluate(ind, X, y, n_splits=2, fast=True):
    """Two-tier evaluation: fast (subsample + 2-fold) or full (all data + 3-fold)."""
    selected = ind.selected_indices()
    if len(selected) < 15 or len(selected) > 200:
        ind.fitness = {"brier": 0.30, "roi": 0.0, "sharpe": 0.0, "calibration": 0.15, "composite": -1.0,
                       "features_pruned": 0}
        return

    # Fast mode: subsample recent games for speed
    if fast and X.shape[0] > FAST_EVAL_GAMES:
        X_eval, y_eval = X[-FAST_EVAL_GAMES:], y[-FAST_EVAL_GAMES:]
    else:
        X_eval, y_eval = X, y

    X_sub = np.nan_to_num(X_eval[:, selected], nan=0.0, posinf=1e6, neginf=-1e6)

    # ── Correlation pruning: drop redundant features (|corr| > 0.95) ──
    n_before = X_sub.shape[1]
    X_sub, _keep_mask = _prune_correlated_features(X_sub, threshold=0.95)
    n_pruned = n_before - X_sub.shape[1]

    hp = ind.hyperparams

    # Cap estimators for speed (fast mode)
    hp_eval = dict(hp)
    if fast:
        hp_eval["n_estimators"] = min(hp["n_estimators"], 120)

    # ── STACKING: special path ──
    if hp_eval["model_type"] == "stacking":
        briers, rois, all_p, all_y = _evaluate_stacking(ind, X_sub, y_eval, hp_eval, n_splits, fast)
    else:
        # ── Standard single-model path ──
        model = _build(hp_eval)
        if model is None: ind.fitness["composite"] = -1.0; return

        splits = n_splits if fast else max(n_splits, 3)
        tscv = TimeSeriesSplit(n_splits=splits)
        briers, rois, all_p, all_y = [], [], [], []
        for ti, vi in tscv.split(X_sub):
            try:
                m = clone(model)
                if hp_eval["calibration"] != "none":
                    m = CalibratedClassifierCV(m, method=hp_eval["calibration"], cv=2)
                m.fit(X_sub[ti], y_eval[ti])
                p = m.predict_proba(X_sub[vi])[:, 1]
                briers.append(brier_score_loss(y_eval[vi], p))
                rois.append(_log_loss_score(p, y_eval[vi]))
                all_p.extend(p); all_y.extend(y_eval[vi])
            except Exception:
                briers.append(0.28); rois.append(0.0)

    if not briers:
        ind.fitness = {"brier": 0.30, "roi": 0.0, "sharpe": 0.0, "calibration": 0.15, "composite": -1.0,
                       "features_pruned": n_pruned}
        return
    ab = np.mean(briers)
    ar = np.mean(rois)  # log_loss_score: 0 = coin flip, 1 = great calibration
    # Sharpe of calibration: consistency of log-loss scores across folds
    sh = np.mean(rois) / max(np.std(rois), 0.01) if len(rois) > 1 else 0.0
    ce = _ece(np.array(all_p), np.array(all_y)) if all_p else 0.15
    # FITNESS: Brier 40% + log_loss_score 25% + Sharpe_of_calibration 20% + ECE 15%
    # log_loss_score breaks the circular ROI problem: rewards calibration without
    # comparing model to itself. Sharpe measures consistency across folds.
    n_features = len(selected)
    feature_penalty = 0.001 * max(0, n_features - 80)  # Soft pressure toward 80 features
    composite = (0.40 * (1 - ab)
                 + 0.25 * ar  # ar is already in [0, 1] from _log_loss_score
                 + 0.20 * max(0, min(sh, 3) / 3)
                 + 0.15 * (1 - ce)
                 - feature_penalty)
    ind.fitness = {"brier": round(ab, 5), "roi": round(ar, 4), "sharpe": round(sh, 4),
                   "calibration": round(ce, 4),
                   "composite": round(composite, 5),
                   "features_pruned": n_pruned}


def _build(hp):
    try:
        mt = hp["model_type"]
        n_est = min(hp["n_estimators"], 200)  # Cap for speed
        depth = min(hp["max_depth"], 8)       # Cap to prevent overfitting
        lr = hp["learning_rate"]
        if mt == "xgboost":
            return xgb.XGBClassifier(n_estimators=n_est, max_depth=depth,
                                     learning_rate=lr, subsample=hp["subsample"],
                                     colsample_bytree=hp["colsample_bytree"], min_child_weight=hp["min_child_weight"],
                                     reg_alpha=hp["reg_alpha"], reg_lambda=hp["reg_lambda"],
                                     eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist")
        elif mt == "lightgbm":
            return lgbm.LGBMClassifier(n_estimators=n_est, max_depth=depth,
                                       learning_rate=lr, subsample=hp["subsample"],
                                       num_leaves=min(2 ** depth - 1, 31),
                                       reg_alpha=hp["reg_alpha"], reg_lambda=hp["reg_lambda"],
                                       min_child_samples=20, feature_fraction=0.7,
                                       verbose=-1, random_state=42, n_jobs=-1)
        elif mt == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(iterations=min(n_est, 100), depth=min(depth, 6),
                                      learning_rate=lr, l2_leaf_reg=hp["reg_lambda"],
                                      verbose=0, random_state=42, thread_count=-1)
        elif mt == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                          min_samples_leaf=max(int(hp["min_child_weight"]), 5),
                                          max_features="sqrt", random_state=42, n_jobs=-1)
        elif mt == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(n_estimators=n_est, max_depth=depth,
                                        min_samples_leaf=max(int(hp["min_child_weight"]), 5),
                                        max_features="sqrt", random_state=42, n_jobs=-1)
        elif mt == "mlp":
            try:
                return Pipeline([
                    ('scaler', StandardScaler()),
                    ('mlp', MLPClassifier(
                        hidden_layer_sizes=(128, 64, 32),
                        activation='relu',
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=42
                    ))
                ])
            except Exception:
                # Fallback to XGBoost if MLP fails to build
                return xgb.XGBClassifier(n_estimators=n_est, max_depth=depth,
                                         learning_rate=lr, eval_metric="logloss",
                                         random_state=42, n_jobs=-1, tree_method="hist")
        elif mt == "stacking":
            # Return None — stacking is handled specially in evaluate()
            return None
        else:
            return xgb.XGBClassifier(n_estimators=n_est, max_depth=depth,
                                     learning_rate=lr, eval_metric="logloss",
                                     random_state=42, n_jobs=-1, tree_method="hist")
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=min(hp["n_estimators"], 100), max_depth=min(hp["max_depth"], 6),
                                          learning_rate=hp["learning_rate"], random_state=42)


def _simulate_betting_legacy(probs, actuals, edge=0.05):
    """LEGACY: Circular ROI — compares model to itself (1/prob as fair odds). DEPRECATED."""
    stake, profit, n = 10, 0, 0
    for p, a in zip(probs, actuals):
        if p > 0.5 + edge:
            n += 1; profit += stake * (1 / p - 1) if a == 1 else -stake
        elif p < 0.5 - edge:
            n += 1; profit += stake * (1 / (1 - p) - 1) if a == 0 else -stake
    return profit / (n * stake) if n > 0 else 0.0


def _log_loss_score(probs, actuals):
    """Log-loss proxy for ROI: measures calibration quality without circular odds.

    Returns a score in [0, 1] range where:
      - 0.6931 (coin flip / no skill) maps to 0.0
      - 0.5 (good calibration) maps to 1.0
      - Lower log-loss = higher score = better edge potential
    Clamps to [0, 1] range.
    """
    try:
        probs = np.clip(np.asarray(probs, dtype=float), 1e-7, 1 - 1e-7)
        actuals = np.asarray(actuals, dtype=float)
        if len(probs) < 10:
            return 0.0
        ll = log_loss(actuals, probs)
        # Linear map: 0.6931 (coin flip) → 0.0, 0.5 → 1.0
        # score = (0.6931 - ll) / (0.6931 - 0.5)
        COIN_FLIP = 0.6931
        GOOD_TARGET = 0.5
        score = (COIN_FLIP - ll) / (COIN_FLIP - GOOD_TARGET)
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        # Fallback: use legacy method if log_loss fails
        return max(0.0, _simulate_betting_legacy(probs, actuals))


def _ece(probs, actuals, n_bins=10):
    if len(probs) == 0: return 1.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0: continue
        ece += mask.sum() / len(probs) * abs(probs[mask].mean() - actuals[mask].mean())
    return ece


# ═══════════════════════════════════════════════════════
# EVOLUTION ENGINE
# ═══════════════════════════════════════════════════════

POP_SIZE = 60            # Moderate pop — balance diversity and speed
ELITE_SIZE = 5           # Top ~8% preserved
BASE_MUT = 0.04          # Low mutation — allow convergence on good regions
CROSSOVER_RATE = 0.80    # High recombination
TARGET_FEATURES = 80     # TIGHT feature sets — prevent overfitting (9551 games / 80 features = ~120 samples/feature)
N_SPLITS = 3             # 3-fold walk-forward for reliable Brier estimates
GENS_PER_CYCLE = 3       # Save every 3 gens
COOLDOWN = 5             # Minimal cooldown
TOURNAMENT_SIZE = 4      # Moderate selection pressure — preserve exploration
DIVERSITY_RESTART = 20   # Give more time before nuking population
FAST_EVAL_GAMES = 7000   # More data in fast eval — better signal
FULL_EVAL_TOP = 10       # Full eval for top 10 — better selection of champion

# ── Feature importance tracking (directed mutation) ──
_feature_importance = None  # numpy array: how often each feature appears in top individuals

# ── Shared state for /api/predict (set from evolution_loop) ──
_evo_X = None          # Full feature matrix
_evo_y = None          # Labels
_evo_features = None   # Feature names list
_evo_best = None       # Best individual (dict with features, hyperparams, fitness)
_evo_games = None      # Raw games list (for building today's features)

# ── Checkpoint system (monotonic ratchet) ──
_checkpoints = []      # Last 10 checkpoints: [{best, brier, generation, timestamp}]
CHECKPOINT_DIR = Path(__file__).parent / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Remote Config (mutable at runtime via API) ──
remote_config = {
    "pending_reset": False,
    "pending_params": {},
    "injected_features": [],     # Feature ideas from OpenClaw
    "commands": [],              # Queued commands (restart, diversify, etc.)
}


def evolution_loop():
    """Main 24/7 genetic evolution loop — runs in background thread."""
    global TARGET_FEATURES, CROSSOVER_RATE, COOLDOWN, POP_SIZE
    global _evo_X, _evo_y, _evo_features, _evo_best, _evo_games
    log("=" * 60)
    log("REAL GENETIC EVOLUTION LOOP v3 — STARTING")
    log(f"Pop: {POP_SIZE} | Target features: {TARGET_FEATURES} | Gens/cycle: {GENS_PER_CYCLE}")
    log("=" * 60)

    # ── Supabase Run Logger + Auto-Cut ──
    global _global_logger
    run_logger = None
    if _HAS_LOGGER:
        try:
            run_logger = RunLogger(local_dir=str(DATA_DIR / "run-logs"))
            _global_logger = run_logger
            # Test DB connection explicitly
            db_url = os.environ.get("DATABASE_URL", "")
            if db_url:
                log(f"[RUN-LOGGER] DATABASE_URL set ({db_url[:25]}...)")
                try:
                    import psycopg2
                    conn = psycopg2.connect(db_url, connect_timeout=10, options="-c search_path=public")
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM public.nba_evolution_gens")
                    cnt = cur.fetchone()[0]
                    conn.close()
                    log(f"[RUN-LOGGER] Supabase CONNECTED — {cnt} existing gen rows")
                except Exception as e:
                    log(f"[RUN-LOGGER] Supabase connection FAILED: {e}", "ERROR")
            else:
                log("[RUN-LOGGER] DATABASE_URL NOT SET — logging local only", "WARN")
            log("[RUN-LOGGER] Initialized — auto-cut ACTIVE")
        except Exception as e:
            log(f"[RUN-LOGGER] Init failed: {e} — continuing without", "WARN")

    live["status"] = "LOADING DATA"
    pull_seasons()
    games = load_all_games()
    live["games"] = len(games)
    log(f"Games loaded: {len(games)}")

    if len(games) < 500:
        log("NOT ENOUGH GAMES", "ERROR")
        live["status"] = "ERROR: no data"
        return

    live["status"] = "BUILDING FEATURES"
    X, y, feature_names = build_features(games)
    log(f"Raw feature matrix: {X.shape} ({X.shape[1]} features)")

    # ── CRITICAL: Remove zero-variance (noise) features ──
    # Categories 11-15 (referee, player, quarter, defense, polymarket) have no data
    # and compute to constant defaults (0.0, 0.5, etc.). These add noise, not signal.
    live["status"] = "FILTERING NOISE FEATURES"
    variances = np.var(X, axis=0)
    valid_mask = variances > 1e-10  # Keep only features with real variance
    n_removed = int((~valid_mask).sum())
    if n_removed > 0:
        X = X[:, valid_mask]
        feature_names = [f for f, v in zip(feature_names, valid_mask) if v]
        log(f"NOISE FILTER: removed {n_removed} zero-variance features (no real data)")
    n_feat = X.shape[1]
    live["feature_candidates"] = n_feat
    log(f"Clean feature matrix: {X.shape} ({n_feat} usable features)")

    # ── Share state for /api/predict ──
    _evo_X = X
    _evo_y = y
    _evo_features = feature_names
    _evo_games = games

    # Try restore state
    population = []
    generation = 0
    best_ever = None
    stagnation = 0
    mutation_rate = BASE_MUT

    state_file = STATE_DIR / "population.json"
    if state_file.exists():
        try:
            st = json.loads(state_file.read_text())
            generation = st["generation"]
            stagnation = st.get("stagnation_counter", 0)
            mutation_rate = st.get("mutation_rate", BASE_MUT)
            for d in st["population"]:
                ind = Individual.__new__(Individual)
                ind.features = d["features"]; ind.hyperparams = d["hyperparams"]
                ind.fitness = d["fitness"]; ind.generation = d.get("generation", 0)
                ind.n_features = sum(ind.features)
                population.append(ind)
            if st.get("best_ever"):
                be = st["best_ever"]
                best_ever = Individual.__new__(Individual)
                best_ever.features = be["features"]; best_ever.hyperparams = be["hyperparams"]
                best_ever.fitness = be["fitness"]; best_ever.generation = be.get("generation", 0)
                best_ever.n_features = sum(best_ever.features)
            log(f"RESTORED: gen {generation}, {len(population)} individuals")
        except Exception as e:
            log(f"Restore failed: {e}", "WARN")
            population = []

    # Force reset if population size or feature count changed
    needs_reset = False
    if population and len(population) != POP_SIZE:
        log(f"POP_SIZE changed: {len(population)} → {POP_SIZE}. Resetting population.")
        needs_reset = True
    if population and len(population[0].features) != n_feat:
        log(f"Feature count changed: {len(population[0].features)} → {n_feat}. Resetting population.")
        needs_reset = True
    if needs_reset:
        # Keep top 5 elites, re-init rest
        elites = sorted(population, key=lambda x: x.fitness.get("composite", 0), reverse=True)[:5]
        population = []
        for e in elites:
            # Resize feature mask if needed
            if len(e.features) < n_feat:
                e.features.extend([random.randint(0, 1) for _ in range(n_feat - len(e.features))])
            elif len(e.features) > n_feat:
                e.features = e.features[:n_feat]
            e.n_features = sum(e.features)
            # Add new model types to elite hyperparams
            if e.hyperparams.get("model_type") not in ["xgboost", "lightgbm", "catboost", "random_forest", "extra_trees", "stacking", "mlp"]:
                e.hyperparams["model_type"] = random.choice(["xgboost", "lightgbm", "catboost", "random_forest", "extra_trees", "stacking", "mlp"])
            population.append(e)
        log(f"Kept {len(population)} elites, generating {POP_SIZE - len(population)} fresh individuals")

    if not population or needs_reset:
        # Equal distribution of all 6 model types in initial population
        all_model_types = ["xgboost", "lightgbm", "catboost", "random_forest", "extra_trees", "stacking", "mlp"]
        existing_count = len(population)
        slots_left = POP_SIZE - existing_count
        idx = 0
        while len(population) < POP_SIZE:
            ind = Individual(n_feat, TARGET_FEATURES)
            ind.hyperparams["model_type"] = all_model_types[idx % len(all_model_types)]
            population.append(ind)
            idx += 1
        log(f"Population ready: {POP_SIZE} individuals, {n_feat} feature candidates (equal model distribution)")

    live["pop_size"] = len(population)

    # ── INFINITE LOOP ──
    cycle = 0
    while True:
        cycle += 1
        live["cycle"] = cycle
        live["status"] = f"EVOLVING (cycle {cycle})"
        log(f"\n{'='*50}")
        log(f"CYCLE {cycle} — {GENS_PER_CYCLE} generations")
        log(f"{'='*50}")
        cycle_start = time.time()

        for gen_i in range(GENS_PER_CYCLE):
            generation += 1
            live["generation"] = generation
            gen_start = time.time()

            # ── FAST EVAL: all non-elites (elites keep their fitness) ──
            for i, ind in enumerate(population):
                if i < ELITE_SIZE and ind.fitness.get("composite", 0) > 0:
                    continue  # Skip re-eval of elites — already scored
                evaluate(ind, X, y, N_SPLITS, fast=True)

            # Sort by composite
            population.sort(key=lambda x: x.fitness["composite"], reverse=True)

            # ── FULL EVAL: top candidates get precise 3-fold on ALL data ──
            for ind in population[:FULL_EVAL_TOP]:
                evaluate(ind, X, y, n_splits=3, fast=False)

            # Re-sort after full eval
            population.sort(key=lambda x: x.fitness["composite"], reverse=True)
            best = population[0]

            # ── Update feature importance from top performers ──
            global _feature_importance
            if _feature_importance is None:
                _feature_importance = np.zeros(n_feat)
            top_features = np.zeros(n_feat)
            for ind in population[:10]:
                for idx in ind.selected_indices():
                    if idx < n_feat:
                        top_features[idx] += 1
            _feature_importance = 0.7 * _feature_importance + 0.3 * (top_features / 10.0)

            # Track best
            prev_brier = best_ever.fitness["brier"] if best_ever else 1.0
            if best_ever is None or best.fitness["composite"] > best_ever.fitness["composite"]:
                best_ever = Individual.__new__(Individual)
                best_ever.features = best.features[:]
                best_ever.hyperparams = dict(best.hyperparams)
                best_ever.fitness = dict(best.fitness)
                best_ever.n_features = best.n_features
                best_ever.generation = generation
                # Share for /api/predict
                _evo_best = {
                    "features": best_ever.features[:],
                    "hyperparams": dict(best_ever.hyperparams),
                    "fitness": dict(best_ever.fitness),
                    "generation": best_ever.generation,
                    "n_features": best_ever.n_features,
                }

            # Stagnation detection
            if abs(best.fitness["brier"] - prev_brier) < 0.0005:
                stagnation += 1
            else:
                stagnation = 0

            # Adaptive mutation — gentle escalation, capped to prevent chaos
            if stagnation >= 12:
                mutation_rate = min(0.10, mutation_rate * 1.3)  # Hard cap at 10%
            elif stagnation >= 6:
                mutation_rate = min(0.08, mutation_rate * 1.15)
            elif stagnation == 0:
                mutation_rate = max(BASE_MUT, mutation_rate * 0.90)

            # Update live state
            live["best_brier"] = best_ever.fitness["brier"]
            live["best_roi"] = best_ever.fitness["roi"]
            live["best_sharpe"] = best_ever.fitness["sharpe"]
            live["best_features"] = best_ever.n_features
            live["best_model_type"] = best_ever.hyperparams["model_type"]
            live["mutation_rate"] = round(mutation_rate, 4)
            live["stagnation"] = stagnation
            live["history"].append({
                "gen": generation, "brier": best.fitness["brier"], "roi": best.fitness["roi"],
                "composite": best.fitness["composite"], "features": best.n_features,
            })
            if len(live["history"]) > 500:
                live["history"] = live["history"][-500:]

            ge = time.time() - gen_start
            log(f"Gen {generation}: Brier={best.fitness['brier']:.4f} ROI={best.fitness['roi']:.1%} "
                f"Sharpe={best.fitness['sharpe']:.2f} Feat={best.n_features} "
                f"Pruned={best.fitness.get('features_pruned', 0)} "
                f"Model={best.hyperparams['model_type']} Stag={stagnation} ({ge:.0f}s)")

            # ── Supabase: log generation (direct psycopg2 — bypasses pool) ──
            try:
                _db_url = os.environ.get("DATABASE_URL", "")
                if _db_url:
                    import psycopg2
                    _dbconn = psycopg2.connect(_db_url, connect_timeout=10, options="-c search_path=public")
                    _dbconn.autocommit = True
                    _cur = _dbconn.cursor()
                    pop_diversity = float(np.std([ind.fitness.get("composite", 0) for ind in population]))
                    avg_comp = float(np.mean([ind.fitness.get("composite", 0) for ind in population]))
                    _cur.execute("""INSERT INTO public.nba_evolution_gens
                        (cycle, generation, best_brier, best_roi, best_sharpe, best_composite,
                         n_features, model_type, mutation_rate, avg_composite, pop_diversity,
                         gen_duration_s, improved)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (cycle, generation, float(best.fitness["brier"]), float(best.fitness["roi"]),
                         float(best.fitness["sharpe"]), float(best.fitness["composite"]),
                         int(best.n_features), str(best.hyperparams["model_type"]),
                         float(mutation_rate), avg_comp, pop_diversity, float(ge),
                         bool(best_ever is not None and best.fitness["brier"] < best_ever.fitness["brier"] - 0.0001)))
                    log(f"[SUPABASE] Gen {generation} logged OK")
                    # Log top 10 evals
                    for rank, ind in enumerate(population[:10]):
                        _cur.execute("""INSERT INTO public.nba_evolution_evals
                            (generation, individual_rank, brier, roi, sharpe, composite, n_features, model_type)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                            (generation, rank + 1, float(ind.fitness["brier"]), float(ind.fitness["roi"]),
                             float(ind.fitness["sharpe"]), float(ind.fitness["composite"]),
                             int(ind.n_features), str(ind.hyperparams["model_type"])))
                    log(f"[SUPABASE] Top 10 evals logged OK")
                    _cur.close()
                    _dbconn.close()
            except Exception as e:
                log(f"[SUPABASE] Gen log FAILED: {e}", "ERROR")

            # ── Auto-Cut check ──
            if run_logger:
                try:
                    pop_diversity = float(np.std([ind.fitness.get("composite", 0) for ind in population]))
                    avg_comp = float(np.mean([ind.fitness.get("composite", 0) for ind in population]))
                    run_logger.log_generation(
                        cycle=cycle, generation=generation,
                        best={"brier": best.fitness["brier"], "roi": best.fitness["roi"],
                              "sharpe": best.fitness["sharpe"], "composite": best.fitness["composite"],
                              "n_features": best.n_features, "model_type": best.hyperparams["model_type"]},
                        mutation_rate=mutation_rate, avg_composite=avg_comp,
                        pop_diversity=pop_diversity, duration_s=ge)

                    # Log top 10 evals
                    top10 = [{"brier": ind.fitness["brier"], "roi": ind.fitness["roi"],
                              "sharpe": ind.fitness["sharpe"], "composite": ind.fitness["composite"],
                              "n_features": ind.n_features, "model_type": ind.hyperparams["model_type"]}
                             for ind in population[:10]]
                    run_logger.log_top_evals(generation, top10)

                    # ── Auto-Cut check ──
                    engine_state = {
                        "mutation_rate": mutation_rate, "stagnation": stagnation,
                        "pop_size": len(population), "pop_diversity": float(pop_diversity),
                    }
                    cut_actions = run_logger.check_auto_cut(best.fitness, engine_state)
                    for action in cut_actions:
                        atype = action["type"]
                        params = action.get("params", {})
                        if atype == "config":
                            remote_config["pending_params"].update(params)
                            log(f"[AUTO-CUT] Config queued: {params}")
                        elif atype == "emergency_diversify":
                            remote_config["commands"].append("diversify")
                            if "pop_size" in params:
                                remote_config["pending_params"]["pop_size"] = params["pop_size"]
                            if "mutation_rate" in params:
                                remote_config["pending_params"]["mutation_rate"] = params["mutation_rate"]
                            log(f"[AUTO-CUT] Emergency diversify: {params}")
                        elif atype == "inject":
                            count = params.get("count", 10)
                            remote_config["commands"].append("diversify")
                            log(f"[AUTO-CUT] Injecting {count} random individuals")
                        elif atype == "full_reset":
                            remote_config["pending_reset"] = True
                            remote_config["pending_params"].update(params)
                            log(f"[AUTO-CUT] FULL RESET triggered: {params}")
                        elif atype == "flag":
                            live["pause_betting"] = params.get("pause_betting", False)
                            log(f"[AUTO-CUT] Flag set: {params}")
                except Exception as e:
                    log(f"[RUN-LOGGER] Gen log error: {e}", "WARN")

            # Next generation
            new_pop = []
            for i in range(ELITE_SIZE):
                e = Individual.__new__(Individual)
                e.features = population[i].features[:]; e.hyperparams = dict(population[i].hyperparams)
                e.fitness = dict(population[i].fitness); e.n_features = population[i].n_features
                e.generation = population[i].generation; new_pop.append(e)

            if stagnation >= 10:
                n_inject = POP_SIZE // 4  # 25% fresh blood
                for _ in range(n_inject):
                    new_pop.append(Individual(n_feat, TARGET_FEATURES))
                log(f"  INJECTION: {n_inject} fresh individuals (25% population reset)")

            if stagnation >= DIVERSITY_RESTART:
                n_restart = POP_SIZE // 2  # 50% restart for extreme stagnation
                new_pop = new_pop[:ELITE_SIZE]  # Keep only elite
                for _ in range(n_restart):
                    new_pop.append(Individual(n_feat, TARGET_FEATURES))
                log(f"  EMERGENCY RESTART: {n_restart} fresh individuals (50% population)")
                stagnation = 0  # Reset counter

            # Anti-convergence: if top 5 all same model, force diversity
            all_model_types = ["xgboost", "lightgbm", "catboost", "random_forest", "extra_trees", "stacking", "mlp"]
            top_models = [population[i].hyperparams["model_type"] for i in range(min(5, len(population)))]
            if len(set(top_models)) == 1 and stagnation >= 3:
                diverse_types = [t for t in all_model_types if t != top_models[0]]
                for _ in range(3):
                    fresh = Individual(n_feat, TARGET_FEATURES)
                    fresh.hyperparams["model_type"] = random.choice(diverse_types)
                    new_pop.append(fresh)
                log(f"  ANTI-CONVERGENCE: injected 3 diverse models (all top5 were {top_models[0]})")

            # Force model diversity: if >40% same model_type, force others to alternatives
            # Good stacking requires diverse base models — CatBoost monoculture kills ensemble quality
            pop_models = [ind.hyperparams["model_type"] for ind in new_pop]
            model_counts = Counter(pop_models)
            dominant_type, dominant_count = model_counts.most_common(1)[0]
            if dominant_count > 0.40 * len(new_pop):
                n_force = int(0.25 * len(new_pop))
                alt_types = [t for t in all_model_types if t != dominant_type]
                candidates = list(range(ELITE_SIZE, len(new_pop)))
                random.shuffle(candidates)
                forced = 0
                for ci in candidates:
                    if forced >= n_force:
                        break
                    if new_pop[ci].hyperparams["model_type"] == dominant_type:
                        new_pop[ci].hyperparams["model_type"] = random.choice(alt_types)
                        forced += 1
                log(f"  MODEL DIVERSITY: forced {forced} individuals away from {dominant_type} (was {dominant_count}/{len(new_pop)})")

            # Every 5 generations: inject 3 stacking individuals
            if generation % 5 == 0:
                for _ in range(3):
                    stk = Individual(n_feat, TARGET_FEATURES)
                    stk.hyperparams["model_type"] = "stacking"
                    new_pop.append(stk)
                log(f"  STACKING INJECTION: 3 stacking individuals added (gen {generation})")

            while len(new_pop) < POP_SIZE:
                cs = random.sample(population, min(TOURNAMENT_SIZE, len(population)))
                p1 = max(cs, key=lambda x: x.fitness["composite"])
                cs2 = random.sample(population, min(TOURNAMENT_SIZE, len(population)))
                p2 = max(cs2, key=lambda x: x.fitness["composite"])
                child = Individual.crossover(p1, p2) if random.random() < CROSSOVER_RATE else Individual.__new__(Individual)
                if not hasattr(child, 'features') or child.features is None:
                    child.features = p1.features[:]; child.hyperparams = dict(p1.hyperparams)
                    child.fitness = dict(p1.fitness); child.n_features = p1.n_features; child.generation = generation
                child.mutate(mutation_rate, feature_importance=_feature_importance)
                new_pop.append(child)

            population = new_pop

        # Save state
        live["status"] = "SAVING"
        try:
            sel_names = [feature_names[i] for i in best_ever.selected_indices() if i < len(feature_names)] if best_ever else []
            state_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "generation": generation, "n_features": n_feat,
                "stagnation_counter": stagnation, "mutation_rate": mutation_rate,
                "population": [{"features": ind.features,
                                "hyperparams": {k: (float(v) if isinstance(v, np.floating) else v) for k, v in ind.hyperparams.items()},
                                "fitness": ind.fitness, "generation": ind.generation}
                               for ind in population],
                "best_ever": {"features": best_ever.features,
                              "hyperparams": {k: (float(v) if isinstance(v, np.floating) else v) for k, v in best_ever.hyperparams.items()},
                              "fitness": best_ever.fitness, "generation": best_ever.generation} if best_ever else None,
                "history": live["history"][-200:],
            }
            (STATE_DIR / "population.json").write_text(json.dumps(state_data, default=str))

            results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": cycle, "generation": generation, "pop_size": POP_SIZE,
                "mutation_rate": round(mutation_rate, 4), "stagnation": stagnation,
                "best": {"brier": best_ever.fitness["brier"], "roi": best_ever.fitness["roi"],
                         "sharpe": best_ever.fitness["sharpe"], "n_features": best_ever.n_features,
                         "model_type": best_ever.hyperparams["model_type"],
                         "selected_features": sel_names[:30]},
                "top5": [ind.to_dict() for ind in sorted(population, key=lambda x: x.fitness["composite"], reverse=True)[:5]],
                "history_last20": live["history"][-20:],
            }
            ts = datetime.now().strftime("%Y%m%d-%H%M")
            (RESULTS_DIR / f"evolution-{ts}.json").write_text(json.dumps(results, indent=2, default=str))
            (RESULTS_DIR / "evolution-latest.json").write_text(json.dumps(results, indent=2, default=str))

            live["top5"] = results["top5"]
            log(f"Results saved: evolution-{ts}.json")

            # ── Supabase: log cycle (direct psycopg2) ──
            try:
                _db_url = os.environ.get("DATABASE_URL", "")
                if _db_url:
                    import psycopg2
                    _dbconn = psycopg2.connect(_db_url, connect_timeout=10, options="-c search_path=public")
                    _dbconn.autocommit = True
                    _cur = _dbconn.cursor()
                    pop_diversity = float(np.std([ind.fitness.get("composite", 0) for ind in population]))
                    avg_comp = float(np.mean([ind.fitness.get("composite", 0) for ind in population]))
                    sel_features = [feature_names[i] for i in best_ever.selected_indices() if i < len(feature_names)] if best_ever else []
                    _cur.execute("""INSERT INTO public.nba_evolution_runs
                        (cycle, generation, best_brier, best_roi, best_sharpe, best_calibration,
                         best_composite, best_features, best_model_type, pop_size, mutation_rate,
                         crossover_rate, stagnation, games, feature_candidates, cycle_duration_s,
                         avg_composite, pop_diversity, top5, selected_features)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (cycle, generation, float(best_ever.fitness["brier"]), float(best_ever.fitness["roi"]),
                         float(best_ever.fitness["sharpe"]), float(best_ever.fitness.get("calibration", 0)),
                         float(best_ever.fitness["composite"]), int(best_ever.n_features),
                         str(best_ever.hyperparams["model_type"]), len(population), float(mutation_rate),
                         float(CROSSOVER_RATE), stagnation, len(games), n_feat,
                         float(time.time() - cycle_start), avg_comp, pop_diversity,
                         json.dumps(results.get("top5", [])[:5], default=str),
                         json.dumps(sel_features[:50], default=str)))
                    _cur.close()
                    _dbconn.close()
                    log("[SUPABASE] Cycle logged OK")
            except Exception as e:
                log(f"[SUPABASE] Cycle log FAILED: {e}", "ERROR")
        except Exception as e:
            log(f"Save error: {e}", "ERROR")

        # VM callback
        try:
            import urllib.request
            body = json.dumps({"generation": generation, "brier": best_ever.fitness["brier"],
                               "roi": best_ever.fitness["roi"], "features": best_ever.n_features}, default=str).encode()
            req = urllib.request.Request(f"{VM_URL}/callback/evolution", data=body,
                                        headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
            log("VM callback OK")
        except Exception:
            pass  # Best-effort

        # ── Check remote commands from OpenClaw ──
        if remote_config["pending_params"]:
            params = remote_config["pending_params"]
            log(f"REMOTE CONFIG UPDATE: {params}")
            if "pop_size" in params:
                POP_SIZE_NEW = min(int(params["pop_size"]), 80)  # HARD CAP: too large = slow + no convergence
                if POP_SIZE_NEW != len(population):
                    # Resize population
                    if POP_SIZE_NEW > len(population):
                        while len(population) < POP_SIZE_NEW:
                            population.append(Individual(n_feat, TARGET_FEATURES))
                        log(f"Population expanded: {len(population)} → {POP_SIZE_NEW}")
                    else:
                        population = sorted(population, key=lambda x: x.fitness["composite"], reverse=True)[:POP_SIZE_NEW]
                        log(f"Population reduced to top {POP_SIZE_NEW}")
                    live["pop_size"] = len(population)
            if "mutation_rate" in params:
                mutation_rate = min(float(params["mutation_rate"]), 0.10)  # HARD CAP: never above 10%
                log(f"Mutation rate set to {mutation_rate}")
            if "target_features" in params:
                TARGET_FEATURES = min(int(params["target_features"]), 150)  # HARD CAP: prevent feature bloat
                log(f"Target features set to {TARGET_FEATURES}")
            if "crossover_rate" in params:
                CROSSOVER_RATE = float(params["crossover_rate"])
            if "cooldown" in params:
                COOLDOWN = int(params["cooldown"])
            remote_config["pending_params"] = {}

        if remote_config["pending_reset"]:
            log("REMOTE RESET: Reinitializing 50% of population with fresh individuals")
            n_keep = max(ELITE_SIZE, len(population) // 4)
            population = sorted(population, key=lambda x: x.fitness["composite"], reverse=True)[:n_keep]
            while len(population) < POP_SIZE:
                population.append(Individual(n_feat, TARGET_FEATURES))
            live["pop_size"] = len(population)
            stagnation = 0
            mutation_rate = BASE_MUT
            remote_config["pending_reset"] = False
            log(f"Reset complete: kept {n_keep} elites, {len(population)} total")

        for cmd in remote_config.get("commands", []):
            if cmd == "diversify":
                n_new = POP_SIZE // 3
                worst = len(population) - n_new
                population = sorted(population, key=lambda x: x.fitness["composite"], reverse=True)[:worst]
                for _ in range(n_new):
                    population.append(Individual(n_feat, TARGET_FEATURES))
                log(f"REMOTE DIVERSIFY: replaced {n_new} worst with fresh individuals")
            elif cmd == "boost_mutation":
                mutation_rate = min(0.25, mutation_rate * 2)
                log(f"REMOTE: mutation boosted to {mutation_rate}")
        remote_config["commands"] = []

        # Refresh data every 10 cycles
        if cycle % 10 == 0:
            try:
                pull_seasons()
                new_games = load_all_games()
                if len(new_games) > len(games):
                    games = new_games
                    X, y, feature_names = build_features(games)
                    n_feat = X.shape[1]
                    live["games"] = len(games)
                    log(f"Data refreshed: {len(games)} games")
            except Exception as e:
                log(f"Refresh failed: {e}", "WARN")

        ce = time.time() - cycle_start
        log(f"\nCycle {cycle} done in {ce:.0f}s | Best: Brier={best_ever.fitness['brier']:.4f}")
        live["status"] = f"COOLDOWN ({COOLDOWN}s)"
        time.sleep(COOLDOWN)


# ═══════════════════════════════════════════════════════
# GRADIO DASHBOARD
# ═══════════════════════════════════════════════════════

def dash_status():
    if not live["history"]:
        brier_trend = "No data yet"
    else:
        last5 = live["history"][-5:]
        brier_trend = " -> ".join(f"{h['brier']:.4f}" for h in last5)

    return f"""## NOMOS NBA QUANT — Genetic Evolution LIVE

| Metric | Value |
|--------|-------|
| **Status** | {live['status']} |
| **Cycle** | {live['cycle']} |
| **Generation** | {live['generation']} |
| **Best Brier** | {live['best_brier']:.4f} |
| **Best ROI** | {live['best_roi']:.1%} |
| **Best Sharpe** | {live['best_sharpe']:.2f} |
| **Best Features** | {live['best_features']} |
| **Model Type** | {live['best_model_type']} |
| **Population** | {live['pop_size']} |
| **Mutation Rate** | {live['mutation_rate']:.4f} |
| **Stagnation** | {live['stagnation']} gens |
| **Games** | {live['games']:,} |
| **Feature Candidates** | {live['feature_candidates']} |
| **Started** | {live['started_at'][:19]} |

**Brier Trend**: {brier_trend}
**Target**: Brier < 0.20 | ROI > 5% | Sharpe > 1.0
"""


def dash_logs():
    return "\n".join(live["log"][-100:])


def dash_top5():
    rows = []
    for i, ind in enumerate(live.get("top5", [])):
        f = ind.get("fitness", {})
        h = ind.get("hyperparams", {})
        rows.append(f"| #{i+1} | {f.get('brier', '?'):.4f} | {f.get('roi', 0):.1%} | "
                     f"{f.get('sharpe', 0):.2f} | {ind.get('n_features', '?')} | "
                     f"{h.get('model_type', '?')} | {f.get('composite', 0):.4f} |")
    if not rows:
        return "No data yet — evolution starting..."
    return f"""### Top 5 Individuals

| Rank | Brier | ROI | Sharpe | Features | Model | Composite |
|------|-------|-----|--------|----------|-------|-----------|
{chr(10).join(rows)}
"""


def dash_history():
    if not live["history"]:
        return "{}"
    return json.dumps(live["history"][-30:], indent=2)


def dash_api():
    latest = RESULTS_DIR / "evolution-latest.json"
    if latest.exists():
        return latest.read_text()
    return "{}"


# ═══════════════════════════════════════════════════════
# FASTAPI REMOTE CONTROL API (called by OpenClaw autonomously)
# ═══════════════════════════════════════════════════════

control_api = FastAPI()


@control_api.get("/api/status")
async def api_status():
    # Deep-convert numpy types to native Python for JSON serialization
    def _safe(obj):
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(x) for x in obj]
        if isinstance(obj, float):
            return round(obj, 6)
        return obj
    return JSONResponse(_safe(dict(live)))


@control_api.get("/api/results")
async def api_results():
    latest = RESULTS_DIR / "evolution-latest.json"
    if latest.exists():
        return JSONResponse(json.loads(latest.read_text()))
    return JSONResponse({"status": "no results yet"})


@control_api.post("/api/config")
async def api_config(request: Request):
    """Update GA parameters at runtime. OpenClaw calls this to apply improvements."""
    body = await request.json()
    allowed = {"pop_size", "mutation_rate", "target_features", "crossover_rate",
               "cooldown", "elite_size", "tournament_size"}
    updates = {k: v for k, v in body.items() if k in allowed}
    if not updates:
        return JSONResponse({"error": "no valid params", "allowed": list(allowed)}, status_code=400)
    remote_config["pending_params"].update(updates)
    log(f"[REMOTE] Config queued: {updates}")
    return JSONResponse({"status": "queued", "params": updates})


@control_api.post("/api/reset")
async def api_reset(request: Request):
    """Force population reset — keeps elites, replaces rest with fresh individuals."""
    remote_config["pending_reset"] = True
    log("[REMOTE] Population reset requested")
    return JSONResponse({"status": "reset queued"})


@control_api.post("/api/command")
async def api_command(request: Request):
    """Execute a command: diversify, boost_mutation, backfill_boxscores, etc."""
    body = await request.json()
    cmd = body.get("command", "")
    valid = ["diversify", "boost_mutation", "backfill_boxscores"]
    if cmd not in valid:
        return JSONResponse({"error": f"unknown command, valid: {valid}"}, status_code=400)

    if cmd == "backfill_boxscores":
        import threading
        def _run_backfill():
            try:
                import subprocess, sys
                log("[BACKFILL] Starting box score backfill...")
                live["backfill_progress"] = "running"
                result = subprocess.run(
                    [sys.executable, "ops/backfill-boxscores.py"],
                    capture_output=True, text=True, timeout=14400,  # 4h max
                    cwd=str(Path(__file__).parent.parent)
                )
                live["backfill_progress"] = "complete" if result.returncode == 0 else f"failed: {result.stderr[-200:]}"
                log(f"[BACKFILL] Done: rc={result.returncode}")
            except Exception as e:
                live["backfill_progress"] = f"error: {e}"
                log(f"[BACKFILL] Error: {e}")
        threading.Thread(target=_run_backfill, daemon=True).start()
        return JSONResponse({"status": "backfill started", "monitor": "/api/status → backfill_progress"})

    remote_config["commands"].append(cmd)
    log(f"[REMOTE] Command queued: {cmd}")
    return JSONResponse({"status": "command queued", "command": cmd})


@control_api.post("/api/checkpoint")
async def api_checkpoint(request: Request):
    """Save current best individual as a checkpoint."""
    global _checkpoints, _evo_best
    if not _evo_best:
        return JSONResponse({"error": "No evolution best available"}, status_code=400)

    cp = {
        "best": dict(_evo_best),
        "brier": _evo_best.get("fitness", {}).get("brier", 1.0),
        "generation": _evo_best.get("generation", 0),
        "timestamp": datetime.now().isoformat(),
    }
    _checkpoints.append(cp)
    if len(_checkpoints) > 10:
        _checkpoints = _checkpoints[-10:]

    # Persist to disk
    cp_file = CHECKPOINT_DIR / f"cp-{cp['generation']}-{int(time.time())}.json"
    cp_file.write_text(json.dumps(cp, indent=2, default=str))
    log(f"[CHECKPOINT] Saved gen {cp['generation']}, Brier {cp['brier']:.4f}")
    return JSONResponse({"status": "checkpoint saved", "brier": cp["brier"], "generation": cp["generation"]})


@control_api.post("/api/rollback")
async def api_rollback(request: Request):
    """Rollback to a specific checkpoint (by generation) or best checkpoint."""
    global _evo_best, _checkpoints
    body = await request.json()
    target_gen = body.get("generation")

    if not _checkpoints:
        return JSONResponse({"error": "No checkpoints available"}, status_code=400)

    if target_gen is not None:
        cp = next((c for c in _checkpoints if c["generation"] == target_gen), None)
        if not cp:
            return JSONResponse({"error": f"No checkpoint for gen {target_gen}"}, status_code=404)
    else:
        # Rollback to best (lowest Brier)
        cp = min(_checkpoints, key=lambda c: c["brier"])

    _evo_best = cp["best"]
    log(f"[ROLLBACK] Reverted to gen {cp['generation']}, Brier {cp['brier']:.4f}")
    return JSONResponse({"status": "rolled back", "brier": cp["brier"], "generation": cp["generation"]})


@control_api.get("/api/checkpoint/best")
async def api_checkpoint_best(request: Request):
    """Return the checkpoint with lowest Brier score."""
    if not _checkpoints:
        return JSONResponse({"error": "No checkpoints"}, status_code=404)
    best_cp = min(_checkpoints, key=lambda c: c["brier"])
    return JSONResponse({
        "brier": best_cp["brier"],
        "generation": best_cp["generation"],
        "timestamp": best_cp["timestamp"],
        "total_checkpoints": len(_checkpoints),
    })


@control_api.post("/api/inject-features")
async def api_inject_features(request: Request):
    """Receive feature ideas from OpenClaw's research. Stored for logging/tracking."""
    body = await request.json()
    features = body.get("features", [])
    if not features:
        return JSONResponse({"error": "no features provided"}, status_code=400)
    remote_config["injected_features"].extend(features)
    if len(remote_config["injected_features"]) > 200:
        remote_config["injected_features"] = remote_config["injected_features"][-200:]
    log(f"[REMOTE] {len(features)} feature ideas injected: {[f.get('name', '?') for f in features[:5]]}")
    return JSONResponse({"status": "injected", "count": len(features),
                         "total_pool": len(remote_config["injected_features"])})


@control_api.get("/api/remote-log")
async def api_remote_log():
    """Return recent remote commands and injections."""
    return JSONResponse({
        "pending_params": remote_config["pending_params"],
        "pending_reset": remote_config["pending_reset"],
        "queued_commands": remote_config["commands"],
        "injected_features_count": len(remote_config["injected_features"]),
        "recent_features": remote_config["injected_features"][-10:],
        "log_tail": live["log"][-30:],
    })


# ═══════════════════════════════════════════════════════
# RUN LOGGER API — Supabase monitoring endpoints
# ═══════════════════════════════════════════════════════

# Global logger ref (set from evolution_loop thread)
_global_logger = None


@control_api.get("/api/run-stats")
async def api_run_stats():
    """Evolution run statistics from Supabase."""
    if not _global_logger:
        return JSONResponse({"error": "logger not initialized"}, status_code=503)
    try:
        stats = _global_logger.get_stats()
        return JSONResponse(stats)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@control_api.get("/api/cuts")
async def api_cuts():
    """Recent auto-cut events."""
    if not _global_logger:
        return JSONResponse({"error": "logger not initialized"}, status_code=503)
    try:
        cuts = _global_logger.get_recent_cuts(20)
        return JSONResponse({"cuts": [list(c) for c in cuts] if cuts else []})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@control_api.get("/api/brier-trend")
async def api_brier_trend():
    """Brier score trend (last 50 generations)."""
    if not _global_logger:
        return JSONResponse({"error": "logger not initialized"}, status_code=503)
    try:
        trend = _global_logger.get_brier_trend(50)
        return JSONResponse({"trend": trend})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@control_api.get("/api/recent-runs")
async def api_recent_runs():
    """Recent cycle summaries from Supabase."""
    if not _global_logger:
        return JSONResponse({"error": "logger not initialized"}, status_code=503)
    try:
        runs = _global_logger.get_recent_runs(20)
        return JSONResponse({"runs": [list(r) for r in runs] if runs else []})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════
# EXPERIMENT SUBMISSION (works on both S10 and S11)
# ═══════════════════════════════════════════════════════

@control_api.post("/api/experiment/submit")
async def api_experiment_submit(request: Request):
    """Submit an experiment to the Supabase queue for S11 evaluation.

    Works on both S10 and S11 — writes to Supabase, S11 picks up.
    Body: {
        "experiment_id": "optional-custom-id",
        "agent_name": "eve|crew-research|crew-feature|manual",
        "experiment_type": "feature_test|model_test|calibration_test|config_change",
        "description": "What this experiment tests",
        "hypothesis": "Expected outcome",
        "params": { ... },
        "priority": 5,
        "baseline_brier": 0.2205
    }
    """
    try:
        body = await request.json()
        valid_types = ["feature_test", "model_test", "calibration_test", "config_change"]
        exp_type = body.get("experiment_type")
        if exp_type not in valid_types:
            return JSONResponse({"error": f"Invalid experiment_type. Valid: {valid_types}"}, status_code=400)
        if not body.get("description"):
            return JSONResponse({"error": "description is required"}, status_code=400)

        exp_id_str = body.get("experiment_id", f"submit-{int(time.time())}")
        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            return JSONResponse({"error": "DATABASE_URL not configured"}, status_code=503)

        import psycopg2
        conn = psycopg2.connect(db_url, connect_timeout=10, options="-c search_path=public")
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO public.nba_experiments
            (experiment_id, agent_name, experiment_type, description, hypothesis,
             params, priority, status, target_space, baseline_brier)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s)
            RETURNING id
        """, (
            exp_id_str,
            body.get("agent_name", "unknown"),
            exp_type,
            body["description"],
            body.get("hypothesis", ""),
            json.dumps(body.get("params", {})),
            body.get("priority", 5),
            body.get("target_space", "S11"),
            body.get("baseline_brier"),
        ))
        db_id = cur.fetchone()[0]
        conn.commit()
        conn.close()

        log(f"[EXPERIMENT] Queued: {exp_id_str} (type={exp_type}, agent={body.get('agent_name', 'unknown')})")
        return JSONResponse({
            "status": "queued",
            "id": db_id,
            "experiment_id": exp_id_str,
            "message": "Experiment queued for S11. It polls every 60s.",
        })
    except Exception as e:
        return JSONResponse({"error": f"Failed to queue experiment: {str(e)[:300]}"}, status_code=500)


# ═══════════════════════════════════════════════════════
# PREDICT API — Use evolved model for live predictions
# ═══════════════════════════════════════════════════════

@control_api.post("/api/predict")
async def api_predict(request: Request):
    """Generate predictions using the best evolved individual.

    Body: {"games": [{"home_team": "Boston Celtics", "away_team": "Miami Heat"}, ...]}
    Or: {"date": "2026-03-18"} to predict all games on a date (fetches from ESPN schedule).

    Returns probabilities from the evolved model trained on ALL historical data.
    """
    global _evo_X, _evo_y, _evo_features, _evo_best, _evo_games

    if _evo_best is None or _evo_X is None:
        return JSONResponse({"error": "evolution not ready — model still loading"}, status_code=503)

    body = await request.json()
    games_to_predict = body.get("games", [])
    date_str = body.get("date")

    # If date provided, fetch today's schedule from ESPN
    if date_str and not games_to_predict:
        try:
            import urllib.request
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str.replace('-', '')}"
            req = urllib.request.Request(url, headers={"User-Agent": "NomosQuant/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            for ev in data.get("events", []):
                comps = ev.get("competitions", [{}])[0]
                teams = comps.get("competitors", [])
                if len(teams) == 2:
                    home = next((t for t in teams if t.get("homeAway") == "home"), None)
                    away = next((t for t in teams if t.get("homeAway") == "away"), None)
                    if home and away:
                        games_to_predict.append({
                            "home_team": home["team"]["displayName"],
                            "away_team": away["team"]["displayName"],
                            "game_id": ev.get("id"),
                            "status": comps.get("status", {}).get("type", {}).get("name", ""),
                        })
        except Exception as e:
            return JSONResponse({"error": f"ESPN fetch failed: {e}"}, status_code=502)

    if not games_to_predict:
        return JSONResponse({"error": "no games to predict — provide 'games' array or 'date'"}, status_code=400)

    try:
        # Get best individual's config
        best = _evo_best
        selected = [i for i, b in enumerate(best["features"]) if b]
        hp = best["hyperparams"]

        if len(selected) < 5:
            return JSONResponse({"error": "best individual has too few features"}, status_code=503)

        # Train model on ALL historical data using best individual's features + hyperparams
        X_train = np.nan_to_num(_evo_X[:, selected], nan=0.0, posinf=1e6, neginf=-1e6)
        y_train = _evo_y

        hp_build = dict(hp)
        hp_build["n_estimators"] = min(hp.get("n_estimators", 150), 200)
        hp_build["max_depth"] = min(hp.get("max_depth", 6), 8)

        if hp_build.get("model_type") == "stacking":
            # For stacking, use XGBoost as fallback for prediction
            hp_build["model_type"] = "xgboost"

        model = _build(hp_build)
        if model is None:
            return JSONResponse({"error": "model build failed"}, status_code=500)

        # Apply calibration if specified
        if hp_build.get("calibration", "none") != "none":
            model = CalibratedClassifierCV(model, method=hp_build["calibration"], cv=3)

        model.fit(X_train, y_train)

        # Build features for today's games
        # We append each game to historical data and build features for the last row
        selected_names = [_evo_features[i] for i in selected if i < len(_evo_features)]
        predictions = []

        for game in games_to_predict:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            h_abbr = resolve(home)
            a_abbr = resolve(away)

            if not h_abbr or not a_abbr:
                predictions.append({
                    "home_team": home, "away_team": away,
                    "error": "team not recognized"
                })
                continue

            # Create a synthetic game entry (no score yet — we just need features)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            synthetic_game = {
                "game_date": today,
                "home_team": home, "away_team": away,
                "home": {"team_name": home, "pts": 0},
                "away": {"team_name": away, "pts": 0},
            }

            # Build features for this game using historical context
            try:
                all_games = list(_evo_games) + [synthetic_game]
                X_all, y_all, fn_all = build_features(all_games)

                # Apply same variance filter
                if X_all.shape[1] != _evo_X.shape[1]:
                    # Feature count mismatch — use column mapping by name
                    fn_map = {name: i for i, name in enumerate(fn_all)}
                    game_vec = np.zeros(len(selected))
                    for j, si in enumerate(selected):
                        if si < len(_evo_features):
                            fname = _evo_features[si]
                            if fname in fn_map:
                                game_vec[j] = X_all[-1, fn_map[fname]]
                else:
                    game_vec = np.nan_to_num(X_all[-1, selected], nan=0.0, posinf=1e6, neginf=-1e6)

                prob = float(model.predict_proba(game_vec.reshape(1, -1))[0, 1])

                # Kelly criterion (25% fractional)
                if prob > 0.55:
                    edge = prob - 0.5
                    kelly = (edge / 0.5) * 0.25  # fractional Kelly
                else:
                    kelly = 0.0

                predictions.append({
                    "home_team": home, "away_team": away,
                    "home_win_prob": round(prob, 4),
                    "away_win_prob": round(1 - prob, 4),
                    "confidence": round(abs(prob - 0.5) * 2, 4),
                    "kelly_stake": round(kelly, 4),
                    "model_type": best["hyperparams"]["model_type"],
                    "features_used": best["n_features"],
                    "brier_cv": best["fitness"]["brier"],
                })
            except Exception as e:
                predictions.append({
                    "home_team": home, "away_team": away,
                    "error": f"feature build failed: {str(e)[:100]}"
                })

        return JSONResponse({
            "predictions": predictions,
            "model": {
                "type": best["hyperparams"]["model_type"],
                "generation": best["generation"],
                "brier_cv": best["fitness"]["brier"],
                "roi_cv": best["fitness"]["roi"],
                "features": best["n_features"],
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    except Exception as e:
        return JSONResponse({"error": f"prediction failed: {str(e)[:200]}"}, status_code=500)


with gr.Blocks(title="NOMOS NBA QUANT — Genetic Evolution", theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# NOMOS NBA QUANT AI — Real Genetic Evolution 24/7")
    gr.Markdown("*Population of 60 individuals evolving feature selection + hyperparameters. Multi-objective: Brier + ROI + Sharpe + Calibration.*")

    with gr.Row():
        with gr.Column(scale=2):
            status_md = gr.Markdown(dash_status)
            top5_md = gr.Markdown(dash_top5)
        with gr.Column(scale=3):
            logs_box = gr.Textbox(label="Evolution Logs", value=dash_logs, lines=30, max_lines=30)

    with gr.Row():
        with gr.Column():
            hist_json = gr.Code(label="History (last 30 gens)", value=dash_history, language="json")
        with gr.Column():
            api_json = gr.Code(label="Latest Results (API)", value=dash_api, language="json")

    with gr.Row():
        refresh_btn = gr.Button("Refresh All", variant="primary")
        refresh_btn.click(dash_status, outputs=status_md)
        refresh_btn.click(dash_top5, outputs=top5_md)
        refresh_btn.click(dash_logs, outputs=logs_box)
        refresh_btn.click(dash_history, outputs=hist_json)
        refresh_btn.click(dash_api, outputs=api_json)

    timer = gr.Timer(15)
    timer.tick(dash_status, outputs=status_md)
    timer.tick(dash_top5, outputs=top5_md)
    timer.tick(dash_logs, outputs=logs_box)


# ── Register experiment endpoints BEFORE Gradio mount (Gradio catch-all at / swallows late routes) ──
if os.environ.get("EXPERIMENT_MODE"):
    from experiment_runner import experiment_loop, register_experiment_endpoints
    register_experiment_endpoints(control_api)
    log("Experiment endpoints registered on control_api")

# ── Mount FastAPI control API onto Gradio's app ──
gr_app = gr.mount_gradio_app(control_api, app, path="/")

# ── Launch background threads ──
if os.environ.get("EXPERIMENT_MODE"):
    _thread = threading.Thread(target=experiment_loop, daemon=True, name="ExperimentRunner")
    _thread.start()
    log("S11 EXPERIMENT RUNNER thread launched (EXPERIMENT_MODE=true)")
else:
    _thread = threading.Thread(target=evolution_loop, daemon=True, name="GeneticEvolution")
    _thread.start()
    log("Genetic evolution thread launched")

if __name__ == "__main__":
    import uvicorn
    mode = "EXPERIMENT RUNNER" if os.environ.get("EXPERIMENT_MODE") else "EVOLUTION"
    log(f"Starting with FastAPI + Gradio ({mode} mode, remote control API enabled)")
    uvicorn.run(gr_app, host="0.0.0.0", port=7860)
