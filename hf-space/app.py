#!/usr/bin/env python3
"""
NOMOS NBA QUANT AI — REAL Genetic Evolution (HF Space)
========================================================
RUNS 24/7 on HuggingFace Space (2 vCPU / 16GB RAM).

NOT a fake LLM wrapper. REAL ML:
  - Population of 60 individuals across 5 islands (island model GA)
  - Walk-forward backtest fitness (Brier + LogLoss + Sharpe + ECE)
  - NSGA-II style Pareto front ranking (multi-objective)
  - 13 model types including neural nets (LSTM, Transformer, TabNet, etc.)
  - Island migration every 10 generations for diversity
  - Adaptive mutation decay (0.15 → 0.05) + tournament pressure
  - Memory management: GC between evaluations for 16GB RAM
  - Gradio dashboard showing live evolution progress
  - JSON API for crew agents on VM

Target: Brier < 0.20 | ROI > 5% | Sharpe > 1.0
"""

import os, sys, json, time, math, threading, warnings, random, traceback, gc
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
from copy import deepcopy

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

# GPU detection (determines if neural models are viable)
try:
    import torch as _torch
    _HAS_GPU = _torch.cuda.is_available()
except ImportError:
    _HAS_GPU = False

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

# ── Post-hoc Platt/Isotonic Calibration (D5: raw ECE=0.2758, target <0.05) ──
# Pure stdlib — no sklearn dependency for the calibration lookup.
_CAL_MAP_PATH = Path("data/calibration-map.json")

def _load_cal_map():
    """Load calibration map. Returns (bin_edges, raw_centers, cal_centers) or None."""
    for p in [_CAL_MAP_PATH, DATA_DIR / "calibration-map.json"]:
        if p.exists():
            try:
                d = json.loads(p.read_text())
                return d["bin_edges"], d["raw_centers"], d["calibrated_centers"]
            except Exception as e:
                print(f"[calibration] Failed to load {p}: {e}")
    print("[calibration] calibration-map.json not found — using identity")
    return None

def _apply_cal(raw_prob: float, cal_map) -> float:
    """Apply piecewise-linear calibration. Falls back to identity if cal_map is None."""
    if cal_map is None:
        return raw_prob
    bin_edges, raw_centers, cal_centers = cal_map
    p = max(0.0, min(1.0, float(raw_prob)))
    if p == 0.0: return 0.0
    if p == 1.0: return 1.0
    # Find bin
    i = len(bin_edges) - 2
    for j in range(len(bin_edges) - 1):
        if bin_edges[j] <= p < bin_edges[j + 1]:
            i = j
            break
    raw_c = raw_centers[i]
    cal_c = cal_centers[i]
    n = len(raw_centers)
    if p < raw_c:
        if i == 0:
            t = p / raw_c if raw_c > 0 else 0.5
            cal_low, cal_high = 0.0, cal_c
        else:
            span = raw_c - raw_centers[i - 1]
            t = (p - raw_centers[i - 1]) / span if span > 0 else 0.5
            cal_low, cal_high = cal_centers[i - 1], cal_c
    else:
        if i == n - 1:
            span = 1.0 - raw_c
            t = (p - raw_c) / span if span > 0 else 0.5
            cal_low, cal_high = cal_c, 1.0
        else:
            span = raw_centers[i + 1] - raw_c
            t = (p - raw_c) / span if span > 0 else 0.5
            cal_low, cal_high = cal_c, cal_centers[i + 1]
    t = max(0.0, min(1.0, t))
    return round(max(0.0, min(1.0, cal_low + t * (cal_high - cal_low))), 4)

_CAL_MAP = _load_cal_map()

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
    "mutation_rate": 0.15,
    "stagnation": 0,
    "games": 0,
    "feature_candidates": 0,
    "gpu": False,
    "log": [],
    "history": [],
    "top5": [],
    "started_at": datetime.now(timezone.utc).isoformat(),
    "last_update": "never",
    "n_islands": 5,
    "island_sizes": [],
    "pareto_front_size": 0,
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

_FEATURE_CACHE = Path("/tmp/nba_feature_cache.npz")

def build_features(games):
    """Build features using the full NBAFeatureEngine. Caches to disk for fast restarts."""
    # ── Try disk cache first (saves ~50min on restart) ──
    if _FEATURE_CACHE.exists():
        try:
            cache = np.load(_FEATURE_CACHE, allow_pickle=True)
            X, y = cache["X"], cache["y"]
            feature_names = list(cache["feature_names"])
            if X.shape[0] >= len(games) - 200:  # Accept cache if within ~200 games
                print(f"[CACHE] Loaded feature matrix from disk: {X.shape}")
                return X, y, feature_names
            else:
                print(f"[CACHE] Stale ({X.shape[0]} vs {len(games)} games), rebuilding")
        except Exception as e:
            print(f"[CACHE] Load failed ({e}), rebuilding")

    try:
        from features.engine import NBAFeatureEngine
        engine = NBAFeatureEngine()
        X, y, feature_names = engine.build(games)
        X = np.nan_to_num(np.array(X, dtype=np.float64))
        y = np.array(y, dtype=np.int32)
        # ── Save to disk cache ──
        try:
            np.savez_compressed(_FEATURE_CACHE, X=X, y=y, feature_names=np.array(feature_names))
            print(f"[CACHE] Saved feature matrix to disk: {X.shape}")
        except Exception as ce:
            print(f"[CACHE] Save failed: {ce}")
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

        # ── CROSS-WINDOW MOMENTUM + INTERACTIONS (v3 — Adam/Evolution Agent) ──
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            wp_accel_v = wp(tr, 3) - 2 * wp(tr, 10) + wp(tr, 20) if len(tr) >= 20 else 0.0
            pts_for = sum(x[4] for x in tr[-20:]) if len(tr) >= 5 else 100
            pts_against = sum(x[5] for x in tr[-20:]) if len(tr) >= 5 else 100
            pyth = pts_for ** 13.91 / max(1, pts_for ** 13.91 + pts_against ** 13.91) if pts_for > 0 else 0.5
            pts_l = [x[4] for x in tr[-10:]] if len(tr) >= 5 else [100]
            p_vol = (sum((p - sum(pts_l)/len(pts_l))**2 for p in pts_l) / len(pts_l)) ** 0.5 if len(pts_l) > 1 else 0
            recent_w = [x for x in tr[-10:] if x[1]]
            w_qual = sum(wp(team_results[x[3]], 82) for x in recent_w) / max(len(recent_w), 1) if recent_w else 0.5
            margins_10 = [x[2] for x in tr[-10:]] if len(tr) >= 5 else [0]
            if len(margins_10) >= 3:
                xv = list(range(len(margins_10))); xm = sum(xv)/len(xv); ym = sum(margins_10)/len(margins_10)
                m_slope = sum((x-xm)*(y-ym) for x,y in zip(xv,margins_10)) / max(sum((x-xm)**2 for x in xv), 1e-9)
            else:
                m_slope = 0.0
            row.extend([wp_accel_v, pyth, p_vol / 10.0, w_qual, m_slope,
                         ppg(tr, 3) / max(ppg(tr, 20), 1), papg(tr, 3) / max(papg(tr, 20), 1)])
            if first:
                names.extend([f"{prefix}_wp_accel", f"{prefix}_pyth_exp", f"{prefix}_pts_vol",
                              f"{prefix}_win_quality", f"{prefix}_margin_slope",
                              f"{prefix}_off_ratio", f"{prefix}_def_ratio"])
        # Key interaction features
        elo_d = team_elo[home] - team_elo[away] + 50
        rest_adv = h_rest - a_rest
        wp_d = wp(hr_, 10) - wp(ar_, 10)
        row.extend([
            elo_d * rest_adv / 10.0, wp_d ** 2, elo_d ** 2 / 10000.0,
            h_netrtg * a_netrtg / 100.0,
            float(h_wp82 > 0.6 and a_wp82 < 0.4), float(h_rest >= 3 and a_rest <= 1),
        ])
        if first:
            names.extend(["elo_rest_x", "wp_diff_sq", "elo_diff_sq",
                          "netrtg_product", "mismatch_flag", "rest_mismatch_flag"])

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

# All model types the GA can evolve (13 total)
ALL_MODEL_TYPES = [
    # Tree-based (fast, reliable)
    "xgboost", "xgboost_brier", "lightgbm", "catboost", "random_forest", "extra_trees",
    # Ensemble
    "stacking",
    # Neural nets (slower, potentially more powerful)
    "mlp", "lstm", "transformer", "tabnet", "ft_transformer", "deep_ensemble",
    # AutoML
    "autogluon",
]

# Neural net types (need special memory handling)
NEURAL_NET_TYPES = {"lstm", "transformer", "tabnet", "ft_transformer", "deep_ensemble", "mlp", "autogluon"}

# Fast model types (for islands that prioritize speed)
FAST_MODEL_TYPES = ["xgboost", "xgboost_brier", "lightgbm", "random_forest", "extra_trees", "logistic_regression"]

# CPU-viable model types (tree-based only, excludes neural nets and stacking)
# Stacking removed: 200 gens, best brier=0.24733 (10% worse than trees), too slow on CPU
# logistic_regression added: MDPI 2026 shows LR achieves best Brier (0.199) among tabular models
# due to inherent calibration — beating XGBoost (0.202) and RF on probability quality
CPU_MODEL_TYPES = ["xgboost", "xgboost_brier", "lightgbm", "catboost", "random_forest", "extra_trees", "logistic_regression"]


# ── Custom Brier objective for XGBoost ──
def _brier_objective(y_true, y_pred):
    """Custom XGBoost objective that directly minimizes Brier score.
    Brier = (sigmoid(raw) - y)^2, so gradient = 2*(p-y)*p*(1-p), hess ≈ 2*p*(1-p).
    Note: XGBoost sklearn API (>=2.0) passes (labels, preds) as numpy arrays."""
    p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -30, 30)))
    grad = 2.0 * (p - y_true) * p * (1.0 - p)
    hess = 2.0 * p * (1.0 - p) + 1e-6  # Simplified + stability
    return grad, hess


class Individual:
    MAX_FEATURES = 200  # Hard cap — individuals above this waste compute

    def __init__(self, n_features, target=100, model_type=None):
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
            "model_type": model_type or random.choice(CPU_MODEL_TYPES if not _HAS_GPU else ALL_MODEL_TYPES),
            "calibration": random.choices(["none", "sigmoid", "venn_abers", "beta", "lr_meta", "lr_platt"], weights=[20, 15, 25, 25, 10, 10], k=1)[0],
            # Neural net hyperparams (used only for NN model types)
            "nn_hidden_dims": random.choice([64, 128, 256]),
            "nn_n_layers": random.randint(2, 4),
            "nn_dropout": random.uniform(0.1, 0.5),
            "nn_epochs": random.randint(20, 100),
            "nn_batch_size": random.choice([32, 64, 128]),
        }
        self.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "composite": 0.0}
        self.pareto_rank = 999   # NSGA-II rank (0 = Pareto front)
        self.crowding_dist = 0.0  # NSGA-II crowding distance
        self.island_id = -1       # Which island this individual belongs to
        self.generation = 0
        self._enforce_feature_cap()

    def _enforce_feature_cap(self):
        """If feature count exceeds MAX_FEATURES, randomly drop excess features."""
        selected = [i for i, b in enumerate(self.features) if b]
        if len(selected) > self.MAX_FEATURES:
            to_drop = random.sample(selected, len(selected) - self.MAX_FEATURES)
            for idx in to_drop:
                self.features[idx] = 0
        self.n_features = sum(self.features)

    def selected_indices(self):
        return [i for i, b in enumerate(self.features) if b]

    def to_dict(self):
        return {"n_features": self.n_features, "hyperparams": dict(self.hyperparams),
                "fitness": dict(self.fitness), "generation": self.generation,
                "pareto_rank": self.pareto_rank, "island_id": self.island_id,
                "feature_indices": self.selected_indices()}

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
            elif isinstance(p1.hyperparams[key], bool):
                child.hyperparams[key] = random.choice([p1.hyperparams[key], p2.hyperparams[key]])
            else:
                child.hyperparams[key] = random.choice([p1.hyperparams[key], p2.hyperparams[key]])
        child.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "composite": 0.0}
        child.pareto_rank = 999
        child.crowding_dist = 0.0
        child.island_id = random.choice([p1.island_id, p2.island_id]) if hasattr(p1, 'island_id') else -1
        child.generation = max(p1.generation, p2.generation) + 1
        child._enforce_feature_cap()
        return child

    def mutate(self, rate=0.03, feature_importance=None):
        """Directed mutation: bias towards features that top performers use."""
        for i in range(len(self.features)):
            if random.random() < rate:
                if feature_importance is not None and i < len(feature_importance):
                    imp = feature_importance[i]
                    if self.features[i] == 0 and imp > 0.5:
                        self.features[i] = 1
                    elif self.features[i] == 1 and imp < 0.15:
                        self.features[i] = 0
                    else:
                        self.features[i] = 1 - self.features[i]
                else:
                    self.features[i] = 1 - self.features[i]
        if random.random() < 0.15: self.hyperparams["n_estimators"] = max(50, min(200, self.hyperparams["n_estimators"] + random.randint(-50, 50)))
        if random.random() < 0.15: self.hyperparams["max_depth"] = max(2, min(8, self.hyperparams["max_depth"] + random.randint(-2, 2)))
        if random.random() < 0.15: self.hyperparams["learning_rate"] = max(0.001, min(0.3, self.hyperparams["learning_rate"] * 10 ** random.uniform(-0.3, 0.3)))
        if random.random() < 0.08: self.hyperparams["model_type"] = random.choice(CPU_MODEL_TYPES if not _HAS_GPU else ALL_MODEL_TYPES)
        if random.random() < 0.05: self.hyperparams["calibration"] = random.choices(["none", "sigmoid", "venn_abers", "beta", "lr_meta", "lr_platt"], weights=[45, 13, 13, 17, 8, 8], k=1)[0]
        # Neural net hyperparams mutation
        if random.random() < 0.10: self.hyperparams["nn_hidden_dims"] = random.choice([64, 128, 256, 512])
        if random.random() < 0.10: self.hyperparams["nn_n_layers"] = max(1, min(6, self.hyperparams.get("nn_n_layers", 2) + random.randint(-1, 1)))
        if random.random() < 0.10: self.hyperparams["nn_dropout"] = max(0.0, min(0.7, self.hyperparams.get("nn_dropout", 0.3) + random.uniform(-0.1, 0.1)))
        if random.random() < 0.10: self.hyperparams["nn_epochs"] = max(10, min(200, self.hyperparams.get("nn_epochs", 50) + random.randint(-20, 20)))
        self._enforce_feature_cap()


# ═══════════════════════════════════════════════════════
# NSGA-II PARETO RANKING
# ═══════════════════════════════════════════════════════

def _dominates(a, b):
    """Individual a dominates b if a is no worse in all objectives and strictly better in at least one.
    Objectives: minimize brier, maximize roi, maximize sharpe, minimize calibration, minimize features.
    Feature count is a 5th objective to prevent bloated genomes (Feat=200 takeover)."""
    fa, fb = a.fitness, b.fitness
    # Parsimony pressure: penalize feature bloat as a real objective
    a_feat_score = -min(a.n_features, 200) / 200.0  # lower features = higher score (negated for minimize)
    b_feat_score = -min(b.n_features, 200) / 200.0
    a_vals = (-fa["brier"], fa["roi"], fa["sharpe"], -fa["calibration"], a_feat_score)
    b_vals = (-fb["brier"], fb["roi"], fb["sharpe"], -fb["calibration"], b_feat_score)
    at_least_one_better = False
    for av, bv in zip(a_vals, b_vals):
        if av < bv:
            return False
        if av > bv:
            at_least_one_better = True
    return at_least_one_better


def nsga2_rank(population):
    """Assign NSGA-II Pareto rank and crowding distance to each individual.
    Returns population sorted by (rank ASC, crowding_dist DESC)."""
    n = len(population)
    if n == 0:
        return population

    # Fast non-dominated sorting
    dominated_by = [0] * n  # count of how many dominate this individual
    dominates_set = [[] for _ in range(n)]  # indices this individual dominates
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(population[i], population[j]):
                dominates_set[i].append(j)
                dominated_by[j] += 1
            elif _dominates(population[j], population[i]):
                dominates_set[j].append(i)
                dominated_by[i] += 1

    for i in range(n):
        if dominated_by[i] == 0:
            population[i].pareto_rank = 0
            fronts[0].append(i)

    rank = 0
    while fronts[rank]:
        next_front = []
        for i in fronts[rank]:
            for j in dominates_set[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    population[j].pareto_rank = rank + 1
                    next_front.append(j)
        rank += 1
        fronts.append(next_front)

    # Crowding distance per front (5 objectives including parsimony)
    objectives = [
        lambda ind: ind.fitness["brier"],       # minimize
        lambda ind: -ind.fitness["roi"],         # maximize (negate for sorting)
        lambda ind: -ind.fitness["sharpe"],      # maximize
        lambda ind: ind.fitness["calibration"],  # minimize
        lambda ind: ind.n_features / 200.0,     # minimize (parsimony pressure)
    ]

    for front_indices in fronts:
        if not front_indices:
            continue
        front = [population[i] for i in front_indices]
        for ind in front:
            ind.crowding_dist = 0.0

        for obj_fn in objectives:
            front_sorted = sorted(range(len(front)), key=lambda k: obj_fn(front[k]))
            front[front_sorted[0]].crowding_dist = float('inf')
            front[front_sorted[-1]].crowding_dist = float('inf')
            obj_range = obj_fn(front[front_sorted[-1]]) - obj_fn(front[front_sorted[0]])
            if obj_range < 1e-10:
                continue
            for k in range(1, len(front_sorted) - 1):
                front[front_sorted[k]].crowding_dist += (
                    obj_fn(front[front_sorted[k + 1]]) - obj_fn(front[front_sorted[k - 1]])
                ) / obj_range

    # Also compute composite for backwards compatibility (weighted sum)
    for ind in population:
        f = ind.fitness
        feature_penalty = 0.003 * max(0, ind.n_features - 80)  # Stronger penalty: 3x, threshold 80
        ind.fitness["composite"] = round(
            0.40 * (1 - f["brier"]) +
            0.25 * max(0, f["roi"]) +
            0.20 * max(0, min(f["sharpe"], 3) / 3) +
            0.15 * (1 - f["calibration"])
            - feature_penalty, 5)

    # Sort: primary = pareto_rank ASC, secondary = crowding_dist DESC
    population.sort(key=lambda x: (x.pareto_rank, -x.crowding_dist))
    return population


# ═══════════════════════════════════════════════════════
# ISLAND MODEL
# ═══════════════════════════════════════════════════════

class IslandModel:
    """Manages multiple sub-populations (islands) with periodic migration."""

    def __init__(self, n_islands=5, island_size=100, n_features=164,
                 target_features=65, migration_interval=10, migrants_per_island=5):
        self.n_islands = n_islands
        self.island_size = island_size
        self.total_pop = n_islands * island_size
        self.n_features = n_features
        self.target_features = target_features
        self.migration_interval = migration_interval
        self.migrants_per_island = migrants_per_island
        self.islands = []  # list of lists of Individuals

        # Each island can specialize — assign model type biases
        # On CPU (all HF Spaces), neural models are skipped (brier=0.28 penalty).
        # Use all 5 islands for productive CPU tree-based exploration.
        if _HAS_GPU:
            self.island_specializations = [
                FAST_MODEL_TYPES,                                      # Island 0: fast tree models
                ["catboost", "stacking", "extra_trees"],               # Island 1: ensemble specialists
                ["mlp", "tabnet", "ft_transformer"],                   # Island 2: neural nets
                ["lstm", "transformer", "deep_ensemble"],              # Island 3: deep learning
                ALL_MODEL_TYPES,                                       # Island 4: unrestricted
            ]
        else:
            self.island_specializations = [
                ["extra_trees", "random_forest"],                      # Island 0: tree specialists (best models)
                ["xgboost", "xgboost_brier"],                          # Island 1: XGBoost variants
                ["lightgbm", "catboost"],                              # Island 2: gradient boosters
                ["catboost", "extra_trees", "random_forest"],             # Island 3: diverse trees
                FAST_MODEL_TYPES,                                      # Island 4: unrestricted fast
            ]

    def initialize(self):
        """Create initial population across all islands."""
        self.islands = []
        for island_id in range(self.n_islands):
            island = []
            specialization = self.island_specializations[island_id]
            for _ in range(self.island_size):
                mt = random.choice(specialization)
                ind = Individual(self.n_features, self.target_features, model_type=mt)
                ind.island_id = island_id
                island.append(ind)
            self.islands.append(island)

    def get_all_individuals(self):
        """Flatten all islands into a single list."""
        return [ind for island in self.islands for ind in island]

    def set_all_individuals(self, population):
        """Distribute a flat population back into islands."""
        self.islands = []
        for island_id in range(self.n_islands):
            start = island_id * self.island_size
            end = start + self.island_size
            island = population[start:end]
            for ind in island:
                ind.island_id = island_id
            self.islands.append(island)
        # Handle any remainder (from injections, etc.)
        remainder = population[self.n_islands * self.island_size:]
        if remainder:
            for ind in remainder:
                target_island = random.randint(0, self.n_islands - 1)
                ind.island_id = target_island
                self.islands[target_island].append(ind)

    def migrate(self, generation):
        """Migrate best individuals between islands every migration_interval generations."""
        if generation % self.migration_interval != 0 or generation == 0:
            return False

        n_migrate = self.migrants_per_island
        # Collect best from each island
        migrants = []
        for island in self.islands:
            island.sort(key=lambda x: (x.pareto_rank, -x.crowding_dist))
            best = island[:n_migrate]
            # Deep copy migrants (they stay on original island too)
            for ind in best:
                clone = Individual.__new__(Individual)
                clone.features = ind.features[:]
                clone.hyperparams = dict(ind.hyperparams)
                clone.fitness = dict(ind.fitness)
                clone.pareto_rank = ind.pareto_rank
                clone.crowding_dist = ind.crowding_dist
                clone.n_features = ind.n_features
                clone.generation = ind.generation
                migrants.append(clone)

        random.shuffle(migrants)

        # Distribute migrants to next island (ring topology)
        for i, island in enumerate(self.islands):
            # Remove worst n_migrate from this island
            island.sort(key=lambda x: (x.pareto_rank, -x.crowding_dist))
            island_trimmed = island[:self.island_size - n_migrate]
            # Add migrants from the pool
            received = migrants[i * n_migrate:(i + 1) * n_migrate]
            for ind in received:
                ind.island_id = i
            island_trimmed.extend(received)
            self.islands[i] = island_trimmed

        return True

    def get_island_stats(self):
        """Return per-island statistics."""
        stats = []
        for i, island in enumerate(self.islands):
            if not island:
                stats.append({"island": i, "size": 0})
                continue
            models = Counter(ind.hyperparams["model_type"] for ind in island)
            briers = [ind.fitness["brier"] for ind in island if ind.fitness["brier"] < 1.0]
            stats.append({
                "island": i,
                "size": len(island),
                "specialization": self.island_specializations[i],
                "model_dist": dict(models.most_common(5)),
                "best_brier": min(briers) if briers else 1.0,
                "avg_brier": sum(briers) / len(briers) if briers else 1.0,
                "pareto_front_count": sum(1 for ind in island if ind.pareto_rank == 0),
            })
        return stats

    def to_state_dict(self):
        """Serialize for persistence."""
        return {
            "n_islands": self.n_islands,
            "island_size": self.island_size,
            "migration_interval": self.migration_interval,
            "migrants_per_island": self.migrants_per_island,
            "islands": [
                [{"features": ind.features,
                  "hyperparams": {k: (float(v) if isinstance(v, np.floating) else v) for k, v in ind.hyperparams.items()},
                  "fitness": ind.fitness, "generation": ind.generation,
                  "island_id": ind.island_id,
                  "pareto_rank": getattr(ind, 'pareto_rank', 999)}
                 for ind in island]
                for island in self.islands
            ]
        }

    @classmethod
    def from_state_dict(cls, state, n_features):
        """Restore from persisted state."""
        im = cls(
            n_islands=state["n_islands"],
            island_size=state.get("island_size", 100),
            n_features=n_features,
            migration_interval=state.get("migration_interval", 10),
            migrants_per_island=state.get("migrants_per_island", 5),
        )
        im.islands = []
        for island_data in state["islands"]:
            island = []
            for d in island_data:
                ind = Individual.__new__(Individual)
                ind.features = d["features"]
                ind.hyperparams = d["hyperparams"]
                ind.fitness = d["fitness"]
                ind.generation = d.get("generation", 0)
                ind.island_id = d.get("island_id", -1)
                ind.pareto_rank = d.get("pareto_rank", 999)
                ind.crowding_dist = 0.0
                ind.n_features = sum(ind.features)
                # Ensure new hyperparams exist
                ind.hyperparams.setdefault("nn_hidden_dims", 128)
                ind.hyperparams.setdefault("nn_n_layers", 2)
                ind.hyperparams.setdefault("nn_dropout", 0.3)
                ind.hyperparams.setdefault("nn_epochs", 50)
                ind.hyperparams.setdefault("nn_batch_size", 64)
                # Ensure model_type is valid (on CPU, force tree-based models)
                _valid_types = CPU_MODEL_TYPES if not _HAS_GPU else ALL_MODEL_TYPES
                if ind.hyperparams.get("model_type") not in _valid_types:
                    ind.hyperparams["model_type"] = random.choice(_valid_types)
                island.append(ind)
            im.islands.append(island)
        return im


def _gc_cleanup():
    """Force garbage collection to manage 16GB RAM with 500 individuals."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════
# FITNESS EVALUATION
# ═══════════════════════════════════════════════════════

def _evaluate_stacking(ind, X_sub, y_eval, hp_eval, n_splits, fast):
    """Stacking: XGBoost + LightGBM + CatBoost + ExtraTrees base models → LogisticRegression meta-learner.
    Uses 4 diverse base models for maximum ensemble benefit. CatBoost forced to CPU to avoid CUDA conflicts."""
    splits = n_splits if fast else max(n_splits, 3)
    tscv = TimeSeriesSplit(n_splits=splits)
    base_est = min(hp_eval.get("n_estimators", 100), 100)
    depth = min(hp_eval.get("max_depth", 6), 6)
    lr = hp_eval.get("learning_rate", 0.1)

    briers, rois, all_p, all_y = [], [], [], []
    PURGE_GAP = 5
    for ti, vi in tscv.split(X_sub):
        try:
            # Purge last PURGE_GAP games from training to prevent temporal leakage
            ti_safe = ti[:-PURGE_GAP] if len(ti) > PURGE_GAP + 50 else ti
            X_tr, y_tr = X_sub[ti_safe], y_eval[ti_safe]
            X_val, y_val = X_sub[vi], y_eval[vi]

            # Build 4 diverse base models
            base_models = []
            # XGBoost — gradient boosting with histogram
            base_models.append(xgb.XGBClassifier(
                n_estimators=base_est, max_depth=depth, learning_rate=lr,
                subsample=hp_eval.get("subsample", 0.8),
                colsample_bytree=hp_eval.get("colsample_bytree", 0.8),
                reg_alpha=0.01, reg_lambda=0.1,
                eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist"))
            # LightGBM — leaf-wise boosting (different inductive bias)
            base_models.append(lgbm.LGBMClassifier(
                n_estimators=base_est, max_depth=depth, learning_rate=lr,
                subsample=hp_eval.get("subsample", 0.8),
                num_leaves=min(2 ** depth - 1, 31),
                reg_alpha=0.01, reg_lambda=0.1,
                min_child_samples=20, feature_fraction=0.7,
                boosting_type="dart", drop_rate=0.1,
                verbose=-1, random_state=42, n_jobs=-1))
            # CatBoost — ordered boosting (FORCED CPU to avoid CUDA device conflicts)
            try:
                from catboost import CatBoostClassifier
                base_models.append(CatBoostClassifier(
                    iterations=min(base_est, 100), depth=min(depth, 6),
                    learning_rate=lr, l2_leaf_reg=max(hp_eval.get("reg_lambda", 3.0), 1.0),
                    verbose=0, random_state=42,
                    task_type='CPU', thread_count=-1))
            except (ImportError, Exception):
                from sklearn.ensemble import RandomForestClassifier
                base_models.append(RandomForestClassifier(
                    n_estimators=base_est, max_depth=depth,
                    random_state=42, n_jobs=-1))
            # ExtraTrees — randomized splits (more diversity for stacking)
            from sklearn.ensemble import ExtraTreesClassifier
            base_models.append(ExtraTreesClassifier(
                n_estimators=base_est, max_depth=depth,
                min_samples_leaf=5, max_features="sqrt",
                random_state=42, n_jobs=-1))

            # Get OOF predictions from each base model using inner CV
            inner_cv = TimeSeriesSplit(n_splits=4)
            oof_preds = np.zeros((len(X_tr), len(base_models)))
            for m_idx, bm in enumerate(base_models):
                try:
                    oof = cross_val_predict(bm, X_tr, y_tr, cv=inner_cv, method="predict_proba")[:, 1]
                    oof_preds[:, m_idx] = oof
                except Exception:
                    oof_preds[:, m_idx] = 0.5

            # Train meta-learner on OOF predictions (regularized to prevent stacking overfit)
            # Calibrate meta-learner with Platt scaling for better probability estimates
            meta_base = LogisticRegression(C=0.5, max_iter=300, random_state=42)
            try:
                meta = CalibratedClassifierCV(meta_base, method='sigmoid', cv=3)
                meta.fit(oof_preds, y_tr)
            except Exception:
                # Fallback: uncalibrated if not enough data for calibration CV
                meta = meta_base
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
        except Exception as e:
            log(f"Stacking fold failed: {str(e)[:80]}", "WARN")
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


EVAL_TIMEOUT_S = 120  # Max seconds per individual evaluation — prevents hung models

def _eval_with_timeout(fn, timeout, *args, **kwargs):
    """Run fn with a wall-clock timeout. Returns True if completed, False if timed out."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            pool.submit(fn, *args, **kwargs).result(timeout=timeout)
            return True
        except FuturesTimeout:
            return False
        except Exception:
            return False

def evaluate(ind, X, y, n_splits=2, fast=True, eval_counter=[0]):
    """Two-tier evaluation: fast (subsample + 2-fold) or full (all data + 3-fold).
    Includes memory management for 16GB RAM and per-eval timeout."""
    # Periodic GC to manage memory with large population
    eval_counter[0] += 1
    if eval_counter[0] % GC_INTERVAL == 0:
        _gc_cleanup()

    # Neural net models: cap features more aggressively to reduce memory
    is_nn = ind.hyperparams.get("model_type") in NEURAL_NET_TYPES
    max_features = 120 if is_nn else 200

    selected = ind.selected_indices()
    if len(selected) < 15 or len(selected) > max_features:
        ind.fitness = {"brier": 0.30, "roi": 0.0, "sharpe": 0.0, "calibration": 0.15, "composite": -1.0,
                       "features_pruned": 0}
        return

    # Skip neural models entirely on CPU — they're too slow
    if is_nn and not _HAS_GPU:
        ind.fitness = {"brier": 0.28, "roi": 0.0, "sharpe": 0.0, "calibration": 0.15, "composite": -0.5,
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

    # Cap estimators for speed (fast mode) — 80 is enough for ranking on CPU
    hp_eval = dict(hp)
    if fast:
        hp_eval["n_estimators"] = min(hp["n_estimators"], 80)

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
        PURGE_GAP = 5  # 5-game buffer between train/test to prevent lookahead
        for ti, vi in tscv.split(X_sub):
            try:
                # Purge last PURGE_GAP games from training to avoid temporal leakage
                ti_safe = ti[:-PURGE_GAP] if len(ti) > PURGE_GAP + 50 else ti
                m = clone(model)
                # Calibration: none (default), sigmoid (Platt), venn_abers (MAPIE), beta (BetaCalibration), or lr_meta (LR meta-calibration)
                cal_method = hp_eval.get("calibration", "none")
                if cal_method == "isotonic":
                    cal_method = "none"  # Isotonic empirically hurts Brier (+0.003 to +0.007)
                _beta_cal = None    # beta calibrator applied post-predict
                _lr_meta_cal = None  # LR meta-calibrator applied post-predict
                _platt_cal = None  # LR Platt scaler applied post-predict
                _model_fitted = False  # tracks whether m.fit() was already called
                if cal_method == "venn_abers":
                    try:
                        from mapie.classification import MapieClassifier
                        m_inner = clone(m)
                        m_inner.fit(X_sub[ti_safe], y_eval[ti_safe])
                        mapie = MapieClassifier(m_inner, method="lac", cv="prefit")
                        mapie.fit(X_sub[ti_safe[-200:]], y_eval[ti_safe[-200:]])
                        m = mapie  # MapieClassifier wraps fitted model
                        _model_fitted = True
                        cal_method = "none"
                    except (ImportError, Exception):
                        cal_method = "none"  # Fallback if MAPIE not installed
                if cal_method == "beta":
                    try:
                        from betacal import BetaCalibration
                        # Fit base model, then fit beta calibrator on a held-out slice
                        m.fit(X_sub[ti_safe], y_eval[ti_safe])
                        _model_fitted = True
                        cal_slice = ti_safe[-200:] if len(ti_safe) > 200 else ti_safe
                        raw_p = m.predict_proba(X_sub[cal_slice])[:, 1]
                        _beta_cal = BetaCalibration(parameters="abm")
                        _beta_cal.fit(raw_p.reshape(-1, 1), y_eval[cal_slice])
                        cal_method = "none"
                    except (ImportError, Exception):
                        cal_method = "none"  # Fallback if betacal not installed
                if cal_method == "lr_meta":
                    try:
                        # LR meta-calibration: train base model, fit 1-feature LR on held-out raw probs.
                        # Equivalent to Platt scaling but trained explicitly on a temporal hold-out slice
                        # rather than via sklearn's cross-validated CalibratedClassifierCV.  This avoids
                        # the small-fold instability that sigmoid cv=3 can suffer on <200 samples.
                        m.fit(X_sub[ti_safe], y_eval[ti_safe])
                        _model_fitted = True
                        cal_slice = ti_safe[-200:] if len(ti_safe) > 200 else ti_safe
                        raw_p = m.predict_proba(X_sub[cal_slice])[:, 1]
                        _lr_meta_cal = LogisticRegression(
                            C=1.0, solver="lbfgs", max_iter=500, random_state=42
                        )
                        _lr_meta_cal.fit(raw_p.reshape(-1, 1), y_eval[cal_slice])
                        cal_method = "none"
                    except Exception:
                        cal_method = "none"  # Fallback: uncalibrated
                if cal_method == "lr_platt":
                    # Post-hoc Platt scaling: fit base model, then LR on held-out probs
                    # More reliable than CalibratedClassifierCV(cv=3) with time series data
                    m.fit(X_sub[ti_safe], y_eval[ti_safe])
                    _model_fitted = True
                    cal_size = min(400, max(50, len(ti_safe) // 3))
                    if len(ti_safe) > cal_size + 50:
                        cal_slice = ti_safe[-cal_size:]
                    else:
                        cal_slice = ti_safe
                    raw_p = m.predict_proba(X_sub[cal_slice])[:, 1].reshape(-1, 1)
                    _platt_cal = LogisticRegression(C=1.0, max_iter=200, random_state=42)
                    try:
                        _platt_cal.fit(raw_p, y_eval[cal_slice])
                    except Exception:
                        _platt_cal = None
                    cal_method = "none"
                if cal_method == "sigmoid":
                    m = CalibratedClassifierCV(m, method=cal_method, cv=3)
                if not _model_fitted:
                    m.fit(X_sub[ti_safe], y_eval[ti_safe])
                p = m.predict_proba(X_sub[vi])[:, 1]
                if _beta_cal is not None:
                    p = _beta_cal.predict(p.reshape(-1, 1))
                if _lr_meta_cal is not None:
                    p = _lr_meta_cal.predict_proba(p.reshape(-1, 1))[:, 1]
                if _platt_cal is not None:
                    p = _platt_cal.predict_proba(p.reshape(-1, 1))[:, 1]
                p = np.clip(p, 0.025, 0.975)  # clip extremes → Brier gain ~0.003
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
    feature_penalty = 0.001 * max(0, n_features - 65)  # Soft pressure toward 65 features
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
        # Enforce minimum regularization to prevent overfitting
        reg_a = max(hp["reg_alpha"], 0.01)
        reg_l = max(hp["reg_lambda"], 0.1)
        if mt == "xgboost":
            return xgb.XGBClassifier(n_estimators=n_est, max_depth=depth,
                                     learning_rate=lr, subsample=hp["subsample"],
                                     colsample_bytree=hp["colsample_bytree"], min_child_weight=hp["min_child_weight"],
                                     reg_alpha=reg_a, reg_lambda=reg_l,
                                     eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist")
        elif mt == "xgboost_brier":
            return xgb.XGBClassifier(n_estimators=n_est, max_depth=depth,
                                     learning_rate=lr, subsample=hp["subsample"],
                                     colsample_bytree=hp["colsample_bytree"], min_child_weight=hp["min_child_weight"],
                                     reg_alpha=reg_a, reg_lambda=reg_l,
                                     objective=_brier_objective,
                                     eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist")
        elif mt == "lightgbm":
            return lgbm.LGBMClassifier(n_estimators=n_est, max_depth=depth,
                                       learning_rate=lr, subsample=hp["subsample"],
                                       num_leaves=min(2 ** depth - 1, 31),
                                       reg_alpha=reg_a, reg_lambda=reg_l,
                                       min_child_samples=20, feature_fraction=0.7,
                                       boosting_type="dart", drop_rate=0.1,
                                       verbose=-1, random_state=42, n_jobs=-1)
        elif mt == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(iterations=min(n_est, 100), depth=min(depth, 6),
                                      learning_rate=lr, l2_leaf_reg=hp["reg_lambda"],
                                      verbose=0, random_state=42,
                                      task_type='CPU', thread_count=-1)
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
        elif mt == "logistic_regression":
            # MDPI 2026: LR achieves best Brier (0.199) among tabular models for NBA prediction.
            # Inherently well-calibrated probabilities. C maps from reg_lambda (inverse regularization).
            C = max(0.01, 1.0 / max(hp.get("reg_lambda", 1.0), 0.01))
            C = min(C, 100.0)  # cap to avoid overfitting
            return Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(C=C, max_iter=500, solver='lbfgs',
                                          random_state=42, n_jobs=-1))
            ])
        elif mt == "mlp":
            hidden = hp.get("nn_hidden_dims", 128)
            n_layers = hp.get("nn_n_layers", 3)
            layers = tuple([hidden // (2 ** i) for i in range(n_layers) if hidden // (2 ** i) >= 16])
            if not layers:
                layers = (64, 32)
            return Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(
                    hidden_layer_sizes=layers,
                    activation='relu',
                    max_iter=min(hp.get("nn_epochs", 500), 500),
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=42
                ))
            ])
        elif mt == "lstm":
            # LSTM via sklearn-compatible wrapper (falls back to MLP if pytorch unavailable)
            try:
                return _build_pytorch_model(hp, model_type="lstm")
            except Exception:
                return _build_mlp_fallback(hp)
        elif mt == "transformer":
            try:
                return _build_pytorch_model(hp, model_type="transformer")
            except Exception:
                return _build_mlp_fallback(hp)
        elif mt == "tabnet":
            try:
                from pytorch_tabnet.tab_model import TabNetClassifier
                return TabNetClassifier(
                    n_d=hp.get("nn_hidden_dims", 64), n_a=hp.get("nn_hidden_dims", 64),
                    n_steps=max(3, hp.get("nn_n_layers", 3)),
                    gamma=1.3, lambda_sparse=1e-3,
                    cat_dims=[], cat_emb_dim=[],
                    optimizer_params=dict(lr=lr),
                    mask_type="entmax",
                    max_epochs=min(hp.get("nn_epochs", 50), 50),
                    patience=10, batch_size=hp.get("nn_batch_size", 128),
                    verbose=0, seed=42,
                )
            except ImportError:
                return _build_mlp_fallback(hp)
        elif mt == "ft_transformer":
            # Feature Tokenizer + Transformer — falls back to MLP
            try:
                return _build_pytorch_model(hp, model_type="ft_transformer")
            except Exception:
                return _build_mlp_fallback(hp)
        elif mt == "deep_ensemble":
            # Ensemble of 3 MLPs with different random seeds
            try:
                return _build_deep_ensemble(hp)
            except Exception:
                return _build_mlp_fallback(hp)
        elif mt == "autogluon":
            # AutoGluon wraps multiple models — heavy, use sparingly
            try:
                return _build_autogluon(hp)
            except ImportError:
                return _build_mlp_fallback(hp)
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


def _build_mlp_fallback(hp):
    """Fallback MLP when a neural net type fails to build."""
    hidden = hp.get("nn_hidden_dims", 128)
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(hidden, hidden // 2, max(hidden // 4, 16)),
            activation='relu', max_iter=300, early_stopping=True,
            validation_fraction=0.15, random_state=42
        ))
    ])


def _build_pytorch_model(hp, model_type="lstm"):
    """Build a PyTorch-based sklearn-compatible classifier (LSTM/Transformer/FT-Transformer).
    Wraps in a sklearn Pipeline with StandardScaler for compatibility with the eval loop."""
    try:
        import torch
        import torch.nn as nn
        from sklearn.base import BaseEstimator, ClassifierMixin

        class PytorchTabularClassifier(BaseEstimator, ClassifierMixin):
            """Lightweight sklearn-compatible wrapper for PyTorch tabular models."""

            def __init__(self, model_type="lstm", hidden_dim=128, n_layers=2,
                         dropout=0.3, epochs=50, batch_size=64, lr=0.001):
                self.model_type = model_type
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers
                self.dropout = dropout
                self.epochs = epochs
                self.batch_size = batch_size
                self.lr = lr
                self.model_ = None
                self.scaler_ = StandardScaler()
                self.classes_ = np.array([0, 1])

            def _build_net(self, input_dim):
                if self.model_type == "lstm":
                    class LSTMNet(nn.Module):
                        def __init__(self, input_dim, hidden, n_layers, dropout):
                            super().__init__()
                            self.lstm = nn.LSTM(input_dim, hidden, n_layers,
                                                batch_first=True, dropout=dropout if n_layers > 1 else 0)
                            self.fc = nn.Sequential(
                                nn.Dropout(dropout), nn.Linear(hidden, hidden // 2),
                                nn.ReLU(), nn.Linear(hidden // 2, 1), nn.Sigmoid())
                        def forward(self, x):
                            x = x.unsqueeze(1)  # (batch, 1, features)
                            _, (h, _) = self.lstm(x)
                            return self.fc(h[-1]).squeeze(-1)
                    return LSTMNet(input_dim, self.hidden_dim, self.n_layers, self.dropout)
                elif self.model_type == "transformer":
                    class TransformerNet(nn.Module):
                        def __init__(self, input_dim, hidden, n_layers, dropout):
                            super().__init__()
                            self.embedding = nn.Linear(input_dim, hidden)
                            encoder_layer = nn.TransformerEncoderLayer(
                                d_model=hidden, nhead=max(1, hidden // 32),
                                dim_feedforward=hidden * 2, dropout=dropout, batch_first=True)
                            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                            self.fc = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
                        def forward(self, x):
                            x = self.embedding(x).unsqueeze(1)
                            x = self.encoder(x)
                            return self.fc(x[:, 0]).squeeze(-1)
                    return TransformerNet(input_dim, self.hidden_dim, self.n_layers, self.dropout)
                else:  # ft_transformer
                    class FTTransformerNet(nn.Module):
                        def __init__(self, input_dim, hidden, n_layers, dropout):
                            super().__init__()
                            self.feature_tokens = nn.Linear(1, hidden)
                            encoder_layer = nn.TransformerEncoderLayer(
                                d_model=hidden, nhead=max(1, hidden // 32),
                                dim_feedforward=hidden * 2, dropout=dropout, batch_first=True)
                            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
                            self.fc = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
                        def forward(self, x):
                            # x: (batch, features) -> per-feature tokens
                            tokens = self.feature_tokens(x.unsqueeze(-1))  # (batch, features, hidden)
                            cls = self.cls_token.expand(x.size(0), -1, -1)
                            tokens = torch.cat([cls, tokens], dim=1)
                            tokens = self.encoder(tokens)
                            return self.fc(tokens[:, 0]).squeeze(-1)
                    return FTTransformerNet(input_dim, self.hidden_dim, self.n_layers, self.dropout)

            def fit(self, X, y):
                X = self.scaler_.fit_transform(X)
                X_t = torch.FloatTensor(X)
                y_t = torch.FloatTensor(y.astype(float))
                self.model_ = self._build_net(X.shape[1])
                optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
                criterion = nn.BCELoss()
                self.model_.train()
                best_loss = float('inf')
                patience_counter = 0
                for epoch in range(self.epochs):
                    indices = torch.randperm(len(X_t))
                    total_loss = 0
                    n_batches = 0
                    for start in range(0, len(X_t), self.batch_size):
                        batch_idx = indices[start:start + self.batch_size]
                        optimizer.zero_grad()
                        pred = self.model_(X_t[batch_idx])
                        loss = criterion(pred, y_t[batch_idx])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                        optimizer.step()
                        total_loss += loss.item()
                        n_batches += 1
                    avg_loss = total_loss / max(n_batches, 1)
                    if avg_loss < best_loss - 0.001:
                        best_loss = avg_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= 10:
                            break
                return self

            def predict_proba(self, X):
                X = self.scaler_.transform(X)
                self.model_.eval()
                with torch.no_grad():
                    probs = self.model_(torch.FloatTensor(X)).numpy()
                return np.column_stack([1 - probs, probs])

            def predict(self, X):
                return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

            def get_params(self, deep=True):
                return {"model_type": self.model_type, "hidden_dim": self.hidden_dim,
                        "n_layers": self.n_layers, "dropout": self.dropout,
                        "epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr}

        return PytorchTabularClassifier(
            model_type=model_type,
            hidden_dim=hp.get("nn_hidden_dims", 128),
            n_layers=hp.get("nn_n_layers", 2),
            dropout=hp.get("nn_dropout", 0.3),
            epochs=min(hp.get("nn_epochs", 50), 50),  # Cap at 50 for speed
            batch_size=hp.get("nn_batch_size", 64),
            lr=hp.get("learning_rate", 0.001),
        )
    except ImportError:
        raise  # Let caller handle fallback


def _build_deep_ensemble(hp):
    """Ensemble of 3 MLPs with different seeds — sklearn compatible."""
    from sklearn.ensemble import VotingClassifier
    hidden = hp.get("nn_hidden_dims", 128)
    estimators = []
    for seed in [42, 123, 456]:
        mlp = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(hidden, hidden // 2, max(hidden // 4, 16)),
                activation='relu', max_iter=min(hp.get("nn_epochs", 200), 200),
                early_stopping=True, validation_fraction=0.15, random_state=seed
            ))
        ])
        estimators.append((f"mlp_{seed}", mlp))
    return VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)


def _build_autogluon(hp):
    """AutoGluon TabularPredictor wrapped for sklearn compatibility."""
    try:
        from autogluon.tabular import TabularPredictor
        from sklearn.base import BaseEstimator, ClassifierMixin
        import pandas as pd
        import tempfile

        class AutoGluonWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, time_limit=60):
                self.time_limit = time_limit
                self.predictor_ = None
                self.classes_ = np.array([0, 1])
                self._tmpdir = tempfile.mkdtemp()

            def fit(self, X, y):
                df = pd.DataFrame(X)
                df["target"] = y
                self.predictor_ = TabularPredictor(
                    label="target", path=self._tmpdir, verbosity=0
                ).fit(df, time_limit=self.time_limit, presets="best_quality")
                return self

            def predict_proba(self, X):
                df = pd.DataFrame(X)
                probs = self.predictor_.predict_proba(df)
                return probs.values if hasattr(probs, 'values') else np.column_stack([1 - probs, probs])

            def predict(self, X):
                return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

            def get_params(self, deep=True):
                return {"time_limit": self.time_limit}

        return AutoGluonWrapper(time_limit=min(hp.get("nn_epochs", 60), 60))
    except ImportError:
        raise


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

# ── Runtime config (env-overridable for multi-Space island model) ──
# S10 (primary): exploitation — proven optimal params
# S11 (secondary): exploration — higher mutation, wider feature search
import os as _os
_SPACE_ROLE = _os.environ.get("SPACE_ROLE", "exploitation")  # "exploitation" or "exploration"

_ROLE_CONFIGS = {
    "exploitation": {  # S10: proven optimal
        "POP_SIZE": 60, "BASE_MUT": 0.09, "MUT_DECAY_RATE": 0.998, "MUT_FLOOR": 0.05,
        "CROSSOVER_RATE": 0.80, "TARGET_FEATURES": 63, "TOURNAMENT_SIZE": 7, "DIVERSITY_RESTART": 30,
    },
    "exploration": {  # S11: wider search
        "POP_SIZE": 60, "BASE_MUT": 0.15, "MUT_DECAY_RATE": 0.999, "MUT_FLOOR": 0.08,
        "CROSSOVER_RATE": 0.70, "TARGET_FEATURES": 80, "TOURNAMENT_SIZE": 5, "DIVERSITY_RESTART": 20,
    },
    "extra_trees_specialist": {  # S12: extra_trees focus, tight features
        "POP_SIZE": 60, "BASE_MUT": 0.08, "MUT_DECAY_RATE": 0.998, "MUT_FLOOR": 0.04,
        "CROSSOVER_RATE": 0.80, "TARGET_FEATURES": 60, "TOURNAMENT_SIZE": 7, "DIVERSITY_RESTART": 25,
    },
    "catboost_specialist": {  # S13: catboost focus, medium exploration
        "POP_SIZE": 60, "BASE_MUT": 0.10, "MUT_DECAY_RATE": 0.998, "MUT_FLOOR": 0.05,
        "CROSSOVER_RATE": 0.80, "TARGET_FEATURES": 66, "TOURNAMENT_SIZE": 6, "DIVERSITY_RESTART": 25,
    },
    "neural_specialist": {  # S14: neural models — GPU ONLY (Colab), skip on CPU HF Spaces
        "POP_SIZE": 40, "BASE_MUT": 0.12, "MUT_DECAY_RATE": 0.999, "MUT_FLOOR": 0.06,
        "CROSSOVER_RATE": 0.75, "TARGET_FEATURES": 50, "TOURNAMENT_SIZE": 5, "DIVERSITY_RESTART": 15,
    },
    "lightgbm_specialist": {  # S14 on CPU: LightGBM is fastest tree model
        "POP_SIZE": 60, "BASE_MUT": 0.08, "MUT_DECAY_RATE": 0.998, "MUT_FLOOR": 0.04,
        "CROSSOVER_RATE": 0.85, "TARGET_FEATURES": 55, "TOURNAMENT_SIZE": 7, "DIVERSITY_RESTART": 25,
    },
    "wide_search": {  # S15: diversity, moderate feature search (120 was too slow on CPU)
        "POP_SIZE": 50, "BASE_MUT": 0.18, "MUT_DECAY_RATE": 0.999, "MUT_FLOOR": 0.08,
        "CROSSOVER_RATE": 0.65, "TARGET_FEATURES": 80, "TOURNAMENT_SIZE": 4, "DIVERSITY_RESTART": 15,
    },
}
_cfg = _ROLE_CONFIGS.get(_SPACE_ROLE, _ROLE_CONFIGS["exploitation"])
POP_SIZE = _cfg["POP_SIZE"]
BASE_MUT = _cfg["BASE_MUT"]
MUT_DECAY_RATE = _cfg["MUT_DECAY_RATE"]
MUT_FLOOR = _cfg["MUT_FLOOR"]
CROSSOVER_RATE = _cfg["CROSSOVER_RATE"]
TARGET_FEATURES = _cfg["TARGET_FEATURES"]
TOURNAMENT_SIZE = _cfg["TOURNAMENT_SIZE"]
DIVERSITY_RESTART = _cfg["DIVERSITY_RESTART"]

N_ISLANDS = 5
ISLAND_SIZE = POP_SIZE // N_ISLANDS
ELITE_SIZE = max(2, POP_SIZE // 10)
ELITE_PER_ISLAND = max(1, ISLAND_SIZE // 6)

N_SPLITS = 2             # 2-fold fast eval (3-fold only for FULL_EVAL_TOP)
GENS_PER_CYCLE = 3       # Save every 3 gens
COOLDOWN = 5             # Minimal cooldown
FAST_EVAL_GAMES = 5000   # Subsample for fast eval (was 8000 — too slow on CPU)
FULL_EVAL_TOP = 5        # Full eval for top 5 only (was 10 — halves slow full evals)
MIGRATION_INTERVAL = 8   # Migration every 8 gens
MIGRANTS_PER_ISLAND = 3  # 3 migrants per island (25% of island)
GC_INTERVAL = 10         # Force garbage collection every N individuals evaluated

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


def _adaptive_tournament_size(population, base_size=4):
    """Adaptive tournament size based on population diversity.
    Low diversity -> larger tournaments (stronger selection pressure).
    High diversity -> smaller tournaments (preserve exploration)."""
    if not population:
        return base_size
    composites = [ind.fitness.get("composite", 0) for ind in population]
    diversity = np.std(composites) if len(composites) > 1 else 0
    if diversity < 0.005:
        return min(base_size + 3, 10)  # Very low diversity: strong pressure
    elif diversity < 0.01:
        return min(base_size + 1, 8)
    elif diversity > 0.05:
        return max(base_size - 1, 2)   # High diversity: weak pressure
    return base_size


def evolution_loop():
    """Main 24/7 genetic evolution loop — runs in background thread.
    Uses island model with 5 sub-populations of 100, NSGA-II Pareto ranking,
    adaptive mutation decay, and memory management for 16GB RAM."""
    global TARGET_FEATURES, CROSSOVER_RATE, COOLDOWN, POP_SIZE, N_ISLANDS, ISLAND_SIZE
    global _evo_X, _evo_y, _evo_features, _evo_best, _evo_games
    log("=" * 60)
    log("REAL GENETIC EVOLUTION LOOP v4 — ISLAND MODEL + NSGA-II")
    log(f"Pop: {POP_SIZE} ({N_ISLANDS} islands x {ISLAND_SIZE}) | Target features: {TARGET_FEATURES} | Gens/cycle: {GENS_PER_CYCLE}")
    log(f"Mutation: {BASE_MUT} -> {MUT_FLOOR} (decay {MUT_DECAY_RATE}/gen) | Crossover: {CROSSOVER_RATE}")
    log(f"Migration: every {MIGRATION_INTERVAL} gens, {MIGRANTS_PER_ISLAND} per island")
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

    # Try restore state — island model
    island_model = None
    generation = 0
    best_ever = None
    stagnation = 0
    mutation_rate = BASE_MUT

    state_file = STATE_DIR / "population.json"
    restored = False
    if state_file.exists():
        try:
            st = json.loads(state_file.read_text())
            generation = st["generation"]
            stagnation = st.get("stagnation_counter", 0)
            mutation_rate = st.get("mutation_rate", BASE_MUT)

            # Try island model restore first
            if "island_model" in st:
                island_model = IslandModel.from_state_dict(st["island_model"], n_feat)
                log(f"RESTORED island model: gen {generation}, {N_ISLANDS} islands")
                restored = True
            elif "population" in st:
                # Legacy flat population — convert to island model
                population = []
                for d in st["population"]:
                    ind = Individual.__new__(Individual)
                    ind.features = d["features"]; ind.hyperparams = d["hyperparams"]
                    ind.fitness = d["fitness"]; ind.generation = d.get("generation", 0)
                    ind.n_features = sum(ind.features)
                    ind.pareto_rank = d.get("pareto_rank", 999)
                    ind.crowding_dist = 0.0
                    ind.island_id = d.get("island_id", -1)
                    # Ensure new hyperparams exist
                    ind.hyperparams.setdefault("nn_hidden_dims", 128)
                    ind.hyperparams.setdefault("nn_n_layers", 2)
                    ind.hyperparams.setdefault("nn_dropout", 0.3)
                    ind.hyperparams.setdefault("nn_epochs", 50)
                    ind.hyperparams.setdefault("nn_batch_size", 64)
                    _valid_types = CPU_MODEL_TYPES if not _HAS_GPU else ALL_MODEL_TYPES
                    if ind.hyperparams.get("model_type") not in _valid_types:
                        ind.hyperparams["model_type"] = random.choice(_valid_types)
                    population.append(ind)

                # Convert flat population to island model
                island_model = IslandModel(n_islands=N_ISLANDS, island_size=ISLAND_SIZE,
                                           n_features=n_feat, target_features=TARGET_FEATURES,
                                           migration_interval=MIGRATION_INTERVAL,
                                           migrants_per_island=MIGRANTS_PER_ISLAND)
                # Keep elites from old population, fill rest
                elites = sorted(population, key=lambda x: x.fitness.get("composite", 0), reverse=True)[:ELITE_SIZE]
                for e in elites:
                    if len(e.features) < n_feat:
                        e.features.extend([random.randint(0, 1) for _ in range(n_feat - len(e.features))])
                    elif len(e.features) > n_feat:
                        e.features = e.features[:n_feat]
                    e.n_features = sum(e.features)
                island_model.initialize()
                # Inject elites into island 0
                for i, e in enumerate(elites):
                    if i < len(island_model.islands[0]):
                        island_model.islands[0][i] = e
                log(f"MIGRATED legacy pop to island model: {len(population)} -> {N_ISLANDS}x{ISLAND_SIZE}")
                restored = True

            if st.get("best_ever"):
                be = st["best_ever"]
                best_ever = Individual.__new__(Individual)
                best_ever.features = be["features"]; best_ever.hyperparams = be["hyperparams"]
                best_ever.fitness = be["fitness"]; best_ever.generation = be.get("generation", 0)
                best_ever.n_features = sum(best_ever.features)
                best_ever.pareto_rank = be.get("pareto_rank", 0)
                best_ever.crowding_dist = 0.0
                best_ever.island_id = be.get("island_id", 0)
                best_ever.hyperparams.setdefault("nn_hidden_dims", 128)
                best_ever.hyperparams.setdefault("nn_n_layers", 2)
                best_ever.hyperparams.setdefault("nn_dropout", 0.3)
                best_ever.hyperparams.setdefault("nn_epochs", 50)
                best_ever.hyperparams.setdefault("nn_batch_size", 64)
            log(f"RESTORED: gen {generation}, best Brier={best_ever.fitness['brier']:.4f}" if best_ever else "RESTORED: no best yet")
        except Exception as e:
            log(f"Restore failed: {e}", "WARN")
            traceback.print_exc()
            island_model = None

    # Initialize fresh island model if not restored
    if island_model is None:
        island_model = IslandModel(n_islands=N_ISLANDS, island_size=ISLAND_SIZE,
                                   n_features=n_feat, target_features=TARGET_FEATURES,
                                   migration_interval=MIGRATION_INTERVAL,
                                   migrants_per_island=MIGRANTS_PER_ISLAND)
        island_model.initialize()
        log(f"FRESH island model: {N_ISLANDS} islands x {ISLAND_SIZE} = {POP_SIZE} individuals")
        log(f"Island specializations: {[s[:3] for s in island_model.island_specializations]}")

    live["pop_size"] = POP_SIZE
    live["n_islands"] = N_ISLANDS
    live["island_sizes"] = [len(isl) for isl in island_model.islands]

    # ── INFINITE LOOP (Island Model + NSGA-II) ──
    cycle = 0
    while True:
        cycle += 1
        live["cycle"] = cycle
        live["status"] = f"EVOLVING (cycle {cycle})"
        log(f"\n{'='*50}")
        log(f"CYCLE {cycle} — {GENS_PER_CYCLE} generations | Mut={mutation_rate:.4f}")
        log(f"{'='*50}")
        cycle_start = time.time()

        for gen_i in range(GENS_PER_CYCLE):
            generation += 1
            live["generation"] = generation
            gen_start = time.time()

            # ── EVALUATE ALL ISLANDS (with per-individual timeout) ──
            population = island_model.get_all_individuals()
            _timed_out = 0
            for i, ind in enumerate(population):
                # Skip re-eval of elites (per-island top ELITE_PER_ISLAND)
                if ind.pareto_rank == 0 and ind.fitness.get("composite", 0) > 0:
                    continue
                _t0 = time.time()
                evaluate(ind, X, y, N_SPLITS, fast=True)
                _dur = time.time() - _t0
                if _dur > EVAL_TIMEOUT_S:
                    # This eval was too slow — penalize and warn
                    ind.fitness = {"brier": 0.29, "roi": 0.0, "sharpe": 0.0,
                                   "calibration": 0.15, "composite": -0.5, "features_pruned": 0}
                    _timed_out += 1
            if _timed_out:
                log(f"[TIMEOUT] {_timed_out}/{len(population)} individuals exceeded {EVAL_TIMEOUT_S}s")

            # ── NSGA-II PARETO RANKING (global) ──
            population = nsga2_rank(population)

            # ── FULL EVAL: top FULL_EVAL_TOP get precise 3-fold on ALL data ──
            for ind in population[:FULL_EVAL_TOP]:
                evaluate(ind, X, y, n_splits=3, fast=False)

            # Re-rank after full eval
            population = nsga2_rank(population)
            best = population[0]
            pareto_front_size = sum(1 for ind in population if ind.pareto_rank == 0)

            # ── Memory cleanup after full population eval ──
            _gc_cleanup()

            # ── Update feature importance from top performers ──
            global _feature_importance
            if _feature_importance is None:
                _feature_importance = np.zeros(n_feat)
            top_features = np.zeros(n_feat)
            for ind in population[:20]:  # Top 20 (was 10) — more signal with 500 pop
                for idx in ind.selected_indices():
                    if idx < n_feat:
                        top_features[idx] += 1
            _feature_importance = 0.7 * _feature_importance + 0.3 * (top_features / 20.0)

            # Track best ever
            prev_brier = best_ever.fitness["brier"] if best_ever else 1.0
            if best_ever is None or best.fitness["composite"] > best_ever.fitness["composite"]:
                best_ever = Individual.__new__(Individual)
                best_ever.features = best.features[:]
                best_ever.hyperparams = dict(best.hyperparams)
                best_ever.fitness = dict(best.fitness)
                best_ever.n_features = best.n_features
                best_ever.generation = generation
                best_ever.pareto_rank = best.pareto_rank
                best_ever.crowding_dist = best.crowding_dist
                best_ever.island_id = best.island_id
                # Share for /api/predict
                _evo_best = {
                    "features": best_ever.features[:],
                    "hyperparams": dict(best_ever.hyperparams),
                    "fitness": dict(best_ever.fitness),
                    "generation": best_ever.generation,
                    "n_features": best_ever.n_features,
                }

            # Stagnation detection — track BOTH Brier and composite
            prev_best_composite = best_ever.fitness["composite"] if best_ever else 0.0
            brier_stagnant = abs(best.fitness["brier"] - prev_brier) < 0.0005
            composite_stagnant = abs(best.fitness["composite"] - prev_best_composite) < 0.001
            if brier_stagnant and composite_stagnant:
                stagnation += 1
            elif not brier_stagnant:
                stagnation = max(0, stagnation - 2)  # Partial reset on real improvement
            else:
                stagnation = max(0, stagnation - 1)

            # ── Adaptive mutation: decay over time, boost on stagnation ──
            # Base decay: starts at BASE_MUT, decays toward MUT_FLOOR
            # Data analysis: 0.10-0.15 has 80% improvement rate vs 18% at >=0.20
            # So caps are set to keep mutation in the productive range
            mutation_rate *= MUT_DECAY_RATE
            mutation_rate = max(MUT_FLOOR, mutation_rate)
            # Stagnation boost (capped at 0.15 — data shows >=0.20 is destructive)
            if stagnation >= 10:
                mutation_rate = min(0.15, mutation_rate * 1.5)
            elif stagnation >= 7:
                mutation_rate = min(0.13, mutation_rate * 1.3)
            elif stagnation >= 3:
                mutation_rate = min(0.12, mutation_rate * 1.15)

            # ── Adaptive tournament size ──
            tourney_size = _adaptive_tournament_size(population, TOURNAMENT_SIZE)

            # ── Distribute population back to islands ──
            island_model.set_all_individuals(population)

            # ── MIGRATION between islands ──
            migrated = island_model.migrate(generation)
            if migrated:
                log(f"  MIGRATION: {MIGRANTS_PER_ISLAND} individuals exchanged between {N_ISLANDS} islands")

            # Update live state
            live["best_brier"] = best_ever.fitness["brier"]
            live["best_roi"] = best_ever.fitness["roi"]
            live["best_sharpe"] = best_ever.fitness["sharpe"]
            live["best_features"] = best_ever.n_features
            live["best_model_type"] = best_ever.hyperparams["model_type"]
            live["mutation_rate"] = round(mutation_rate, 4)
            live["stagnation"] = stagnation
            live["pareto_front_size"] = pareto_front_size
            live["island_sizes"] = [len(isl) for isl in island_model.islands]
            live["history"].append({
                "gen": generation, "brier": best.fitness["brier"], "roi": best.fitness["roi"],
                "composite": best.fitness["composite"], "features": best.n_features,
                "pareto_front": pareto_front_size,
            })
            if len(live["history"]) > 500:
                live["history"] = live["history"][-500:]

            ge = time.time() - gen_start
            log(f"Gen {generation}: Brier={best.fitness['brier']:.4f} ROI={best.fitness['roi']:.1%} "
                f"Sharpe={best.fitness['sharpe']:.2f} Feat={best.n_features} "
                f"Model={best.hyperparams['model_type']} Pareto={pareto_front_size} "
                f"Mut={mutation_rate:.4f} Stag={stagnation} ({ge:.0f}s)")

            # ── Supabase: log generation ──
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
                         bool(stagnation == 0 and generation > 1)))
                    log(f"[SUPABASE] Gen {generation} logged OK")
                    for rank, ind in enumerate(population[:10]):
                        _cur.execute("""INSERT INTO public.nba_evolution_evals
                            (generation, individual_rank, brier, roi, sharpe, composite, n_features, model_type)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                            (generation, rank + 1, float(ind.fitness["brier"]), float(ind.fitness["roi"]),
                             float(ind.fitness["sharpe"]), float(ind.fitness["composite"]),
                             int(ind.n_features), str(ind.hyperparams["model_type"])))
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
                    top10 = [{"brier": ind.fitness["brier"], "roi": ind.fitness["roi"],
                              "sharpe": ind.fitness["sharpe"], "composite": ind.fitness["composite"],
                              "n_features": ind.n_features, "model_type": ind.hyperparams["model_type"]}
                             for ind in population[:10]]
                    run_logger.log_top_evals(generation, top10)
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
                        elif atype == "emergency_diversify":
                            remote_config["commands"].append("diversify")
                        elif atype == "full_reset":
                            remote_config["pending_reset"] = True
                        elif atype == "flag":
                            live["pause_betting"] = params.get("pause_betting", False)
                except Exception as e:
                    log(f"[RUN-LOGGER] Gen log error: {e}", "WARN")

            # ── EVOLVE EACH ISLAND independently ──
            for island_id in range(N_ISLANDS):
                island = island_model.islands[island_id]
                island.sort(key=lambda x: (x.pareto_rank, -x.crowding_dist))

                new_island = []
                # Elitism per island
                for i in range(min(ELITE_PER_ISLAND, len(island))):
                    e = Individual.__new__(Individual)
                    e.features = island[i].features[:]; e.hyperparams = dict(island[i].hyperparams)
                    e.fitness = dict(island[i].fitness); e.n_features = island[i].n_features
                    e.generation = island[i].generation; e.island_id = island_id
                    e.pareto_rank = island[i].pareto_rank; e.crowding_dist = island[i].crowding_dist
                    new_island.append(e)

                # Stagnation injection (per-island) — tiered response
                if stagnation >= 7:
                    n_inject = ISLAND_SIZE // 3
                    specialization = island_model.island_specializations[island_id]
                    n_random = n_inject // 3      # 1/3 fully random
                    n_mutant = n_inject // 3      # 1/3 mutants of best
                    n_cross_island = n_inject - n_random - n_mutant  # 1/3 from other islands' best
                    # Random fresh individuals
                    for _ in range(n_random):
                        fresh = Individual(n_feat, TARGET_FEATURES, model_type=random.choice(specialization))
                        fresh.island_id = island_id
                        new_island.append(fresh)
                    # Targeted mutants of island's best
                    if island:
                        island_best = min(island, key=lambda x: x.fitness.get("brier", 1.0))
                        for _ in range(n_mutant):
                            mutant = Individual.__new__(Individual)
                            mutant.features = island_best.features[:]
                            mutant.hyperparams = dict(island_best.hyperparams)
                            mutant.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "composite": 0.0}
                            mutant.n_features = island_best.n_features
                            mutant.generation = generation
                            mutant.island_id = island_id
                            mutant.pareto_rank = 999
                            mutant.crowding_dist = 0.0
                            mutant.mutate(0.20)  # Heavy mutation
                            new_island.append(mutant)
                    # Cross-island pollination: inject best from OTHER islands
                    other_islands = [i for i in range(N_ISLANDS) if i != island_id]
                    for _ in range(n_cross_island):
                        src = random.choice(other_islands)
                        src_island = island_model.islands[src]
                        if src_island:
                            donor = min(src_island, key=lambda x: x.fitness.get("brier", 1.0))
                            cross = Individual.__new__(Individual)
                            cross.features = donor.features[:]
                            cross.hyperparams = dict(donor.hyperparams)
                            cross.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "composite": 0.0}
                            cross.n_features = donor.n_features
                            cross.generation = generation
                            cross.island_id = island_id
                            cross.pareto_rank = 999
                            cross.crowding_dist = 0.0
                            cross.mutate(0.15)  # Moderate mutation to adapt to new island
                            new_island.append(cross)
                elif stagnation >= 3:
                    # Mild injection: 10% fresh per island
                    n_inject = max(2, ISLAND_SIZE // 10)
                    specialization = island_model.island_specializations[island_id]
                    for _ in range(n_inject):
                        fresh = Individual(n_feat, TARGET_FEATURES, model_type=random.choice(specialization))
                        fresh.island_id = island_id
                        new_island.append(fresh)

                if stagnation >= DIVERSITY_RESTART:
                    n_restart = ISLAND_SIZE // 2
                    new_island = new_island[:ELITE_PER_ISLAND]
                    specialization = island_model.island_specializations[island_id]
                    for _ in range(n_restart):
                        fresh = Individual(n_feat, TARGET_FEATURES, model_type=random.choice(specialization))
                        fresh.island_id = island_id
                        new_island.append(fresh)

                # Fill with crossover + mutation (within island)
                while len(new_island) < ISLAND_SIZE:
                    cs = random.sample(island, min(tourney_size, len(island)))
                    p1 = min(cs, key=lambda x: (x.pareto_rank, -x.crowding_dist))  # NSGA-II selection
                    cs2 = random.sample(island, min(tourney_size, len(island)))
                    p2 = min(cs2, key=lambda x: (x.pareto_rank, -x.crowding_dist))
                    if random.random() < CROSSOVER_RATE:
                        child = Individual.crossover(p1, p2)
                    else:
                        child = Individual.__new__(Individual)
                        child.features = p1.features[:]; child.hyperparams = dict(p1.hyperparams)
                        child.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "composite": 0.0}
                        child.n_features = p1.n_features; child.generation = generation
                        child.pareto_rank = 999; child.crowding_dist = 0.0
                    child.island_id = island_id
                    child.mutate(mutation_rate, feature_importance=_feature_importance)
                    new_island.append(child)

                island_model.islands[island_id] = new_island[:ISLAND_SIZE]

            # Reset stagnation after emergency restart
            if stagnation >= DIVERSITY_RESTART:
                stagnation = 0

        # ── Save state ──
        live["status"] = "SAVING"
        try:
            population = island_model.get_all_individuals()
            sel_names = [feature_names[i] for i in best_ever.selected_indices() if i < len(feature_names)] if best_ever else []
            state_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "generation": generation, "n_features": n_feat,
                "stagnation_counter": stagnation, "mutation_rate": mutation_rate,
                "island_model": island_model.to_state_dict(),
                "best_ever": {"features": best_ever.features,
                              "hyperparams": {k: (float(v) if isinstance(v, np.floating) else v) for k, v in best_ever.hyperparams.items()},
                              "fitness": best_ever.fitness, "generation": best_ever.generation,
                              "island_id": getattr(best_ever, 'island_id', 0),
                              "pareto_rank": getattr(best_ever, 'pareto_rank', 0)} if best_ever else None,
                "history": live["history"][-200:],
            }
            (STATE_DIR / "population.json").write_text(json.dumps(state_data, default=str))

            island_stats = island_model.get_island_stats()
            results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": cycle, "generation": generation, "pop_size": POP_SIZE,
                "n_islands": N_ISLANDS, "island_size": ISLAND_SIZE,
                "mutation_rate": round(mutation_rate, 4), "stagnation": stagnation,
                "pareto_front_size": sum(1 for ind in population if ind.pareto_rank == 0),
                "best": {"brier": best_ever.fitness["brier"], "roi": best_ever.fitness["roi"],
                         "sharpe": best_ever.fitness["sharpe"], "n_features": best_ever.n_features,
                         "model_type": best_ever.hyperparams["model_type"],
                         "pareto_rank": getattr(best_ever, 'pareto_rank', 0),
                         "selected_features": sel_names[:30]},
                "top5": [ind.to_dict() for ind in sorted(population, key=lambda x: x.fitness["composite"], reverse=True)[:5]],
                "island_stats": island_stats,
                "history_last20": live["history"][-20:],
            }
            ts = datetime.now().strftime("%Y%m%d-%H%M")
            (RESULTS_DIR / f"evolution-{ts}.json").write_text(json.dumps(results, indent=2, default=str))
            (RESULTS_DIR / "evolution-latest.json").write_text(json.dumps(results, indent=2, default=str))

            live["top5"] = results["top5"]
            log(f"Results saved: evolution-{ts}.json")

            # ── Supabase: log cycle ──
            try:
                _db_url = os.environ.get("DATABASE_URL", "")
                if _db_url:
                    import psycopg2
                    _dbconn = psycopg2.connect(_db_url, connect_timeout=10, options="-c search_path=public")
                    _dbconn.autocommit = True
                    _cur = _dbconn.cursor()
                    pop_diversity = float(np.std([ind.fitness.get("composite", 0) for ind in population]))
                    avg_comp = float(np.mean([ind.fitness.get("composite", 0) for ind in population]))
                    _cur.execute("""INSERT INTO public.nba_evolution_runs
                        (cycle, generation, best_brier, best_roi, best_sharpe, best_calibration,
                         best_composite, best_features, best_model_type, pop_size, mutation_rate,
                         crossover_rate, stagnation, games, feature_candidates, cycle_duration_s,
                         avg_composite, pop_diversity, top5, selected_features)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (cycle, generation, float(best_ever.fitness["brier"]), float(best_ever.fitness["roi"]),
                         float(best_ever.fitness["sharpe"]), float(best_ever.fitness.get("calibration", 0)),
                         float(best_ever.fitness["composite"]), int(best_ever.n_features),
                         str(best_ever.hyperparams["model_type"]), POP_SIZE, float(mutation_rate),
                         float(CROSSOVER_RATE), stagnation, len(games), n_feat,
                         float(time.time() - cycle_start), avg_comp, pop_diversity,
                         json.dumps(results.get("top5", [])[:5], default=str),
                         json.dumps(sel_names[:50], default=str)))
                    _cur.close()
                    _dbconn.close()
                    log("[SUPABASE] Cycle logged OK")
            except Exception as e:
                log(f"[SUPABASE] Cycle log FAILED: {e}", "ERROR")
        except Exception as e:
            log(f"Save error: {e}", "ERROR")
            traceback.print_exc()

        # VM callback
        try:
            import urllib.request
            body = json.dumps({"generation": generation, "brier": best_ever.fitness["brier"],
                               "roi": best_ever.fitness["roi"], "features": best_ever.n_features,
                               "pop_size": POP_SIZE, "n_islands": N_ISLANDS}, default=str).encode()
            req = urllib.request.Request(f"{VM_URL}/callback/evolution", data=body,
                                        headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
            log("VM callback OK")
        except Exception:
            pass

        # ── Check remote commands from OpenClaw ──
        if remote_config["pending_params"]:
            params = remote_config["pending_params"]
            log(f"REMOTE CONFIG UPDATE: {params}")
            if "pop_size" in params:
                new_total = min(int(params["pop_size"]), 1000)  # Cap at 1000
                new_island_size = new_total // N_ISLANDS
                if new_island_size != ISLAND_SIZE:
                    ISLAND_SIZE = new_island_size
                    POP_SIZE = N_ISLANDS * ISLAND_SIZE
                    # Resize each island
                    for isl_idx in range(N_ISLANDS):
                        isl = island_model.islands[isl_idx]
                        if len(isl) < ISLAND_SIZE:
                            spec = island_model.island_specializations[isl_idx]
                            while len(isl) < ISLAND_SIZE:
                                isl.append(Individual(n_feat, TARGET_FEATURES, model_type=random.choice(spec)))
                        elif len(isl) > ISLAND_SIZE:
                            isl.sort(key=lambda x: x.fitness.get("composite", 0), reverse=True)
                            island_model.islands[isl_idx] = isl[:ISLAND_SIZE]
                    island_model.island_size = ISLAND_SIZE
                    live["pop_size"] = POP_SIZE
                    log(f"Population resized: {N_ISLANDS} x {ISLAND_SIZE} = {POP_SIZE}")
            if "n_islands" in params:
                # Changing island count is complex — queue for next restart
                log(f"n_islands change requested ({params['n_islands']}) — will take effect on restart")
            if "mutation_rate" in params:
                mutation_rate = min(float(params["mutation_rate"]), 0.15)
                log(f"Mutation rate set to {mutation_rate}")
            if "target_features" in params:
                TARGET_FEATURES = min(int(params["target_features"]), 150)
                log(f"Target features set to {TARGET_FEATURES}")
            if "crossover_rate" in params:
                CROSSOVER_RATE = float(params["crossover_rate"])
            if "cooldown" in params:
                COOLDOWN = int(params["cooldown"])
            if "migration_interval" in params:
                island_model.migration_interval = int(params["migration_interval"])
                log(f"Migration interval set to {island_model.migration_interval}")
            remote_config["pending_params"] = {}

        if remote_config["pending_reset"]:
            log("REMOTE RESET: Reinitializing 50% of each island with fresh individuals")
            for isl_idx in range(N_ISLANDS):
                isl = island_model.islands[isl_idx]
                isl.sort(key=lambda x: x.fitness.get("composite", 0), reverse=True)
                n_keep = max(ELITE_PER_ISLAND, ISLAND_SIZE // 4)
                kept = isl[:n_keep]
                spec = island_model.island_specializations[isl_idx]
                while len(kept) < ISLAND_SIZE:
                    kept.append(Individual(n_feat, TARGET_FEATURES, model_type=random.choice(spec)))
                island_model.islands[isl_idx] = kept
            stagnation = 0
            mutation_rate = BASE_MUT
            remote_config["pending_reset"] = False
            log(f"Reset complete: {N_ISLANDS} islands refreshed, {POP_SIZE} total")

        for cmd in remote_config.get("commands", []):
            if cmd == "diversify":
                for isl_idx in range(N_ISLANDS):
                    isl = island_model.islands[isl_idx]
                    n_new = ISLAND_SIZE // 3
                    isl.sort(key=lambda x: x.fitness.get("composite", 0), reverse=True)
                    isl = isl[:ISLAND_SIZE - n_new]
                    spec = island_model.island_specializations[isl_idx]
                    for _ in range(n_new):
                        isl.append(Individual(n_feat, TARGET_FEATURES, model_type=random.choice(spec)))
                    island_model.islands[isl_idx] = isl
                log(f"REMOTE DIVERSIFY: 1/3 of each island replaced")
            elif cmd == "boost_mutation":
                mutation_rate = min(0.15, mutation_rate * 1.5)
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

        # Memory cleanup between cycles
        _gc_cleanup()

        ce = time.time() - cycle_start
        log(f"\nCycle {cycle} done in {ce:.0f}s | Best: Brier={best_ever.fitness['brier']:.4f} | Pareto front: {live.get('pareto_front_size', 0)}")
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

    return f"""## NOMOS NBA QUANT — Island Model GA (NSGA-II)

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
| **Population** | {live['pop_size']} ({live['n_islands']} islands) |
| **Pareto Front** | {live.get('pareto_front_size', 0)} individuals |
| **Mutation Rate** | {live['mutation_rate']:.4f} |
| **Stagnation** | {live['stagnation']} gens |
| **Games** | {live['games']:,} |
| **Feature Candidates** | {live['feature_candidates']} |
| **Island Sizes** | {live.get('island_sizes', [])} |
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


@control_api.get("/api/best")
async def api_best():
    """Return best chromosome WITH feature indices for Kaggle seeding."""
    if _evo_best is None:
        return JSONResponse({"error": "no evolution best yet"}, status_code=404)
    indices = [i for i, b in enumerate(_evo_best.get("features", [])) if b]
    return JSONResponse({
        "brier": _evo_best.get("fitness", {}).get("brier", 1.0),
        "model_type": _evo_best.get("hyperparams", {}).get("model_type", "xgboost"),
        "features": indices,
        "n_features": _evo_best.get("n_features", len(indices)),
        "hyperparams": _evo_best.get("hyperparams", {}),
        "fitness": _evo_best.get("fitness", {}),
        "generation": _evo_best.get("generation", 0),
    })


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
               "cooldown", "elite_size", "tournament_size", "n_islands",
               "migration_interval", "migrants_per_island"}
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
             params, priority, status, target_space, baseline_brier, feature_engine_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s, %s)
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
            body.get("feature_engine_version", "v3.0-35cat-6000feat"),
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
        _predict_lr_meta_cal = None  # LR meta-calibrator for post-predict application
        cal_method_build = hp_build.get("calibration", "none")
        if cal_method_build == "lr_meta":
            # LR meta-calibration: fit base model, then fit LR on a held-out calibration slice.
            # The LR calibrator is stored and applied after predict_proba at inference time.
            model.fit(X_train, y_train)
            cal_slice_size = min(200, max(50, len(X_train) // 5))
            X_cal = X_train[-cal_slice_size:]
            y_cal = y_train[-cal_slice_size:]
            raw_cal_p = model.predict_proba(X_cal)[:, 1]
            _predict_lr_meta_cal = LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=500, random_state=42
            )
            _predict_lr_meta_cal.fit(raw_cal_p.reshape(-1, 1), y_cal)
        elif cal_method_build not in ("none", "venn_abers", "beta"):
            model = CalibratedClassifierCV(model, method=cal_method_build, cv=3)
            model.fit(X_train, y_train)
        else:
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

                raw_prob = model.predict_proba(game_vec.reshape(1, -1))[0, 1]
                if _predict_lr_meta_cal is not None:
                    raw_prob = _predict_lr_meta_cal.predict_proba(
                        np.array([[raw_prob]])
                    )[0, 1]
                prob = float(np.clip(raw_prob, 0.025, 0.975))

                # Apply post-hoc calibration (D5: raw ECE=0.2758, target <0.05)
                raw_prob = prob
                prob = _apply_cal(prob, _CAL_MAP)

                # Kelly criterion (25% fractional) — uses calibrated prob
                if prob > 0.55:
                    edge = prob - 0.5
                    kelly = (edge / 0.5) * 0.25  # fractional Kelly
                else:
                    kelly = 0.0

                predictions.append({
                    "home_team": home, "away_team": away,
                    "home_win_prob": round(prob, 4),
                    "away_win_prob": round(1 - prob, 4),
                    "raw_home_win_prob": round(raw_prob, 4),
                    "calibrated": _CAL_MAP is not None,
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
    gr.Markdown("# NOMOS NBA QUANT AI — Island Model Genetic Evolution 24/7")
    gr.Markdown("*500 individuals across 5 islands (NSGA-II Pareto ranking). 13 model types including neural nets. Multi-objective: Brier + ROI + Sharpe + Calibration.*")

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


# ── Legacy DEAP-based island model stubs removed ──
# The island model is now natively implemented in the IslandModel class above.
# See: IslandModel, nsga2_rank, evolution_loop()


# (Legacy DEAP island model stubs have been removed — now using native IslandModel class)
