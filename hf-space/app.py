#!/usr/bin/env python3
"""
NOMOS NBA QUANT AI — REAL Genetic Evolution (HF Space)
========================================================
RUNS 24/7 on HuggingFace Space (2 vCPU / 16GB RAM).

NOT a fake LLM wrapper. REAL ML:
  - Population of 60 individuals (feature mask + hyperparams)
  - Walk-forward backtest fitness (Brier + ROI + Sharpe + Calibration)
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
from collections import defaultdict

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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgbm

import gradio as gr

# ── Paths ──
DATA_DIR = Path("data")
HIST_DIR = DATA_DIR / "historical"
RESULTS_DIR = DATA_DIR / "results"
STATE_DIR = DATA_DIR / "evolution-state"
for d in [DATA_DIR, HIST_DIR, RESULTS_DIR, STATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
    try:
        from nba_api.stats.endpoints import leaguegamefinder
    except ImportError:
        log("nba_api not installed — using cached data", "WARN")
        return
    existing = {f.stem.replace("games-", "") for f in HIST_DIR.glob("games-*.json")}
    targets = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    for season in [s for s in targets if s not in existing]:
        log(f"Pulling {season}...")
        try:
            time.sleep(3)
            from nba_api.stats.endpoints import leaguegamefinder
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season, league_id_nullable="00",
                season_type_nullable="Regular Season", timeout=60)
            df = finder.get_data_frames()[0]
            if df.empty: continue
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
        except Exception as e:
            log(f"  Error: {e}", "ERROR")


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
            "n_estimators": random.randint(100, 600),
            "max_depth": random.randint(3, 10),
            "learning_rate": 10 ** random.uniform(-2.5, -0.5),
            "subsample": random.uniform(0.5, 1.0),
            "colsample_bytree": random.uniform(0.3, 1.0),
            "min_child_weight": random.randint(1, 15),
            "reg_alpha": 10 ** random.uniform(-6, 1),
            "reg_lambda": 10 ** random.uniform(-6, 1),
            "model_type": random.choice(["xgboost", "lightgbm", "catboost", "random_forest", "extra_trees"]),
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

    def mutate(self, rate=0.03):
        for i in range(len(self.features)):
            if random.random() < rate: self.features[i] = 1 - self.features[i]
        if random.random() < 0.15: self.hyperparams["n_estimators"] = max(50, self.hyperparams["n_estimators"] + random.randint(-100, 100))
        if random.random() < 0.15: self.hyperparams["max_depth"] = max(2, min(12, self.hyperparams["max_depth"] + random.randint(-2, 2)))
        if random.random() < 0.15: self.hyperparams["learning_rate"] = max(0.001, min(0.5, self.hyperparams["learning_rate"] * 10 ** random.uniform(-0.3, 0.3)))
        if random.random() < 0.08: self.hyperparams["model_type"] = random.choice(["xgboost", "lightgbm", "catboost", "random_forest", "extra_trees"])
        if random.random() < 0.05: self.hyperparams["calibration"] = random.choice(["isotonic", "sigmoid", "none"])
        self.n_features = sum(self.features)


# ═══════════════════════════════════════════════════════
# FITNESS EVALUATION
# ═══════════════════════════════════════════════════════

def evaluate(ind, X, y, n_splits=5):
    selected = ind.selected_indices()
    if len(selected) < 20 or len(selected) > 400:
        ind.fitness = {"brier": 0.30, "roi": -0.10, "sharpe": -1.0, "calibration": 0.15, "composite": -1.0}
        return
    X_sub = np.nan_to_num(X[:, selected], nan=0.0, posinf=1e6, neginf=-1e6)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    hp = ind.hyperparams
    model = _build(hp)
    if model is None: ind.fitness["composite"] = -1.0; return
    briers, rois, all_p, all_y = [], [], [], []
    for ti, vi in tscv.split(X_sub):
        try:
            m = type(model)(**model.get_params())
            if hp["calibration"] != "none":
                m = CalibratedClassifierCV(m, method=hp["calibration"], cv=3)
            m.fit(X_sub[ti], y[ti])
            p = m.predict_proba(X_sub[vi])[:, 1]
            briers.append(brier_score_loss(y[vi], p))
            rois.append(_bet(p, y[vi]))
            all_p.extend(p); all_y.extend(y[vi])
        except Exception:
            briers.append(0.28); rois.append(-0.05)
    ab, ar = np.mean(briers), np.mean(rois)
    sh = np.mean(rois) / max(np.std(rois), 0.01) if len(rois) > 1 else 0.0
    ce = _ece(np.array(all_p), np.array(all_y)) if all_p else 0.15
    ind.fitness = {"brier": round(ab, 5), "roi": round(ar, 4), "sharpe": round(sh, 4),
                   "calibration": round(ce, 4),
                   "composite": round(0.40 * (1 - ab) + 0.25 * max(0, ar) + 0.20 * max(0, sh / 3) + 0.15 * (1 - ce), 5)}


def _build(hp):
    try:
        mt = hp["model_type"]
        if mt == "xgboost":
            return xgb.XGBClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                                     learning_rate=hp["learning_rate"], subsample=hp["subsample"],
                                     colsample_bytree=hp["colsample_bytree"], min_child_weight=hp["min_child_weight"],
                                     reg_alpha=hp["reg_alpha"], reg_lambda=hp["reg_lambda"],
                                     eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist")
        elif mt == "lightgbm":
            return lgbm.LGBMClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                                       learning_rate=hp["learning_rate"], subsample=hp["subsample"],
                                       num_leaves=min(2 ** hp["max_depth"] - 1, 63),
                                       reg_alpha=hp["reg_alpha"], reg_lambda=hp["reg_lambda"],
                                       min_child_samples=20, feature_fraction=0.7,
                                       verbose=-1, random_state=42, n_jobs=-1)
        elif mt == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(iterations=hp["n_estimators"], depth=min(hp["max_depth"], 8),
                                      learning_rate=hp["learning_rate"], l2_leaf_reg=hp["reg_lambda"],
                                      verbose=0, random_state=42, thread_count=-1)
        elif mt == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                                          min_samples_leaf=max(int(hp["min_child_weight"]), 5),
                                          max_features="sqrt", random_state=42, n_jobs=-1)
        elif mt == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                                        min_samples_leaf=max(int(hp["min_child_weight"]), 5),
                                        max_features="sqrt", random_state=42, n_jobs=-1)
        else:
            return xgb.XGBClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                                     learning_rate=hp["learning_rate"], eval_metric="logloss",
                                     random_state=42, n_jobs=-1, tree_method="hist")
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=min(hp["n_estimators"], 200), max_depth=hp["max_depth"],
                                          learning_rate=hp["learning_rate"], random_state=42)


def _bet(probs, actuals, edge=0.05):
    stake, profit, n = 10, 0, 0
    for p, a in zip(probs, actuals):
        if p > 0.5 + edge:
            n += 1; profit += stake * (1 / p - 1) if a == 1 else -stake
        elif p < 0.5 - edge:
            n += 1; profit += stake * (1 / (1 - p) - 1) if a == 0 else -stake
    return profit / (n * stake) if n > 0 else 0.0


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

POP_SIZE = 150           # ↑ from 60 — more genetic diversity
ELITE_SIZE = 10          # ↑ from 5 — preserve more good solutions
BASE_MUT = 0.10          # ↑ from 0.03 — CRITICAL: explore more
CROSSOVER_RATE = 0.85    # ↑ from 0.7 — more recombination
TARGET_FEATURES = 200    # ↑ from 100 — use more of the feature space
N_SPLITS = 5
GENS_PER_CYCLE = 10
COOLDOWN = 45            # ↓ from 60 — iterate faster
TOURNAMENT_SIZE = 5      # ↓ from 7 — less selection pressure
DIVERSITY_RESTART = 30   # NEW: restart portion of pop after N stagnant gens


def evolution_loop():
    """Main 24/7 genetic evolution loop — runs in background thread."""
    log("=" * 60)
    log("REAL GENETIC EVOLUTION LOOP v3 — STARTING")
    log(f"Pop: {POP_SIZE} | Target features: {TARGET_FEATURES} | Gens/cycle: {GENS_PER_CYCLE}")
    log("=" * 60)

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
    n_feat = X.shape[1]
    live["feature_candidates"] = n_feat
    log(f"Feature matrix: {X.shape} ({n_feat} features)")

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
            if e.hyperparams.get("model_type") not in ["xgboost", "lightgbm", "catboost", "random_forest", "extra_trees"]:
                e.hyperparams["model_type"] = random.choice(["xgboost", "lightgbm", "catboost", "random_forest", "extra_trees"])
            population.append(e)
        log(f"Kept {len(population)} elites, generating {POP_SIZE - len(population)} fresh individuals")

    if not population or needs_reset:
        while len(population) < POP_SIZE:
            population.append(Individual(n_feat, TARGET_FEATURES))
        log(f"Population ready: {POP_SIZE} individuals, {n_feat} feature candidates")

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

            # Evaluate
            for i, ind in enumerate(population):
                evaluate(ind, X, y, N_SPLITS)

            # Sort
            population.sort(key=lambda x: x.fitness["composite"], reverse=True)
            best = population[0]

            # Track best
            prev_brier = best_ever.fitness["brier"] if best_ever else 1.0
            if best_ever is None or best.fitness["composite"] > best_ever.fitness["composite"]:
                best_ever = Individual.__new__(Individual)
                best_ever.features = best.features[:]
                best_ever.hyperparams = dict(best.hyperparams)
                best_ever.fitness = dict(best.fitness)
                best_ever.n_features = best.n_features
                best_ever.generation = generation

            # Stagnation detection
            if abs(best.fitness["brier"] - prev_brier) < 0.0005:
                stagnation += 1
            else:
                stagnation = 0

            if stagnation >= 5:
                mutation_rate = min(0.15, mutation_rate * 1.5)
            elif stagnation == 0:
                mutation_rate = max(BASE_MUT, mutation_rate * 0.9)

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
                f"Model={best.hyperparams['model_type']} Stag={stagnation} ({ge:.0f}s)")

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

            while len(new_pop) < POP_SIZE:
                cs = random.sample(population, min(TOURNAMENT_SIZE, len(population)))
                p1 = max(cs, key=lambda x: x.fitness["composite"])
                cs2 = random.sample(population, min(TOURNAMENT_SIZE, len(population)))
                p2 = max(cs2, key=lambda x: x.fitness["composite"])
                child = Individual.crossover(p1, p2) if random.random() < CROSSOVER_RATE else Individual.__new__(Individual)
                if not hasattr(child, 'features') or child.features is None:
                    child.features = p1.features[:]; child.hyperparams = dict(p1.hyperparams)
                    child.fitness = dict(p1.fitness); child.n_features = p1.n_features; child.generation = generation
                child.mutate(mutation_rate)
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


# ── Launch ──
_thread = threading.Thread(target=evolution_loop, daemon=True, name="GeneticEvolution")
_thread.start()
log("Genetic evolution thread launched")

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
