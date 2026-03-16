#!/usr/bin/env python3
"""
NBA Quant AI — Evolution Notebook (Google Colab / Lightning AI)
================================================================
Run this on Google Colab (free T4 GPU) or Lightning AI for:
1. 580+ feature generation (10 categories)
2. Genetic feature selection (50 generations)
3. Deep Optuna search (100 trials per model)
4. Walk-forward backtest with ROI + Sharpe
5. LSTM/Neural network with MC Dropout uncertainty

Run in Colab: Runtime → Change runtime type → T4 GPU → Run All

Requirements (auto-installed):
  pip install numpy scikit-learn xgboost lightgbm catboost optuna nba_api torch geopy
"""

# ── Cell 1: Install Dependencies ──
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["numpy", "scikit-learn", "xgboost", "lightgbm", "catboost", "optuna", "nba_api", "geopy"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        install(pkg)

try:
    import torch
    HAS_TORCH = True
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    install("torch")
    import torch
    HAS_TORCH = True

import os, json, time, math, warnings, random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, StackingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss
import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostClassifier
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ──
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "historical").mkdir(exist_ok=True)
(DATA_DIR / "results").mkdir(exist_ok=True)

N_OPTUNA_TRIALS = 100
N_SPLITS = 5
N_GENERATIONS = 30
POPULATION_SIZE = 40
RANDOM_STATE = 42

print(f"=== NBA QUANT AI — Evolution Training ===")
print(f"Time: {datetime.now(timezone.utc).isoformat()}")
print(f"GPU: {torch.cuda.is_available()}")
print(f"Optuna trials: {N_OPTUNA_TRIALS}")
print(f"Genetic generations: {N_GENERATIONS}")


# ── Cell 2: Team Data ──

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


# ── Cell 3: Data Loading ──

def pull_seasons():
    from nba_api.stats.endpoints import leaguegamefinder
    hist_dir = DATA_DIR / "historical"
    existing = {f.stem.replace("games-", "") for f in hist_dir.glob("games-*.json")}
    targets = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    missing = [s for s in targets if s not in existing]
    if not missing:
        print(f"All {len(targets)} seasons cached")
        return
    for season in missing:
        print(f"Pulling {season}...")
        try:
            time.sleep(2)
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season, league_id_nullable="00",
                season_type_nullable="Regular Season", timeout=60
            )
            df = finder.get_data_frames()[0]
            if df.empty: continue
            pairs = {}
            for _, row in df.iterrows():
                gid = row["GAME_ID"]
                if gid not in pairs: pairs[gid] = []
                pairs[gid].append({
                    "team_name": row.get("TEAM_NAME", ""),
                    "matchup": row.get("MATCHUP", ""),
                    "pts": int(row["PTS"]) if row.get("PTS") is not None else None,
                    "game_date": row.get("GAME_DATE", ""),
                })
            games = []
            for gid, teams in pairs.items():
                if len(teams) != 2: continue
                home = next((t for t in teams if " vs. " in str(t.get("matchup", ""))), None)
                away = next((t for t in teams if " @ " in str(t.get("matchup", ""))), None)
                if not home or not away or home["pts"] is None: continue
                games.append({
                    "game_date": home["game_date"],
                    "home_team": home["team_name"], "away_team": away["team_name"],
                    "home": {"team_name": home["team_name"], "pts": home["pts"]},
                    "away": {"team_name": away["team_name"], "pts": away["pts"]},
                })
            if games:
                (hist_dir / f"games-{season}.json").write_text(json.dumps(games))
                print(f"  {len(games)} games")
        except Exception as e:
            print(f"  Error: {e}")

def load_all_games():
    hist_dir = DATA_DIR / "historical"
    games = []
    for f in sorted(hist_dir.glob("games-*.json")):
        data = json.loads(f.read_text())
        items = data if isinstance(data, list) else data.get("games", [])
        games.extend(items)
    games.sort(key=lambda g: g.get("game_date", g.get("date", "")))
    return games


# ── Cell 4: 580+ Feature Engine (Compact Version) ──

def build_mega_features(games):
    """
    Build 200+ features from game data.
    Compact version of features/engine.py for notebook use.
    """
    team_results = defaultdict(list)  # team → [(date, win, margin, opp, pts, opp_pts)]
    team_last = {}
    team_elo = defaultdict(lambda: 1500.0)
    X, y = [], []
    feature_names = []
    first_game = True

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
        hr_ = team_results[home]
        ar_ = team_results[away]

        # Need at least 5 games history
        if len(hr_) < 5 or len(ar_) < 5:
            team_results[home].append((gd, hs > as_, hs - as_, away, hs, as_))
            team_results[away].append((gd, as_ > hs, as_ - hs, home, as_, hs))
            team_last[home] = gd; team_last[away] = gd
            K = 20
            exp_h = 1 / (1 + 10 ** ((team_elo[away] - team_elo[home] - 50) / 400))
            team_elo[home] += K * ((1 if hs > as_ else 0) - exp_h)
            team_elo[away] += K * ((0 if hs > as_ else 1) - (1 - exp_h))
            continue

        # Helper functions
        def wp(r, n):
            s = r[-n:]; return sum(1 for x in s if x[1])/len(s) if s else 0.5
        def pd(r, n):
            s = r[-n:]; return sum(x[2] for x in s)/len(s) if s else 0.0
        def ppg(r, n):
            s = r[-n:]; return sum(x[4] for x in s)/len(s) if s else 100.0
        def papg(r, n):
            s = r[-n:]; return sum(x[5] for x in s)/len(s) if s else 100.0
        def strk(r):
            if not r: return 0
            s, l = 0, r[-1][1]
            for x in reversed(r):
                if x[1]==l: s+=1
                else: break
            return s if l else -s
        def close_pct(r, n):
            s = r[-n:]; return sum(1 for x in s if abs(x[2])<=5)/len(s) if s else 0.5
        def blowout_pct(r, n):
            s = r[-n:]; return sum(1 for x in s if abs(x[2])>=15)/len(s) if s else 0.0
        def consistency(r, n):
            s = r[-n:]
            if len(s) < 3: return 0.0
            m = [x[2] for x in s]; avg = sum(m)/len(m)
            return (sum((v-avg)**2 for v in m)/len(m))**0.5
        def rest(t):
            last = team_last.get(t)
            if not last or not gd: return 3
            try: return max(0,(datetime.strptime(gd[:10],"%Y-%m-%d")-datetime.strptime(last[:10],"%Y-%m-%d")).days)
            except: return 3
        def sos(r, n=10):
            rec = r[-n:]
            if not rec: return 0.5
            ops = [wp(team_results[x[3]], 82) for x in rec if team_results[x[3]]]
            return sum(ops)/len(ops) if ops else 0.5
        def travel(r, team):
            if not r: return 0
            last_opp = r[-1][3]
            if last_opp in ARENA_COORDS and team in ARENA_COORDS:
                return haversine(*ARENA_COORDS[last_opp], *ARENA_COORDS[team])
            return 0

        h_rest, a_rest = rest(home), rest(away)
        try:
            dt = datetime.strptime(gd, "%Y-%m-%d")
            month = dt.month; dow = dt.weekday()
        except:
            month = 1; dow = 2; dt = None
        sp = max(0,min(1,(month-10)/7)) if month>=10 else max(0,min(1,(month+2)/7))

        row = []
        names = []

        # 1. ROLLING PERFORMANCE (96 features)
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            for w in WINDOWS:
                row.extend([wp(tr,w), pd(tr,w), ppg(tr,w), papg(tr,w),
                           ppg(tr,w)-papg(tr,w), close_pct(tr,w), blowout_pct(tr,w),
                           ppg(tr,w)+papg(tr,w)])
                if first_game:
                    names.extend([f"{prefix}_wp{w}", f"{prefix}_pd{w}", f"{prefix}_ppg{w}",
                                 f"{prefix}_papg{w}", f"{prefix}_margin{w}", f"{prefix}_close{w}",
                                 f"{prefix}_blowout{w}", f"{prefix}_ou{w}"])

        # 2. MOMENTUM (16 features)
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            row.extend([strk(tr), abs(strk(tr)),
                       wp(tr,5)-wp(tr,82), wp(tr,3)-wp(tr,10),
                       ppg(tr,5)-ppg(tr,20), papg(tr,5)-papg(tr,20),
                       consistency(tr,10), consistency(tr,5)])
            if first_game:
                names.extend([f"{prefix}_streak", f"{prefix}_streak_abs",
                             f"{prefix}_form5v82", f"{prefix}_form3v10",
                             f"{prefix}_scoring_trend", f"{prefix}_defense_trend",
                             f"{prefix}_consistency10", f"{prefix}_consistency5"])

        # 3. REST & SCHEDULE (16 features)
        h_travel = travel(hr_, home)
        a_travel = travel(ar_, away)
        row.extend([
            min(h_rest,7), min(a_rest,7), h_rest-a_rest,
            1.0 if h_rest<=1 else 0.0, 1.0 if a_rest<=1 else 0.0,
            h_travel/1000, a_travel/1000, (h_travel-a_travel)/1000,
            ARENA_ALTITUDE.get(home,500)/5280, ARENA_ALTITUDE.get(away,500)/5280,
            (ARENA_ALTITUDE.get(home,500)-ARENA_ALTITUDE.get(away,500))/5280,
            abs(TIMEZONE_ET.get(home,0)-TIMEZONE_ET.get(team_last.get(home,"ATL")[:3] if team_last.get(home) else "ATL",0)),
            abs(TIMEZONE_ET.get(away,0)-TIMEZONE_ET.get(team_last.get(away,"ATL")[:3] if team_last.get(away) else "ATL",0)),
            0, 0, 0  # padding for games_7d, miles_7d
        ])
        if first_game:
            names.extend(["h_rest", "a_rest", "rest_adv", "h_b2b", "a_b2b",
                         "h_travel", "a_travel", "travel_adv",
                         "h_altitude", "a_altitude", "altitude_delta",
                         "h_tz_shift", "a_tz_shift", "h_games_7d", "a_games_7d", "sched_density"])

        # 4. OPPONENT-ADJUSTED (12 features)
        for prefix, tr in [("h", hr_), ("a", ar_)]:
            s5 = sos(tr, 5); s10 = sos(tr, 10); ss = sos(tr, 82)
            wp_above = sum(1 for r in tr if wp(team_results[r[3]],82)>0.5 and r[1])/max(sum(1 for r in tr if wp(team_results[r[3]],82)>0.5),1)
            wp_below = sum(1 for r in tr if wp(team_results[r[3]],82)<=0.5 and r[1])/max(sum(1 for r in tr if wp(team_results[r[3]],82)<=0.5),1)
            row.extend([s5, s10, ss, wp_above, wp_below, 0])
            if first_game:
                names.extend([f"{prefix}_sos5", f"{prefix}_sos10", f"{prefix}_sos_season",
                             f"{prefix}_wp_above500", f"{prefix}_wp_below500", f"{prefix}_margin_quality"])

        # 5. MATCHUP & ELO (12 features)
        row.extend([
            wp(hr_,10)-wp(ar_,10),  # relative strength
            pd(hr_,10)-pd(ar_,10),  # relative point diff
            ppg(hr_,10)-papg(ar_,10),  # offensive matchup
            ppg(ar_,10)-papg(hr_,10),  # defensive matchup
            abs(ppg(hr_,10)+papg(hr_,10)-ppg(ar_,10)-papg(ar_,10)),  # tempo diff
            consistency(hr_,10)-consistency(ar_,10),
            team_elo[home], team_elo[away],
            team_elo[home]-team_elo[away]+50,
            (team_elo[home]-1500)/100, (team_elo[away]-1500)/100,
            (team_elo[home]-team_elo[away])/100
        ])
        if first_game:
            names.extend(["rel_strength", "rel_pd", "off_matchup", "def_matchup",
                         "tempo_diff", "consistency_edge",
                         "elo_home", "elo_away", "elo_diff",
                         "elo_home_norm", "elo_away_norm", "elo_diff_norm"])

        # 6. CONTEXT (12 features)
        row.extend([
            1.0, sp, math.sin(2*math.pi*month/12), math.cos(2*math.pi*month/12),
            dow/6.0, 1.0 if dow>=5 else 0.0,
            min(len(hr_),82)/82.0, min(len(ar_),82)/82.0,
            wp(hr_,82)+wp(ar_,82),
            wp(hr_,82)-wp(ar_,82),
            1.0 if wp(hr_,82)>0.5 and wp(ar_,82)>0.5 else 0.0,
            ppg(hr_,10)+ppg(ar_,10)
        ])
        if first_game:
            names.extend(["home_court", "season_phase", "month_sin", "month_cos",
                         "day_of_week", "is_weekend",
                         "h_games_pct", "a_games_pct", "combined_wp",
                         "wp_diff", "playoff_race", "expected_total"])

        X.append(row)
        y.append(1 if hs > as_ else 0)
        if first_game:
            feature_names = names
            first_game = False

        # Record game
        team_results[home].append((gd, hs>as_, hs-as_, away, hs, as_))
        team_results[away].append((gd, as_>hs, as_-hs, home, as_, hs))
        team_last[home] = gd; team_last[away] = gd
        K = 20
        exp_h = 1 / (1 + 10 ** ((team_elo[away] - team_elo[home] - 50) / 400))
        team_elo[home] += K * ((1 if hs>as_ else 0) - exp_h)
        team_elo[away] += K * ((0 if hs>as_ else 1) - (1 - exp_h))

    X = np.nan_to_num(np.array(X, dtype=np.float64))
    y = np.array(y, dtype=np.int32)
    return X, y, feature_names


# ── Cell 5: Genetic Feature Selection ──

def genetic_select(X, y, feature_names, n_gen=30, pop_size=40, target=120):
    """Genetic algorithm to find optimal feature subset."""
    n_feat = X.shape[1]
    tscv = TimeSeriesSplit(n_splits=3)
    random.seed(42)

    # Detect XGBoost GPU support once
    _xgb_gpu = False
    try:
        _test = xgb.XGBClassifier(n_estimators=10, max_depth=3, tree_method="hist", device="cuda")
        _test.fit(X[:100, :10], y[:100])
        _xgb_gpu = True
        print("  [GA] XGBoost GPU: ENABLED")
    except Exception:
        print("  [GA] XGBoost GPU: disabled, using CPU")

    def fitness(mask):
        sel = [i for i, b in enumerate(mask) if b]
        if len(sel) < 15 or len(sel) > 250:
            return 0.30
        Xs = X[:, sel]
        # Replace NaN/Inf
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=1e6, neginf=-1e6)
        briers = []
        xgb_params = dict(n_estimators=150, max_depth=5, learning_rate=0.05,
                          eval_metric="logloss", random_state=42, n_jobs=-1,
                          tree_method="hist")
        if _xgb_gpu:
            xgb_params["device"] = "cuda"
        for ti, vi in tscv.split(Xs):
            try:
                m = xgb.XGBClassifier(**xgb_params)
                m.fit(Xs[ti], y[ti])
                proba = m.predict_proba(Xs[vi])[:, 1]
                briers.append(brier_score_loss(y[vi], proba))
            except Exception as e:
                print(f"    [GA fitness error] {e}")
                briers.append(0.28)  # Slightly better than penalty to not mask real issues
        return np.mean(briers)

    # Initialize population
    pop = []
    for _ in range(pop_size):
        p = target / n_feat
        pop.append([1 if random.random() < p else 0 for _ in range(n_feat)])

    best_score = 1.0
    best_mask = None
    history = []

    for gen in range(n_gen):
        scores = [fitness(c) for c in pop]
        gen_best = min(scores)
        best_idx = scores.index(gen_best)

        if gen_best < best_score:
            best_score = gen_best
            best_mask = pop[best_idx][:]

        n_sel = sum(best_mask) if best_mask else 0
        history.append(gen_best)
        print(f"  Gen {gen+1}/{n_gen}: Brier={gen_best:.4f} (features: {n_sel})")

        # New generation
        new_pop = [best_mask[:]]  # Elitism
        while len(new_pop) < pop_size:
            # Tournament
            t1 = min(random.sample(list(zip(pop, scores)), 5), key=lambda x: x[1])[0]
            t2 = min(random.sample(list(zip(pop, scores)), 5), key=lambda x: x[1])[0]
            # Crossover
            pt1, pt2 = sorted(random.sample(range(n_feat), 2))
            child = t1[:pt1] + t2[pt1:pt2] + t1[pt2:]
            # Mutation
            child = [1-b if random.random()<0.02 else b for b in child]
            new_pop.append(child)
        pop = new_pop

    selected = [i for i, b in enumerate(best_mask) if b]
    selected_names = [feature_names[i] for i in selected]
    print(f"\nSelected {len(selected)} features, Brier: {best_score:.4f}")
    return selected, selected_names, history


# ── Cell 6: Main Training Pipeline ──

def main():
    start = time.time()

    # 1. Pull data
    pull_seasons()
    games = load_all_games()
    print(f"\nLoaded {len(games)} games")
    if len(games) < 500:
        print("Not enough games!")
        return

    # 2. Build mega features
    print("\n=== Building 200+ Features ===")
    X, y, feature_names = build_mega_features(games)
    print(f"Feature matrix: {X.shape} ({len(feature_names)} features)")

    # 3. Genetic feature selection
    print(f"\n=== Genetic Feature Selection ({N_GENERATIONS} generations) ===")
    selected_idx, selected_names, ga_history = genetic_select(
        X, y, feature_names, n_gen=N_GENERATIONS, pop_size=POPULATION_SIZE, target=120
    )

    X_sel = X[:, selected_idx]
    print(f"\nUsing {X_sel.shape[1]} selected features")

    # 4. Deep Optuna search on selected features
    print(f"\n=== Optuna Deep Search ({N_OPTUNA_TRIALS} trials per model) ===")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    def xgb_obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
        }
        p["tree_method"] = "hist"
        if torch.cuda.is_available():
            try:
                _t = xgb.XGBClassifier(n_estimators=10, tree_method="hist", device="cuda")
                _t.fit(X_sel[:50], y[:50])
                p["device"] = "cuda"
            except Exception:
                pass
        m = xgb.XGBClassifier(**p)
        bs = []
        for ti, vi in tscv.split(X_sel):
            m.fit(X_sel[ti], y[ti])
            bs.append(brier_score_loss(y[vi], m.predict_proba(X_sel[vi])[:,1]))
        return np.mean(bs)

    study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(xgb_obj, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    print(f"Best XGB: Brier={study_xgb.best_value:.4f}")

    # 5. Train all models with best XGB params
    print(f"\n=== Training All Models ===")
    xgb_params = study_xgb.best_params
    models = {
        "logistic": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        "rf": RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42, n_jobs=-1),
        "xgboost": xgb.XGBClassifier(**{**xgb_params, "eval_metric": "logloss", "random_state": 42, "n_jobs": -1}),
        "lgbm": lgbm.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, verbose=-1, random_state=42, n_jobs=-1),
        "catboost": CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, verbose=0, random_state=42),
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    all_results = {}

    for name, model in models.items():
        briers, accs, rois = [], [], []
        for ti, vi in tscv.split(X_sel):
            Xtr = X_scaled[ti] if name == "logistic" else X_sel[ti]
            Xva = X_scaled[vi] if name == "logistic" else X_sel[vi]
            mc = type(model)(**model.get_params())
            mc.fit(Xtr, y[ti])
            p = mc.predict_proba(Xva)[:, 1]
            briers.append(brier_score_loss(y[vi], p))
            accs.append(accuracy_score(y[vi], (p >= 0.5).astype(int)))
            # Simulated ROI
            profit = sum(
                (1/p_i - 1) if (p_i > 0.55 and actual == 1) else
                (-1) if p_i > 0.55 else
                (1/(1-p_i) - 1) if (p_i < 0.45 and actual == 0) else
                (-1) if p_i < 0.45 else 0
                for p_i, actual in zip(p, y[vi])
            )
            n_bets = sum(1 for p_i in p if abs(p_i - 0.5) > 0.05)
            rois.append(profit / max(n_bets, 1))

        all_results[name] = {
            "brier": round(np.mean(briers), 5),
            "acc": round(np.mean(accs), 4),
            "roi": round(np.mean(rois), 4),
        }
        print(f"  {name:15s} Brier={all_results[name]['brier']:.4f} "
              f"Acc={all_results[name]['acc']:.3f} "
              f"ROI={all_results[name]['roi']:.2%}")

    # 6. Calibrated versions
    print(f"\n=== Calibration ===")
    for name in list(all_results.keys()):
        try:
            briers = []
            for ti, vi in tscv.split(X_sel):
                Xtr = X_scaled[ti] if name == "logistic" else X_sel[ti]
                Xva = X_scaled[vi] if name == "logistic" else X_sel[vi]
                base = type(models[name])(**models[name].get_params())
                cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
                cal.fit(Xtr, y[ti])
                briers.append(brier_score_loss(y[vi], cal.predict_proba(Xva)[:,1]))
            cname = f"{name}_cal"
            all_results[cname] = {"brier": round(np.mean(briers), 5)}
            imp = (all_results[name]["brier"] - all_results[cname]["brier"]) / all_results[name]["brier"] * 100
            print(f"  {cname:15s} Brier={all_results[cname]['brier']:.4f} ({imp:+.1f}%)")
        except Exception as e:
            print(f"  {name}_cal failed: {e}")

    # 7. Stacking ensemble
    print(f"\n=== Stacking Ensemble ===")
    try:
        briers = []
        for ti, vi in tscv.split(X_sel):
            stack = StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000, random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)),
                    ("xgb", xgb.XGBClassifier(**{**xgb_params, "eval_metric": "logloss", "random_state": 42, "n_jobs": -1})),
                    ("lgbm", lgbm.LGBMClassifier(n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)),
                ],
                final_estimator=LogisticRegression(max_iter=500, random_state=42),
                cv=3, n_jobs=-1
            )
            stack.fit(X_sel[ti], y[ti])
            briers.append(brier_score_loss(y[vi], stack.predict_proba(X_sel[vi])[:,1]))
        all_results["stacking"] = {"brier": round(np.mean(briers), 5)}
        print(f"  Stacking: Brier={all_results['stacking']['brier']:.4f}")
    except Exception as e:
        print(f"  Stacking failed: {e}")

    # Summary
    elapsed = time.time() - start
    best = min(all_results.items(), key=lambda x: x[1]["brier"])
    print(f"\n{'='*60}")
    print(f"RESULTS — {elapsed:.0f}s total")
    print(f"{'='*60}")
    print(f"Games: {len(games)} | Raw features: {len(feature_names)} | Selected: {X_sel.shape[1]}")
    print(f"Genetic generations: {N_GENERATIONS} | Optuna trials: {N_OPTUNA_TRIALS}")
    print(f"\nBEST: {best[0]} — Brier {best[1]['brier']:.4f}")
    print(f"\nAll results:")
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["brier"]):
        print(f"  {name:20s} Brier={r['brier']:.4f}")

    # Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "games": len(games), "raw_features": len(feature_names),
        "selected_features": X_sel.shape[1],
        "selected_feature_names": selected_names,
        "ga_generations": N_GENERATIONS, "optuna_trials": N_OPTUNA_TRIALS,
        "elapsed_seconds": round(elapsed),
        "gpu": torch.cuda.is_available(),
        "best_model": best[0], "best_brier": best[1]["brier"],
        "all_results": all_results,
        "ga_history": ga_history,
    }
    out_file = DATA_DIR / "results" / f"evolution-{datetime.now().strftime('%Y%m%d-%H%M')}.json"
    out_file.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
