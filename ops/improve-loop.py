#!/usr/bin/env python3
"""
NBA Quant — Continuous Improvement Engine.

⚠️  THIS SCRIPT MUST RUN ON HF SPACES (16GB RAM), NOT ON VM (1GB RAM).
    Deploy via: huggingface_hub.upload_file() to LBJLincoln/nomos-nba-quant

Runs forever. Each cycle:
1. EXPAND DATA    — Pull more seasons from nba_api (target: 8 seasons = 10K+ games)
2. HYPERPARAM     — Optuna search on XGBoost/LightGBM/RF
3. NEW MODELS     — Try CatBoost, SVM, Voting, Stacking
4. FEATURE SELECT — Recursive elimination, importance ranking
5. CALIBRATE      — Isotonic + Platt + beta calibration comparison
6. EVALUATE       — TimeSeriesSplit CV, Brier score, track improvements
7. PERSIST        — Save best model, log results, sync to website

Target: Brier < 0.19 (top-tier sports betting models)
"""

import os, sys, json, time, math, signal, pickle, warnings
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore", category=UserWarning)

# ── VM GUARD ──
try:
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemTotal'):
                kb = int(line.split()[1])
                if kb < 2_000_000 and not os.environ.get('FORCE_LOCAL'):
                    print(f"⚠️  ABORT: {kb//1024}MB RAM. Run on HF Spaces, not VM.")
                    sys.exit(1)
except: pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ops"))

# Load env
for env_file in [ROOT / ".env.local", Path("/home/lahargnedebartoli/mon-ipad/.env.local")]:
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

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
IMPROVE_DIR = DATA_DIR / "improve"
IMPROVE_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = IMPROVE_DIR / "improve-log.jsonl"
BEST_BRIER_FILE = IMPROVE_DIR / "best-brier.json"

# Global state
_shutdown = False
_cycle = 0

def signal_handler(sig, frame):
    global _shutdown
    _shutdown = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).isoformat()
    entry = {"ts": ts, "level": level, "msg": msg}
    print(f"[{ts[:19]}] [{level:5s}] {msg}")
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# ── Step 1: Expand Data ──────────────────────────────────────────────────────

def expand_data() -> int:
    """Pull more seasons from nba_api if available. Target: 8 seasons."""
    hist_dir = DATA_DIR / "historical"
    hist_dir.mkdir(parents=True, exist_ok=True)

    existing_seasons = set()
    for f in hist_dir.glob("games-*.json"):
        season = f.stem.replace("games-", "")
        existing_seasons.add(season)

    log(f"Existing seasons: {sorted(existing_seasons)}")

    target_seasons = [
        "2018-19", "2019-20", "2020-21", "2021-22",
        "2022-23", "2023-24", "2024-25", "2025-26"
    ]
    missing = [s for s in target_seasons if s not in existing_seasons]

    if not missing:
        log("All 8 target seasons present")
        return 0

    log(f"Missing seasons: {missing}")
    new_games = 0

    try:
        from nba_api.stats.endpoints import leaguegamefinder
        from nba_api.stats.static import teams as nba_teams
    except ImportError:
        log("nba_api not installed — cannot expand data", "WARN")
        return 0

    for season_str in missing:
        if _shutdown:
            break

        # Convert "2024-25" to nba_api format "2024-25"
        log(f"Pulling season {season_str}...")
        try:
            time.sleep(2)  # Rate limit
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season_str,
                league_id_nullable="00",
                season_type_nullable="Regular Season",
                timeout=60
            )
            games_df = finder.get_data_frames()[0]

            if games_df.empty:
                log(f"  No data for {season_str}", "WARN")
                continue

            # Convert to our format: group by GAME_ID to get home/away pairs
            game_pairs = {}
            for _, row in games_df.iterrows():
                gid = row["GAME_ID"]
                if gid not in game_pairs:
                    game_pairs[gid] = []
                game_pairs[gid].append({
                    "team_id": row.get("TEAM_ID"),
                    "team_name": row.get("TEAM_NAME", ""),
                    "team_abbreviation": row.get("TEAM_ABBREVIATION", ""),
                    "matchup": row.get("MATCHUP", ""),
                    "pts": int(row["PTS"]) if row.get("PTS") is not None else None,
                    "game_date": row.get("GAME_DATE", ""),
                    "wl": row.get("WL", ""),
                })

            season_games = []
            for gid, teams in game_pairs.items():
                if len(teams) != 2:
                    continue
                # Determine home/away from matchup string ("vs." = home, "@" = away)
                home, away = None, None
                for t in teams:
                    if " vs. " in str(t.get("matchup", "")):
                        home = t
                    elif " @ " in str(t.get("matchup", "")):
                        away = t

                if not home or not away or home["pts"] is None or away["pts"] is None:
                    continue

                season_games.append({
                    "game_id": gid,
                    "game_date": home.get("game_date", ""),
                    "home_team": home["team_name"],
                    "away_team": away["team_name"],
                    "home_abbrev": home.get("team_abbreviation", ""),
                    "away_abbrev": away.get("team_abbreviation", ""),
                    "home_score": home["pts"],
                    "away_score": away["pts"],
                    "home": {"team_name": home["team_name"], "pts": home["pts"]},
                    "away": {"team_name": away["team_name"], "pts": away["pts"]},
                })

            if season_games:
                out_file = hist_dir / f"games-{season_str}.json"
                out_file.write_text(json.dumps(season_games, indent=2))
                log(f"  Saved {len(season_games)} games for {season_str}")
                new_games += len(season_games)

        except Exception as e:
            log(f"  Error pulling {season_str}: {e}", "ERROR")
            time.sleep(5)

    return new_games


# ── Step 2: Load All Data ─────────────────────────────────────────────────────

def load_all_games() -> List[dict]:
    """Load games from all seasons."""
    all_games = []
    hist_dir = DATA_DIR / "historical"

    for f in sorted(hist_dir.glob("games-*.json")):
        try:
            data = json.loads(f.read_text())
            items = data if isinstance(data, list) else data.get("games", [])
            all_games.extend(items)
        except Exception as e:
            log(f"Error loading {f.name}: {e}", "WARN")

    # Sort by date
    all_games.sort(key=lambda g: g.get("game_date", g.get("date", g.get("commence_time", ""))))
    log(f"Loaded {len(all_games)} total games from {len(list(hist_dir.glob('games-*.json')))} seasons")
    return all_games


# ── Step 3: Build Features ────────────────────────────────────────────────────

def build_features_from_games(games: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix from raw games. Simplified but complete.
    Uses rolling stats computed chronologically (no data leakage).
    """
    from collections import defaultdict

    team_results = defaultdict(list)  # team -> [(date, won, margin, opp)]
    team_last_game = {}

    X_rows = []
    y_rows = []

    TEAM_ALIASES = {}
    try:
        # Try importing from karpathy-loop
        sys.path.insert(0, str(ROOT / "ops"))
        # Quick inline resolution
        pass
    except Exception:
        pass

    def resolve(name):
        """Simple team name resolution."""
        # Common abbreviations
        abbrevs = {
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
        if name in abbrevs:
            return abbrevs[name]
        # Check if already an abbreviation
        if len(name) == 3 and name.isupper():
            return name
        # Fuzzy match
        for full, abbr in abbrevs.items():
            if name in full or full in name:
                return abbr
        return name[:3].upper() if name else None

    for game in games:
        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")

        # Handle nba_api nested format
        if "home" in game and isinstance(game["home"], dict):
            h = game["home"]
            a = game.get("away", {})
            home_score = h.get("pts")
            away_score = a.get("pts")
            if not home_raw:
                home_raw = h.get("team_name", "")
            if not away_raw:
                away_raw = a.get("team_name", "")
        else:
            home_score = game.get("home_score")
            away_score = game.get("away_score")

        if home_score is None or away_score is None:
            continue

        home_score = int(home_score)
        away_score = int(away_score)
        home = resolve(home_raw)
        away = resolve(away_raw)

        if not home or not away:
            continue

        game_date = game.get("game_date", game.get("date", game.get("commence_time", "")))[:10]

        # ── Compute features from rolling history (no leakage) ──
        hr = team_results[home]
        ar = team_results[away]

        # Win pct windows
        def win_pct(results, n):
            r = results[-n:] if results else []
            return (sum(1 for x in r if x[1]) / len(r)) if r else 0.5

        def pt_diff(results, n):
            r = results[-n:] if results else []
            return (sum(x[2] for x in r) / len(r)) if r else 0.0

        def streak(results):
            if not results:
                return 0
            s = 0
            last_won = results[-1][1]
            for r in reversed(results):
                if r[1] == last_won:
                    s += 1
                else:
                    break
            return s if last_won else -s

        def rest_days(team):
            last = team_last_game.get(team)
            if not last or not game_date:
                return 3  # default
            try:
                d1 = datetime.strptime(last[:10], "%Y-%m-%d")
                d2 = datetime.strptime(game_date[:10], "%Y-%m-%d")
                return max(0, (d2 - d1).days)
            except Exception:
                return 3

        # SOS: avg opponent win% of recent opponents
        def sos(results, n=10):
            recent = results[-n:]
            if not recent:
                return 0.5
            opp_wpcts = []
            for r in recent:
                opp = r[3]
                opp_r = team_results[opp]
                if opp_r:
                    opp_wpcts.append(sum(1 for x in opp_r if x[1]) / len(opp_r))
            return sum(opp_wpcts) / len(opp_wpcts) if opp_wpcts else 0.5

        # Season phase
        try:
            month = int(game_date[5:7]) if game_date else 1
            season_phase = max(0, min(1, (month - 10) / 7)) if month >= 10 else max(0, min(1, (month + 2) / 7))
        except Exception:
            season_phase = 0.5

        h_rest = rest_days(home)
        a_rest = rest_days(away)

        row = [
            # Win% multi-window (6)
            win_pct(hr, 5), win_pct(ar, 5),
            win_pct(hr, 10), win_pct(ar, 10),
            win_pct(hr, 20), win_pct(ar, 20),
            # Momentum: short vs long (2)
            win_pct(hr, 5) - win_pct(hr, 20),
            win_pct(ar, 5) - win_pct(ar, 20),
            # Point differential windows (4)
            pt_diff(hr, 5), pt_diff(ar, 5),
            pt_diff(hr, 10), pt_diff(ar, 10),
            # Rest & fatigue (4)
            min(h_rest, 7), min(a_rest, 7),
            1.0 if h_rest == 1 else 0.0,  # B2B
            1.0 if a_rest == 1 else 0.0,
            # Home court (1)
            1.0,
            # Streaks (2)
            streak(hr) / 10.0, streak(ar) / 10.0,
            # SOS (2)
            sos(hr), sos(ar),
            # Season phase (1)
            season_phase,
            # Games played (proxy for sample reliability) (2)
            min(len(hr), 82) / 82.0,
            min(len(ar), 82) / 82.0,
        ]

        X_rows.append(row)
        y_rows.append(1 if home_score > away_score else 0)

        # Update rolling stats AFTER computing features
        margin = home_score - away_score
        team_results[home].append((game_date, home_score > away_score, margin, away))
        team_results[away].append((game_date, away_score > home_score, -margin, home))
        team_last_game[home] = game_date
        team_last_game[away] = game_date

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    log(f"Built {X.shape[0]} samples x {X.shape[1]} features")
    return X, y


FEATURE_NAMES_IMPROVE = [
    "home_win5", "away_win5", "home_win10", "away_win10", "home_win20", "away_win20",
    "home_momentum", "away_momentum",
    "home_ptdiff5", "away_ptdiff5", "home_ptdiff10", "away_ptdiff10",
    "home_rest", "away_rest", "home_b2b", "away_b2b",
    "home_court",
    "home_streak", "away_streak",
    "home_sos", "away_sos",
    "season_phase",
    "home_games_played", "away_games_played",
]


# ── Step 4: Optuna Hyperparameter Search ──────────────────────────────────────

def optuna_xgb(X, y, n_trials=30) -> dict:
    """Optuna search for XGBoost hyperparameters."""
    if not HAS_OPTUNA:
        return {}

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": 1,
        }
        model = xgb.XGBClassifier(**params)

        brier_scores = []
        for train_idx, val_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            probs = model.predict_proba(X[val_idx])[:, 1]
            brier_scores.append(brier_score_loss(y[val_idx], probs))

        return np.mean(brier_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log(f"XGBoost Optuna: best Brier = {study.best_value:.4f} in {n_trials} trials")
    return study.best_params


def optuna_lgbm(X, y, n_trials=30) -> dict:
    """Optuna search for LightGBM hyperparameters."""
    if not HAS_OPTUNA:
        return {}

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1,
        }
        model = lgb.LGBMClassifier(**params)

        brier_scores = []
        for train_idx, val_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            probs = model.predict_proba(X[val_idx])[:, 1]
            brier_scores.append(brier_score_loss(y[val_idx], probs))

        return np.mean(brier_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log(f"LightGBM Optuna: best Brier = {study.best_value:.4f} in {n_trials} trials")
    return study.best_params


# ── Step 5: Train & Compare All Models ────────────────────────────────────────

def train_all_models(X, y, xgb_params=None, lgbm_params=None) -> Dict:
    """Train all models with TimeSeriesSplit CV. Return results sorted by Brier."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=10,
            random_state=42, n_jobs=1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=10,
            random_state=42, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
    }

    # XGBoost with optuna params or defaults
    xgb_p = xgb_params or {}
    models["xgboost"] = xgb.XGBClassifier(
        n_estimators=xgb_p.get("n_estimators", 500),
        max_depth=xgb_p.get("max_depth", 6),
        learning_rate=xgb_p.get("learning_rate", 0.05),
        subsample=xgb_p.get("subsample", 0.8),
        colsample_bytree=xgb_p.get("colsample_bytree", 0.8),
        min_child_weight=xgb_p.get("min_child_weight", 3),
        gamma=xgb_p.get("gamma", 0.1),
        reg_alpha=xgb_p.get("reg_alpha", 0.1),
        reg_lambda=xgb_p.get("reg_lambda", 1.0),
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )

    # LightGBM with optuna params or defaults
    lgbm_p = lgbm_params or {}
    models["lightgbm"] = lgb.LGBMClassifier(
        n_estimators=lgbm_p.get("n_estimators", 500),
        max_depth=lgbm_p.get("max_depth", 6),
        learning_rate=lgbm_p.get("learning_rate", 0.05),
        subsample=lgbm_p.get("subsample", 0.8),
        colsample_bytree=lgbm_p.get("colsample_bytree", 0.8),
        min_child_samples=lgbm_p.get("min_child_samples", 20),
        num_leaves=lgbm_p.get("num_leaves", 31),
        reg_alpha=lgbm_p.get("reg_alpha", 0.1),
        reg_lambda=lgbm_p.get("reg_lambda", 1.0),
        verbose=-1,
        random_state=42,
        n_jobs=1,
    )

    # CatBoost if available
    if HAS_CATBOOST:
        models["catboost"] = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            verbose=0, random_state=42, thread_count=1
        )

    for name, model in models.items():
        if _shutdown:
            break

        log(f"  Training {name}...")
        briers = []
        accs = []
        loglosses = []

        try:
            for train_idx, val_idx in tscv.split(X):
                X_tr = X_scaled[train_idx] if name in ("logistic_regression", "svm") else X[train_idx]
                X_va = X_scaled[val_idx] if name in ("logistic_regression", "svm") else X[val_idx]

                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_tr, y[train_idx])

                probs = model_copy.predict_proba(X_va)[:, 1]
                preds = (probs >= 0.5).astype(int)

                briers.append(brier_score_loss(y[val_idx], probs))
                accs.append(accuracy_score(y[val_idx], preds))
                loglosses.append(log_loss(y[val_idx], probs))

            mean_brier = np.mean(briers)
            std_brier = np.std(briers)
            mean_acc = np.mean(accs)

            results[name] = {
                "brier": round(float(mean_brier), 5),
                "brier_std": round(float(std_brier), 5),
                "accuracy": round(float(mean_acc), 4),
                "logloss": round(float(np.mean(loglosses)), 5),
                "model": model,
            }

            log(f"  {name:25s} Brier={mean_brier:.4f}±{std_brier:.4f} | Acc={mean_acc:.3f}")

        except Exception as e:
            log(f"  {name} failed: {e}", "ERROR")

    # ── Calibrated versions ──
    log("  Calibrating top models...")
    top_models = sorted(results.items(), key=lambda x: x[1]["brier"])[:4]

    for name, res in top_models:
        if _shutdown:
            break
        try:
            cal_name = f"{name}_calibrated"
            cal_model = CalibratedClassifierCV(
                res["model"], method="isotonic", cv=3
            )

            briers = []
            for train_idx, val_idx in tscv.split(X):
                X_tr = X_scaled[train_idx] if "logistic" in name else X[train_idx]
                X_va = X_scaled[val_idx] if "logistic" in name else X[val_idx]

                cal_copy = CalibratedClassifierCV(
                    type(res["model"])(**res["model"].get_params()),
                    method="isotonic", cv=3
                )
                cal_copy.fit(X_tr, y[train_idx])
                probs = cal_copy.predict_proba(X_va)[:, 1]
                briers.append(brier_score_loss(y[val_idx], probs))

            mean_brier = np.mean(briers)
            improvement = (res["brier"] - mean_brier) / res["brier"] * 100

            results[cal_name] = {
                "brier": round(float(mean_brier), 5),
                "brier_std": round(float(np.std(briers)), 5),
                "accuracy": res["accuracy"],
                "logloss": res["logloss"],
                "calibration_improvement": round(float(improvement), 1),
            }

            log(f"  {cal_name:25s} Brier={mean_brier:.4f} ({improvement:+.1f}% vs raw)")

        except Exception as e:
            log(f"  Calibration of {name} failed: {e}", "WARN")

    # ── Stacking Ensemble ──
    try:
        log("  Training stacking ensemble...")
        estimators = [
            ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=1)),
            ("xgb", xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                        eval_metric="logloss", random_state=42, n_jobs=1)),
        ]
        stacker = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=3, n_jobs=1, passthrough=False
        )

        briers = []
        for train_idx, val_idx in tscv.split(X):
            stacker_copy = StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=1)),
                    ("xgb", xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                eval_metric="logloss", random_state=42, n_jobs=1)),
                ],
                final_estimator=LogisticRegression(max_iter=1000, random_state=42),
                cv=3, n_jobs=1
            )
            stacker_copy.fit(X[train_idx], y[train_idx])
            probs = stacker_copy.predict_proba(X[val_idx])[:, 1]
            briers.append(brier_score_loss(y[val_idx], probs))

        mean_brier = np.mean(briers)
        results["stacking_ensemble"] = {
            "brier": round(float(mean_brier), 5),
            "brier_std": round(float(np.std(briers)), 5),
            "accuracy": 0,
            "logloss": 0,
        }
        log(f"  {'stacking_ensemble':25s} Brier={mean_brier:.4f}")

    except Exception as e:
        log(f"  Stacking failed: {e}", "WARN")

    return results


# ── Step 6: Persist Best Model ────────────────────────────────────────────────

def persist_results(results: Dict, X, y, games_count: int):
    """Save best model and log improvement."""
    if not results:
        return

    sorted_results = sorted(
        [(k, v) for k, v in results.items()],
        key=lambda x: x[1]["brier"]
    )

    best_name, best = sorted_results[0]
    log(f"\n{'='*60}")
    log(f"BEST MODEL: {best_name} — Brier {best['brier']:.4f} (Acc {best.get('accuracy', 0):.3f})")
    log(f"{'='*60}")

    # Load previous best
    prev_best = 1.0
    if BEST_BRIER_FILE.exists():
        try:
            prev = json.loads(BEST_BRIER_FILE.read_text())
            prev_best = prev.get("brier", 1.0)
        except Exception:
            pass

    improvement = (prev_best - best["brier"]) / prev_best * 100 if prev_best < 1.0 else 0

    # Save new best
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_model": best_name,
        "brier": best["brier"],
        "brier_std": best.get("brier_std", 0),
        "accuracy": best.get("accuracy", 0),
        "improvement_vs_prev": round(improvement, 2),
        "games_trained": games_count,
        "features": X.shape[1] if X is not None else 0,
        "all_models": {
            k: {"brier": v["brier"], "accuracy": v.get("accuracy", 0)}
            for k, v in sorted_results
        },
        "cycle": _cycle,
    }

    BEST_BRIER_FILE.write_text(json.dumps(record, indent=2))

    if improvement > 0:
        log(f"IMPROVEMENT: {improvement:+.2f}% vs previous best ({prev_best:.4f} → {best['brier']:.4f})")
    elif prev_best < 1.0:
        log(f"No improvement vs previous best ({prev_best:.4f})")

    # Sync to mon-ipad for website
    sync_file = Path("/home/lahargnedebartoli/mon-ipad/data/nba-agent/improve-results.json")
    try:
        sync_file.write_text(json.dumps(record, indent=2))
    except Exception:
        pass

    # Also update quant-summary with latest results
    summary_file = Path("/home/lahargnedebartoli/mon-ipad/data/nba-agent/quant-summary.json")
    try:
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        summary["best_brier"] = best["brier"]
        summary["val_accuracy"] = best.get("accuracy", 0)
        summary["best_model"] = best_name
        summary["games_trained"] = games_count
        summary["features"] = X.shape[1]
        summary["last_improve_cycle"] = datetime.now(timezone.utc).isoformat()
        summary["all_model_briers"] = {k: v["brier"] for k, v in sorted_results[:10]}
        summary_file.write_text(json.dumps(summary, indent=2))
    except Exception:
        pass

    return record


# ── Step 7: Feature Importance Analysis ───────────────────────────────────────

def analyze_features(X, y) -> List[Tuple[str, float]]:
    """Rank features by importance using Random Forest."""
    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=1)
    model.fit(X, y)

    importances = model.feature_importances_
    names = FEATURE_NAMES_IMPROVE if len(FEATURE_NAMES_IMPROVE) == X.shape[1] else [f"f{i}" for i in range(X.shape[1])]

    ranked = sorted(zip(names, importances), key=lambda x: -x[1])

    log("Feature importance ranking:")
    for name, imp in ranked[:10]:
        log(f"  {name:25s} {imp:.4f}")

    return ranked


# ── Main Loop ─────────────────────────────────────────────────────────────────

def run_cycle():
    """Run one improvement cycle."""
    global _cycle
    _cycle += 1
    cycle_start = time.time()

    log(f"\n{'='*60}")
    log(f"IMPROVE CYCLE #{_cycle} START")
    log(f"{'='*60}")

    # Step 1: Expand data (pull more seasons)
    if _cycle == 1 or _cycle % 10 == 0:
        new_games = expand_data()
        if new_games > 0:
            log(f"Added {new_games} new games")

    # Step 2: Load all data
    games = load_all_games()
    if len(games) < 100:
        log("Not enough games for meaningful training", "ERROR")
        return

    # Step 3: Build features
    X, y = build_features_from_games(games)
    if X is None or len(X) < 100:
        log("Feature building failed", "ERROR")
        return

    # Step 4: Feature analysis (every 5th cycle)
    if _cycle % 5 == 1:
        analyze_features(X, y)

    # Step 5: Optuna hyperparameter search (every 3rd cycle)
    xgb_params = None
    lgbm_params = None

    if HAS_OPTUNA and (_cycle == 1 or _cycle % 3 == 0):
        n_trials = 20 if _cycle == 1 else 15
        log(f"Running Optuna search ({n_trials} trials each)...")
        xgb_params = optuna_xgb(X, y, n_trials=n_trials)
        lgbm_params = optuna_lgbm(X, y, n_trials=n_trials)

    # Step 6: Train all models
    results = train_all_models(X, y, xgb_params, lgbm_params)

    # Step 7: Persist
    record = persist_results(results, X, y, len(games))

    elapsed = time.time() - cycle_start
    log(f"IMPROVE CYCLE #{_cycle} DONE ({elapsed:.0f}s)")

    return record


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NBA Quant Continuous Improvement")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--once", action="store_true", help="Run one cycle")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between cycles (default: 1h)")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials per search")
    args = parser.parse_args()

    # Save PID
    pid_file = Path("/home/lahargnedebartoli/mon-ipad/data/nba-improve.pid")
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    log(f"NBA Quant Continuous Improvement Engine v1.0")
    log(f"  Optuna: {'YES' if HAS_OPTUNA else 'NO'}")
    log(f"  CatBoost: {'YES' if HAS_CATBOOST else 'NO'}")

    if args.once or not args.daemon:
        run_cycle()
    else:
        log(f"Starting daemon — {args.interval}s cycles")
        while not _shutdown:
            try:
                run_cycle()
            except Exception as e:
                log(f"Cycle error: {e}", "ERROR")
            # Wait, checking for shutdown every 10s
            for _ in range(args.interval // 10):
                if _shutdown:
                    break
                time.sleep(10)

    log("Improvement engine stopped")


if __name__ == "__main__":
    main()
