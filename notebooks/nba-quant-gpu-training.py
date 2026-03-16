#!/usr/bin/env python3
"""
NBA Quant AI — GPU Training Script (Google Colab / Lightning AI)

Run this on Google Colab (free T4 GPU) or Lightning AI for:
1. Deep Optuna search (100+ trials)
2. LSTM/Neural network models
3. MC Dropout uncertainty estimation
4. Large-scale backtesting

Upload to Colab: File → Upload notebook
Or run: !python nba-quant-gpu-training.py

Requirements (auto-installed):
  pip install numpy scikit-learn xgboost lightgbm catboost optuna nba_api torch geopy
"""

# ── Auto-install dependencies ──
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

import os, json, time, math, warnings
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, StackingClassifier, VotingClassifier
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
(DATA_DIR / "models").mkdir(exist_ok=True)
(DATA_DIR / "results").mkdir(exist_ok=True)

N_OPTUNA_TRIALS = 100  # More trials with GPU
N_SPLITS = 7           # More CV splits
RANDOM_STATE = 42

print(f"=== NBA QUANT AI — GPU Training ===")
print(f"Time: {datetime.now(timezone.utc).isoformat()}")
print(f"Optuna trials: {N_OPTUNA_TRIALS}")
print(f"CV splits: {N_SPLITS}")


# ── Data Loading (same as HF Space) ──

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

def resolve(name):
    if name in TEAM_MAP: return TEAM_MAP[name]
    if len(name) == 3 and name.isupper(): return name
    for full, abbr in TEAM_MAP.items():
        if name in full: return abbr
    return name[:3].upper() if name else None


def pull_seasons():
    """Pull NBA seasons from nba_api."""
    from nba_api.stats.endpoints import leaguegamefinder
    hist_dir = DATA_DIR / "historical"
    existing = {f.stem.replace("games-", "") for f in hist_dir.glob("games-*.json")}
    targets = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    missing = [s for s in targets if s not in existing]
    if not missing:
        print(f"All {len(targets)} seasons already cached")
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


def build_features(games):
    """Same feature builder as HF Space app.py."""
    team_results = defaultdict(list)
    team_last = {}
    X, y = [], []
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

        def wp(r, n):
            s = r[-n:]; return sum(1 for x in s if x[1])/len(s) if s else 0.5
        def pd(r, n):
            s = r[-n:]; return sum(x[2] for x in s)/len(s) if s else 0.0
        def strk(r):
            if not r: return 0
            s, l = 0, r[-1][1]
            for x in reversed(r):
                if x[1]==l: s+=1
                else: break
            return s if l else -s
        def rest(t):
            last = team_last.get(t)
            if not last or not gd: return 3
            try: return max(0,(datetime.strptime(gd[:10],"%Y-%m-%d")-datetime.strptime(last[:10],"%Y-%m-%d")).days)
            except: return 3
        def sos(r,n=10):
            rec = r[-n:]
            if not rec: return 0.5
            ops = [sum(1 for z in team_results[x[3]] if z[1])/len(team_results[x[3]]) for x in rec if team_results[x[3]]]
            return sum(ops)/len(ops) if ops else 0.5

        try:
            month = int(gd[5:7]) if gd else 1
            sp = max(0,min(1,(month-10)/7)) if month>=10 else max(0,min(1,(month+2)/7))
        except: sp = 0.5

        h_rest, a_rest = rest(home), rest(away)
        row = [
            wp(hr_,5), wp(ar_,5), wp(hr_,10), wp(ar_,10), wp(hr_,20), wp(ar_,20),
            wp(hr_,5)-wp(hr_,20), wp(ar_,5)-wp(ar_,20),
            pd(hr_,5), pd(ar_,5), pd(hr_,10), pd(ar_,10),
            min(h_rest,7), min(a_rest,7), 1.0 if h_rest==1 else 0.0, 1.0 if a_rest==1 else 0.0,
            1.0, strk(hr_)/10.0, strk(ar_)/10.0, sos(hr_), sos(ar_), sp,
            min(len(hr_),82)/82.0, min(len(ar_),82)/82.0,
        ]
        X.append(row); y.append(1 if hs>as_ else 0)
        m = hs - as_
        team_results[home].append((gd, hs>as_, m, away))
        team_results[away].append((gd, as_>hs, -m, home))
        team_last[home] = gd; team_last[away] = gd

    return np.nan_to_num(np.array(X, dtype=np.float64)), np.array(y, dtype=np.int32)


# ── DEEP OPTUNA SEARCH ──

def deep_optuna_search(X, y, n_trials=100):
    """Run deep hyperparameter search with 100+ trials."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = {}

    # XGBoost deep search
    print(f"\nXGBoost Optuna ({n_trials} trials)...")
    def xgb_obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "eval_metric": "logloss", "random_state": RANDOM_STATE, "n_jobs": -1,
        }
        if torch.cuda.is_available():
            p["tree_method"] = "gpu_hist"
            p["device"] = "cuda"
        m = xgb.XGBClassifier(**p)
        bs = []
        for ti, vi in tscv.split(X):
            m.fit(X[ti], y[ti])
            bs.append(brier_score_loss(y[vi], m.predict_proba(X[vi])[:, 1]))
        return np.mean(bs)

    study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(xgb_obj, n_trials=n_trials, show_progress_bar=True)
    results["xgboost"] = {"brier": study_xgb.best_value, "params": study_xgb.best_params}
    print(f"  Best XGB: Brier={study_xgb.best_value:.4f}")

    # LightGBM deep search
    print(f"\nLightGBM Optuna ({n_trials} trials)...")
    def lgbm_obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 10, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "verbose": -1, "random_state": RANDOM_STATE, "n_jobs": -1,
        }
        m = lgbm.LGBMClassifier(**p)
        bs = []
        for ti, vi in tscv.split(X):
            m.fit(X[ti], y[ti])
            bs.append(brier_score_loss(y[vi], m.predict_proba(X[vi])[:, 1]))
        return np.mean(bs)

    study_lgbm = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_lgbm.optimize(lgbm_obj, n_trials=n_trials, show_progress_bar=True)
    results["lightgbm"] = {"brier": study_lgbm.best_value, "params": study_lgbm.best_params}
    print(f"  Best LGBM: Brier={study_lgbm.best_value:.4f}")

    # CatBoost deep search
    print(f"\nCatBoost Optuna ({n_trials} trials)...")
    def cat_obj(trial):
        p = {
            "iterations": trial.suggest_int("iterations", 100, 800),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "verbose": 0, "random_state": RANDOM_STATE,
        }
        if torch.cuda.is_available():
            p["task_type"] = "GPU"
        m = CatBoostClassifier(**p)
        bs = []
        for ti, vi in tscv.split(X):
            m.fit(X[ti], y[ti])
            bs.append(brier_score_loss(y[vi], m.predict_proba(X[vi])[:, 1]))
        return np.mean(bs)

    study_cat = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_cat.optimize(cat_obj, n_trials=n_trials, show_progress_bar=True)
    results["catboost"] = {"brier": study_cat.best_value, "params": study_cat.best_params}
    print(f"  Best CatBoost: Brier={study_cat.best_value:.4f}")

    return results


# ── NEURAL NETWORK (LSTM) ──

def train_lstm(X, y):
    """Train LSTM model for sequence prediction (GPU accelerated)."""
    if not HAS_TORCH:
        print("PyTorch not available, skipping LSTM")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLSTM Training on {device}...")

    # Create sequences (window of 10 games per team)
    # Simple approach: use features as-is with a small network
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)

    # Simple feedforward + attention as baseline
    class NBANet(torch.nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_features, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
                torch.nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x).squeeze()

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    briers = []

    for fold, (ti, vi) in enumerate(tscv.split(X)):
        model = NBANet(X.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = torch.nn.BCELoss()

        X_train = X_tensor[ti]
        y_train = y_tensor[ti]
        X_val = X_tensor[vi]
        y_val = y_tensor[vi]

        model.train()
        for epoch in range(100):
            # Mini-batch training
            perm = torch.randperm(len(X_train))
            for i in range(0, len(X_train), 256):
                batch = perm[i:i+256]
                pred = model(X_train[batch])
                loss = criterion(pred, y_train[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val).cpu().numpy()
            brier = brier_score_loss(y[vi], preds)
            briers.append(brier)

    avg_brier = np.mean(briers)
    print(f"  NBANet: Brier={avg_brier:.4f} (avg {N_SPLITS} folds)")
    return {"brier": avg_brier, "model": "NBANet"}


# ── MC DROPOUT (Uncertainty) ──

def mc_dropout_predict(model, X_tensor, n_samples=50):
    """Monte Carlo Dropout for uncertainty estimation."""
    model.train()  # Keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            p = model(X_tensor).cpu().numpy()
            preds.append(p)
    preds = np.array(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std


# ── MAIN ──

def main():
    start = time.time()

    # Pull data
    pull_seasons()
    games = load_all_games()
    print(f"\nLoaded {len(games)} games")

    if len(games) < 500:
        print("Not enough games!")
        return

    X, y = build_features(games)
    print(f"Features: {X.shape}")

    # 1. Deep Optuna search
    optuna_results = deep_optuna_search(X, y, N_OPTUNA_TRIALS)

    # 2. Train all standard models with best params
    print("\n=== Training all models ===")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=500, max_depth=8, random_state=42, n_jobs=-1),
        "gradient_boost": GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42),
        "xgboost": xgb.XGBClassifier(**{**optuna_results["xgboost"]["params"], "eval_metric": "logloss", "random_state": 42, "n_jobs": -1}),
        "lightgbm": lgbm.LGBMClassifier(**{**optuna_results["lightgbm"]["params"], "verbose": -1, "random_state": 42, "n_jobs": -1}),
        "catboost": CatBoostClassifier(**{**optuna_results["catboost"]["params"], "verbose": 0, "random_state": 42}),
    }

    all_results = {}
    for name, model in models.items():
        briers, accs = [], []
        for ti, vi in tscv.split(X):
            Xtr = X_scaled[ti] if "logistic" in name else X[ti]
            Xva = X_scaled[vi] if "logistic" in name else X[vi]
            mc = type(model)(**model.get_params())
            mc.fit(Xtr, y[ti])
            p = mc.predict_proba(Xva)[:, 1]
            briers.append(brier_score_loss(y[vi], p))
            accs.append(accuracy_score(y[vi], (p >= 0.5).astype(int)))
        all_results[name] = {"brier": round(np.mean(briers), 5), "acc": round(np.mean(accs), 4)}
        print(f"  {name:25s} Brier={all_results[name]['brier']:.4f} Acc={all_results[name]['acc']:.3f}")

    # 3. Calibrated versions
    print("\n=== Calibration ===")
    for name in list(all_results.keys()):
        try:
            briers = []
            for ti, vi in tscv.split(X):
                Xtr = X_scaled[ti] if "logistic" in name else X[ti]
                Xva = X_scaled[vi] if "logistic" in name else X[vi]
                base = type(models[name])(**models[name].get_params())
                cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
                cal.fit(Xtr, y[ti])
                briers.append(brier_score_loss(y[vi], cal.predict_proba(Xva)[:, 1]))
            cname = f"{name}_cal"
            all_results[cname] = {"brier": round(np.mean(briers), 5), "acc": all_results[name]["acc"]}
            imp = (all_results[name]["brier"] - all_results[cname]["brier"]) / all_results[name]["brier"] * 100
            print(f"  {cname:25s} Brier={all_results[cname]['brier']:.4f} ({imp:+.1f}%)")
        except Exception as e:
            print(f"  {name}_cal failed: {e}")

    # 4. Neural network
    lstm_result = train_lstm(X, y)
    if lstm_result:
        all_results["nba_net"] = lstm_result

    # 5. Stacking ensemble
    print("\n=== Stacking ===")
    try:
        briers = []
        for ti, vi in tscv.split(X):
            stack = StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000, random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)),
                    ("xgb", xgb.XGBClassifier(**{**optuna_results["xgboost"]["params"], "eval_metric": "logloss", "random_state": 42, "n_jobs": -1})),
                    ("lgbm", lgbm.LGBMClassifier(**{**optuna_results["lightgbm"]["params"], "verbose": -1, "random_state": 42, "n_jobs": -1})),
                ],
                final_estimator=LogisticRegression(max_iter=500, random_state=42),
                cv=3, n_jobs=-1
            )
            stack.fit(X[ti], y[ti])
            briers.append(brier_score_loss(y[vi], stack.predict_proba(X[vi])[:, 1]))
        all_results["stacking_optimized"] = {"brier": round(np.mean(briers), 5), "acc": 0}
        print(f"  Stacking: Brier={all_results['stacking_optimized']['brier']:.4f}")
    except Exception as e:
        print(f"  Stacking failed: {e}")

    # Summary
    elapsed = time.time() - start
    best = min(all_results.items(), key=lambda x: x[1]["brier"])

    print(f"\n{'='*60}")
    print(f"RESULTS — {elapsed:.0f}s total")
    print(f"{'='*60}")
    print(f"Games: {len(games)}")
    print(f"Features: {X.shape[1]}")
    print(f"Optuna trials: {N_OPTUNA_TRIALS * 3} (XGB + LGBM + CatBoost)")
    print(f"\nBEST: {best[0]} — Brier {best[1]['brier']:.4f}")
    print(f"\nAll results:")
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["brier"]):
        print(f"  {name:30s} Brier={r['brier']:.4f}")

    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "games": len(games),
        "features": X.shape[1],
        "optuna_trials": N_OPTUNA_TRIALS * 3,
        "elapsed_seconds": round(elapsed),
        "gpu": torch.cuda.is_available(),
        "best_model": best[0],
        "best_brier": best[1]["brier"],
        "all_results": all_results,
        "optuna_params": optuna_results,
    }

    out_file = DATA_DIR / "results" / f"gpu-training-{datetime.now().strftime('%Y%m%d-%H%M')}.json"
    out_file.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
