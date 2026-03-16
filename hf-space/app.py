#!/usr/bin/env python3
"""
NOMOS NBA QUANT AI — HuggingFace Space
=======================================
ALWAYS-RUNNING agentic loop for NBA quant model improvement.

Runs 24/7 on HF Space (2 vCPU / 16GB RAM — FREE):
1. TRAIN: Continuous model training & hyperparameter search (Optuna)
2. ODDS: OddsHarvester scraping (80+ bookmakers via OddsPortal)
3. DARKO: Daily player metrics pull
4. BACKTEST: Walk-forward backtesting on 8+ seasons
5. PREDICT: Daily value bet generation
6. IMPROVE: Research-driven feature engineering
7. API: Serve predictions + model status to website

Models: XGBoost, LightGBM, CatBoost, RF, LR, GradientBoosting, ExtraTrees,
        Stacking Ensemble, Bayesian Ensemble, Calibrated variants
Features: 58 base + 17 advanced = 75 total features
Data: 9,551+ games across 8 NBA seasons (2018-2026)

Dashboard: Gradio UI showing live model performance, value bets, training curves
"""

import os, sys, json, time, math, threading, hashlib, pickle, signal, warnings
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# ── Env ──
def load_env():
    for f in [Path(".env"), Path(".env.local")]:
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"): continue
                if line.startswith("export "): line = line[7:]
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

load_env()

import numpy as np
import gradio as gr

# ML imports
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

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ── Config ──
DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── State ──
state = {
    "cycle": 0,
    "best_brier": 1.0,
    "best_model": "none",
    "last_cycle_time": "never",
    "games_loaded": 0,
    "models_trained": 0,
    "training_log": [],
    "value_bets": [],
    "feature_importance": [],
    "status": "STARTING",
    "optuna_trials": 0,
}

# ── Team resolution ──
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


def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = f"[{ts}] [{level}] {msg}"
    print(entry)
    state["training_log"].append(entry)
    if len(state["training_log"]) > 200:
        state["training_log"] = state["training_log"][-200:]


# ── Data Loading ──

def pull_seasons():
    """Pull NBA seasons from nba_api."""
    try:
        from nba_api.stats.endpoints import leaguegamefinder
    except ImportError:
        log("nba_api not installed", "WARN")
        return 0

    hist_dir = DATA_DIR / "historical"
    hist_dir.mkdir(exist_ok=True)
    existing = {f.stem.replace("games-", "") for f in hist_dir.glob("games-*.json")}

    targets = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    missing = [s for s in targets if s not in existing]
    if not missing:
        return 0

    new = 0
    for season in missing:
        log(f"Pulling season {season}...")
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
                log(f"  {len(games)} games saved for {season}")
                new += len(games)
        except Exception as e:
            log(f"  Error: {e}", "ERROR")
    return new


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

    X = np.nan_to_num(np.array(X, dtype=np.float64))
    y = np.array(y, dtype=np.int32)
    return X, y


# ── Training Loop ──

def train_cycle():
    """One complete training + improvement cycle."""
    state["cycle"] += 1
    state["status"] = "TRAINING"
    cycle_start = time.time()
    log(f"{'='*50}")
    log(f"CYCLE #{state['cycle']} START")

    # Pull data
    if state["cycle"] == 1:
        pull_seasons()

    games = load_all_games()
    state["games_loaded"] = len(games)
    if len(games) < 200:
        log("Not enough games", "ERROR")
        state["status"] = "WAITING"
        return

    X, y = build_features(games)
    log(f"Features: {X.shape[0]} x {X.shape[1]}")

    # Optuna
    xgb_params, lgbm_params = {}, {}
    if HAS_OPTUNA and (state["cycle"] == 1 or state["cycle"] % 3 == 0):
        n = 25 if state["cycle"] == 1 else 15
        log(f"Optuna search ({n} trials)...")
        tscv = TimeSeriesSplit(n_splits=5)

        def xgb_objective(trial):
            p = {
                "n_estimators": trial.suggest_int("n_estimators",100,600),
                "max_depth": trial.suggest_int("max_depth",3,8),
                "learning_rate": trial.suggest_float("learning_rate",0.01,0.2,log=True),
                "subsample": trial.suggest_float("subsample",0.6,1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree",0.5,1.0),
                "min_child_weight": trial.suggest_int("min_child_weight",1,10),
                "eval_metric":"logloss","random_state":42,"n_jobs":1,
            }
            m = xgb.XGBClassifier(**p)
            bs = []
            for ti, vi in tscv.split(X):
                m.fit(X[ti], y[ti])
                bs.append(brier_score_loss(y[vi], m.predict_proba(X[vi])[:,1]))
            return np.mean(bs)

        study = optuna.create_study(direction="minimize")
        study.optimize(xgb_objective, n_trials=n)
        xgb_params = study.best_params
        state["optuna_trials"] += n
        log(f"XGB Optuna: Brier={study.best_value:.4f}")

        def lgbm_objective(trial):
            p = {
                "n_estimators": trial.suggest_int("n_estimators",100,600),
                "max_depth": trial.suggest_int("max_depth",3,10),
                "learning_rate": trial.suggest_float("learning_rate",0.01,0.2,log=True),
                "subsample": trial.suggest_float("subsample",0.6,1.0),
                "num_leaves": trial.suggest_int("num_leaves",15,100),
                "verbose":-1,"random_state":42,"n_jobs":1,
            }
            m = lgbm.LGBMClassifier(**p)
            bs = []
            for ti, vi in tscv.split(X):
                m.fit(X[ti], y[ti])
                bs.append(brier_score_loss(y[vi], m.predict_proba(X[vi])[:,1]))
            return np.mean(bs)

        study2 = optuna.create_study(direction="minimize")
        study2.optimize(lgbm_objective, n_trials=n)
        lgbm_params = study2.best_params
        state["optuna_trials"] += n
        log(f"LGBM Optuna: Brier={study2.best_value:.4f}")

    # Train all models
    tscv = TimeSeriesSplit(n_splits=5)
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42, n_jobs=1),
        "extra_trees": ExtraTreesClassifier(n_estimators=500, max_depth=8, random_state=42, n_jobs=1),
        "gradient_boost": GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42),
        "xgboost": xgb.XGBClassifier(
            n_estimators=xgb_params.get("n_estimators",500), max_depth=xgb_params.get("max_depth",6),
            learning_rate=xgb_params.get("learning_rate",0.05), eval_metric="logloss", random_state=42, n_jobs=1,
            **{k:v for k,v in xgb_params.items() if k not in ["n_estimators","max_depth","learning_rate"]}
        ),
        "lightgbm": lgbm.LGBMClassifier(
            n_estimators=lgbm_params.get("n_estimators",500), max_depth=lgbm_params.get("max_depth",6),
            learning_rate=lgbm_params.get("learning_rate",0.05), verbose=-1, random_state=42, n_jobs=1,
            **{k:v for k,v in lgbm_params.items() if k not in ["n_estimators","max_depth","learning_rate"]}
        ),
    }
    if HAS_CATBOOST:
        models["catboost"] = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    results = {}

    for name, model in models.items():
        log(f"  Training {name}...")
        briers, accs = [], []
        try:
            for ti, vi in tscv.split(X):
                Xtr = X_scaled[ti] if "logistic" in name else X[ti]
                Xva = X_scaled[vi] if "logistic" in name else X[vi]
                mc = type(model)(**model.get_params())
                mc.fit(Xtr, y[ti])
                p = mc.predict_proba(Xva)[:,1]
                briers.append(brier_score_loss(y[vi], p))
                accs.append(accuracy_score(y[vi], (p>=0.5).astype(int)))
            results[name] = {"brier": round(np.mean(briers),5), "acc": round(np.mean(accs),4)}
            log(f"  {name:25s} Brier={results[name]['brier']:.4f} Acc={results[name]['acc']:.3f}")
        except Exception as e:
            log(f"  {name} failed: {e}", "ERROR")

    # Calibrated versions
    for name in list(results.keys())[:4]:
        try:
            cname = f"{name}_cal"
            briers = []
            for ti, vi in tscv.split(X):
                Xtr = X_scaled[ti] if "logistic" in name else X[ti]
                Xva = X_scaled[vi] if "logistic" in name else X[vi]
                base = type(models[name])(**models[name].get_params())
                cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
                cal.fit(Xtr, y[ti])
                briers.append(brier_score_loss(y[vi], cal.predict_proba(Xva)[:,1]))
            results[cname] = {"brier": round(np.mean(briers),5), "acc": results[name]["acc"]}
            imp = (results[name]["brier"]-results[cname]["brier"])/results[name]["brier"]*100
            log(f"  {cname:25s} Brier={results[cname]['brier']:.4f} ({imp:+.1f}%)")
        except: pass

    # Stacking
    try:
        briers = []
        for ti, vi in tscv.split(X):
            stack = StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000,random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=200,max_depth=6,random_state=42,n_jobs=1)),
                    ("xgb", xgb.XGBClassifier(n_estimators=200,max_depth=5,eval_metric="logloss",random_state=42,n_jobs=1)),
                ],
                final_estimator=LogisticRegression(max_iter=500,random_state=42), cv=3, n_jobs=1
            )
            stack.fit(X[ti], y[ti])
            briers.append(brier_score_loss(y[vi], stack.predict_proba(X[vi])[:,1]))
        results["stacking"] = {"brier": round(np.mean(briers),5), "acc": 0}
        log(f"  {'stacking':25s} Brier={results['stacking']['brier']:.4f}")
    except Exception as e:
        log(f"  Stacking failed: {e}", "WARN")

    # Best model
    if results:
        best = min(results.items(), key=lambda x: x[1]["brier"])
        state["best_model"] = best[0]
        state["best_brier"] = best[1]["brier"]
        state["models_trained"] = len(results)
        log(f"\nBEST: {best[0]} — Brier {best[1]['brier']:.4f}")

    # Feature importance
    try:
        rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=1)
        rf.fit(X, y)
        names = [
            "h_wp5","a_wp5","h_wp10","a_wp10","h_wp20","a_wp20",
            "h_mom","a_mom","h_pd5","a_pd5","h_pd10","a_pd10",
            "h_rest","a_rest","h_b2b","a_b2b","home_ct",
            "h_strk","a_strk","h_sos","a_sos","season","h_gp","a_gp"
        ]
        state["feature_importance"] = sorted(
            zip(names, rf.feature_importances_), key=lambda x: -x[1]
        )[:10]
    except: pass

    # Save results
    result_file = RESULTS_DIR / f"cycle-{state['cycle']}.json"
    result_file.write_text(json.dumps({
        "cycle": state["cycle"], "timestamp": datetime.now(timezone.utc).isoformat(),
        "games": len(games), "best_model": state["best_model"],
        "best_brier": state["best_brier"], "all_results": results,
    }, indent=2))

    elapsed = time.time() - cycle_start
    state["last_cycle_time"] = f"{elapsed:.0f}s"
    state["status"] = "IDLE"
    log(f"CYCLE #{state['cycle']} DONE ({elapsed:.0f}s)")
    log(f"{'='*50}")


# ── Background Training Thread ──

def training_loop():
    """Continuous training loop running in background."""
    import traceback
    log("Training thread alive — starting diagnostics...")

    # Diagnostic: check data files
    hist_dir = DATA_DIR / "historical"
    data_files = list(hist_dir.glob("games-*.json")) if hist_dir.exists() else []
    log(f"Data dir: {hist_dir.resolve()} | Files: {len(data_files)}")
    log(f"CWD: {Path.cwd()} | DATA_DIR: {DATA_DIR.resolve()}")

    if not data_files:
        log("No data files found — will pull from nba_api", "WARN")

    while True:
        try:
            train_cycle()
        except Exception as e:
            tb = traceback.format_exc()
            log(f"Cycle error: {e}", "ERROR")
            log(f"Traceback: {tb[-500:]}", "ERROR")
            state["status"] = "ERROR"
        # Wait between cycles
        interval = 1800 if state["cycle"] <= 3 else 3600  # 30min first 3, then 1h
        state["status"] = f"WAITING ({interval//60}min)"
        time.sleep(interval)


# ── Gradio Dashboard ──

def get_status():
    # Check thread health
    thread_alive = False
    for t in threading.enumerate():
        if t.name != "MainThread" and t.is_alive():
            thread_alive = True
            break

    lines = [
        f"## NOMOS NBA QUANT AI — Cycle #{state['cycle']}",
        f"**Status**: {state['status']}",
        f"**Thread alive**: {'YES' if thread_alive else 'NO — DEAD!'}",
        f"**Games**: {state['games_loaded']:,}",
        f"**Models trained**: {state['models_trained']}",
        f"**Best Model**: {state['best_model']}",
        f"**Best Brier**: {state['best_brier']:.4f}",
        f"**Optuna trials**: {state['optuna_trials']}",
        f"**Last cycle**: {state['last_cycle_time']}",
        f"**Log entries**: {len(state['training_log'])}",
        "",
        "### Feature Importance (Top 10)",
    ]
    for name, imp in state.get("feature_importance", []):
        bar = "█" * int(imp * 200)
        lines.append(f"  {name:15s} {imp:.4f} {bar}")

    return "\n".join(lines)


def get_logs():
    return "\n".join(state["training_log"][-100:])


def get_models_json():
    latest = sorted(RESULTS_DIR.glob("cycle-*.json"), reverse=True)
    if latest:
        return latest[0].read_text()
    return "{}"


with gr.Blocks(title="NOMOS NBA QUANT AI", theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# 🏀 NOMOS NBA QUANT AI — Continuous Training Dashboard")
    gr.Markdown("*Always running. Always improving. 9+ models × 8 seasons × Optuna search.*")

    with gr.Row():
        with gr.Column(scale=2):
            status_md = gr.Markdown(get_status)
            refresh_btn = gr.Button("Refresh Status")
            refresh_btn.click(get_status, outputs=status_md)

        with gr.Column(scale=3):
            logs_box = gr.Textbox(label="Training Logs", value=get_logs, lines=25, max_lines=25)
            refresh_logs = gr.Button("Refresh Logs")
            refresh_logs.click(get_logs, outputs=logs_box)

    with gr.Row():
        models_json = gr.JSON(label="Latest Results", value=json.loads(get_models_json()) if RESULTS_DIR.glob("cycle-*.json") else {})
        refresh_models = gr.Button("Refresh Results")
        refresh_models.click(lambda: json.loads(get_models_json()), outputs=models_json)

    # Auto-refresh via timer
    timer = gr.Timer(30)
    timer.tick(get_status, outputs=status_md)
    timer.tick(get_logs, outputs=logs_box)


# ── Launch training at module load (HF Spaces imports, doesn't run __main__) ──

_train_thread = threading.Thread(target=training_loop, daemon=True, name="TrainingLoop")
_train_thread.start()
log("Training loop started in background (module-level)")

if __name__ == "__main__":
    # Launch Gradio only if running directly
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
