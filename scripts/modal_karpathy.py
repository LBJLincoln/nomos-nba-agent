#!/usr/bin/env python3
"""
NBA Quant AI — Karpathy Autoresearch Loop (Modal Serverless GPU)
================================================================
Modal.com: $30/mo free, per-second billing, T4→A100, no session walls.
Pattern: github.com/karpathy/autoresearch

Usage:
  modal run scripts/modal_karpathy.py          # Run one session
  modal run scripts/modal_karpathy.py::status   # Check status
  modal deploy scripts/modal_karpathy.py       # Deploy as cron (every 6h)

Target: Beat ATR 0.21837
"""

import modal
import os

# ── Modal app definition ──
app = modal.App("nba-karpathy-loop")

# Persistent volume for state across runs
vol = modal.Volume.from_name("nba-evolution-state", create_if_missing=True)

# Container image with all deps
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy", "scikit-learn", "xgboost", "lightgbm", "catboost",
        "psycopg2-binary", "requests"
    )
)

# GPU image with TabICL
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "numpy", "scikit-learn", "xgboost", "lightgbm", "catboost",
        "tabicl", "psycopg2-binary", "requests", "huggingface_hub"
    )
)

STATE_DIR = "/data"
REPO_URL = "https://huggingface.co/spaces/Nomos42/nba-quant"


@app.function(
    image=gpu_image,
    gpu="T4",  # or "A10G", "A100" for more power
    timeout=6 * 3600,  # 6 hours max
    volumes={STATE_DIR: vol},
    secrets=[modal.Secret.from_name("nomos42-secrets")],
)
def run_evolution(max_iterations: int = 200, budget_sec: int = 300):
    """Run Karpathy autoresearch loop on GPU."""
    import numpy as np
    import json, time, gc, random, math, subprocess, sys
    from pathlib import Path
    from datetime import datetime
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import TimeSeriesSplit

    # Clone feature engine from HF Space (not GitHub — private repos need different auth)
    repo_dir = Path("/tmp/nba-quant-space")
    if not repo_dir.exists():
        hf_token = os.environ.get("HF_TOKEN", "")
        subprocess.run(["git", "clone", "--depth", "1",
                        f"https://user:{hf_token}@huggingface.co/spaces/Nomos42/nba-quant",
                        str(repo_dir)], check=True)
    sys.path.insert(0, str(repo_dir))

    # Load or build features
    cache_file = Path(f"{STATE_DIR}/features_cache_v38.npz")
    if cache_file.exists():
        print("Loading cached features...")
        data = np.load(str(cache_file), allow_pickle=True)
        X, y, feature_names = data["X"], data["y"], list(data["feature_names"])
    else:
        print("Building features...")
        from features.engine import NBAFeatureEngine

        # Load games: try local JSON first, then Supabase
        games = []
        for data_dir in [repo_dir / "data" / "historical", repo_dir / "hf-space" / "data" / "historical"]:
            if data_dir.exists():
                for f in sorted(data_dir.glob("games-*.json")):
                    raw = json.loads(f.read_text())
                    if isinstance(raw, list): games.extend(raw)
                    elif isinstance(raw, dict) and "games" in raw: games.extend(raw["games"])
                if games:
                    print(f"Loaded {len(games)} games from {data_dir}")
                    break

        if not games:
            print("No local game data — loading from Supabase...")
            db_url = os.environ.get("DATABASE_URL", "")
            if db_url:
                import psycopg2
                conn = psycopg2.connect(db_url, connect_timeout=30, options="-c search_path=public")
                cur = conn.cursor()
                cur.execute("SELECT game_data FROM nba_games ORDER BY game_date LIMIT 15000")
                for row in cur.fetchall():
                    if row[0]: games.append(row[0] if isinstance(row[0], dict) else json.loads(row[0]))
                cur.close(); conn.close()
                print(f"Loaded {len(games)} games from Supabase")

        if not games:
            raise ValueError("No game data available (local or Supabase)!")

        games.sort(key=lambda g: g.get("game_date", g.get("date", "")))
        engine = NBAFeatureEngine()
        X_raw, y_raw, feature_names = engine.build(games)
        X = np.nan_to_num(np.array(X_raw, dtype=np.float32))
        y = np.array(y_raw, dtype=np.int32)
        if len(X.shape) == 2 and X.shape[1] > 0:
            var = np.var(X, axis=0)
            valid = var > 1e-10
            X, feature_names = X[:, valid], [f for f, v in zip(feature_names, valid) if v]
        np.savez_compressed(str(cache_file), X=X, y=np.array(y),
                           feature_names=np.array(feature_names))
        print(f"Cached: {X.shape}")

    if X.shape[0] > 6000:
        X, y = X[-6000:], y[-6000:]
    print(f"Ready: {X.shape}")

    # Import ML
    import xgboost as xgb
    import lightgbm as lgbm
    from catboost import CatBoostClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        has_gpu = False

    # Models
    def make_model(model_type, hp):
        if model_type == "tabicl":
            from tabicl import TabICLClassifier
            return TabICLClassifier()
        elif model_type == "xgboost":
            return xgb.XGBClassifier(max_depth=hp.get("depth", 6), learning_rate=hp.get("lr", 0.1),
                n_estimators=hp.get("n_est", 200), random_state=42, verbosity=0,
                tree_method="hist", device="cuda" if has_gpu else "cpu")
        elif model_type == "xgboost_brier":
            def brier_obj(y_true, y_pred):
                return 2 * (y_pred - y_true), np.full_like(y_pred, 2.0)
            return xgb.XGBClassifier(max_depth=hp.get("depth", 6), learning_rate=hp.get("lr", 0.1),
                n_estimators=hp.get("n_est", 200), random_state=42, objective=brier_obj,
                verbosity=0, tree_method="hist", device="cuda" if has_gpu else "cpu")
        elif model_type == "catboost":
            return CatBoostClassifier(depth=min(hp.get("depth", 6), 10),
                learning_rate=hp.get("lr", 0.1), iterations=hp.get("n_est", 200),
                random_state=42, verbose=0, task_type="GPU" if has_gpu else "CPU")
        elif model_type == "lightgbm":
            return lgbm.LGBMClassifier(max_depth=hp.get("depth", 6),
                learning_rate=hp.get("lr", 0.1), n_estimators=hp.get("n_est", 200),
                random_state=42, verbose=-1)
        elif model_type == "extra_trees":
            return ExtraTreesClassifier(n_estimators=hp.get("n_est", 200),
                max_depth=hp.get("depth", None), random_state=42, n_jobs=-1)
        else:
            return RandomForestClassifier(n_estimators=hp.get("n_est", 200), random_state=42)

    def evaluate(mask, model_type, hp, timeout=120):
        selected = np.where(mask)[0]
        if len(selected) < 5 or len(selected) > 200: return 1.0
        X_sub = X[:, selected]
        tscv = TimeSeriesSplit(n_splits=2)
        try:
            briers = []
            t0 = time.time()
            for tr, te in tscv.split(X_sub):
                if time.time() - t0 > timeout: return 1.0
                model = make_model(model_type, hp)
                model.fit(X_sub[tr], y[tr])
                probs = model.predict_proba(X_sub[te])[:, 1]
                briers.append(brier_score_loss(y[te], probs))
                del model
            gc.collect()
            return float(np.mean(briers))
        except:
            return 1.0

    # Config
    CONFIG = {
        "population_size": 30,
        "iteration_budget_sec": budget_sec,
        "mutation_rate": 0.09,
        "crossover_rate": 0.80,
        "target_features": 63,
        "model_types": ["tabicl", "xgboost", "xgboost_brier", "extra_trees", "catboost", "lightgbm"],
        "model_weights": [0.30, 0.15, 0.15, 0.15, 0.15, 0.10],
    }

    def random_individual():
        n = X.shape[1]
        mask = np.zeros(n, dtype=bool)
        sel = np.random.choice(n, size=min(CONFIG["target_features"], n), replace=False)
        mask[sel] = True
        mt = np.random.choice(CONFIG["model_types"], p=CONFIG["model_weights"])
        hp = {"depth": random.randint(4, 10), "lr": round(random.uniform(0.01, 0.3), 3),
              "n_est": random.randint(100, 500)}
        return {"mask": mask, "model_type": mt, "hp": hp, "brier": 1.0}

    def mutate(ind):
        new = {"mask": ind["mask"].copy(), "model_type": ind["model_type"],
               "hp": dict(ind["hp"]), "brier": 1.0}
        n_flip = max(1, int(CONFIG["mutation_rate"] * np.sum(new["mask"])))
        for _ in range(n_flip):
            idx = random.randint(0, len(new["mask"])-1)
            new["mask"][idx] = not new["mask"][idx]
        n_sel = np.sum(new["mask"])
        while n_sel > 200:
            on = np.where(new["mask"])[0]; new["mask"][np.random.choice(on)] = False; n_sel -= 1
        while n_sel < 10:
            off = np.where(~new["mask"])[0]
            if len(off) == 0: break
            new["mask"][np.random.choice(off)] = True; n_sel += 1
        if random.random() < 0.2:
            new["hp"]["depth"] = max(4, min(10, new["hp"]["depth"] + random.choice([-1,0,1])))
        if random.random() < 0.1:
            new["model_type"] = np.random.choice(CONFIG["model_types"], p=CONFIG["model_weights"])
        return new

    def crossover(p1, p2):
        child = {"mask": np.zeros_like(p1["mask"]), "brier": 1.0}
        for i in range(len(child["mask"])):
            child["mask"][i] = p1["mask"][i] if random.random() < 0.5 else p2["mask"][i]
        child["model_type"] = p1["model_type"] if random.random() < 0.5 else p2["model_type"]
        child["hp"] = dict(p1["hp"] if random.random() < 0.5 else p2["hp"])
        return child

    # Load or init state
    state_file = Path(f"{STATE_DIR}/karpathy_state.json")
    log_file = Path(f"{STATE_DIR}/experiment_log.jsonl")

    if state_file.exists():
        state = json.loads(state_file.read_text())
        population = [{"mask": np.array(ind["mask"], dtype=bool), "model_type": ind["model_type"],
                        "hp": ind["hp"], "brier": ind["brier"]} for ind in state["population"]]
        best_ever = state["best_ever"]
        iteration = state["iteration"]
        print(f"Resumed: iter={iteration}, best={best_ever:.5f}")
    else:
        # Fetch seeds from HF islands
        import urllib.request, ssl
        seeds = []
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        for url in ["https://nomos42-nba-quant.hf.space/api/best",
                     "https://nomos42-nba-quant-2.hf.space/api/best",
                     "https://nomos42-nba-evo-3.hf.space/api/best",
                     "https://nomos42-nba-evo-4.hf.space/api/best",
                     "https://nomos42-nba-evo-5.hf.space/api/best",
                     "https://nomos42-nba-evo-6.hf.space/api/best"]:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Nomos42/1.0"})
                with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
                    d = json.loads(r.read())
                    if d.get("brier", 1.0) < 0.99:
                        mask = np.zeros(X.shape[1], dtype=bool)
                        for idx in d.get("features", []):
                            if 0 <= idx < X.shape[1]: mask[idx] = True
                        seeds.append({"mask": mask, "model_type": d.get("model_type", "xgboost"),
                                      "hp": d.get("hp", {"depth":6,"lr":0.1,"n_est":200}),
                                      "brier": float(d.get("brier", 1.0))})
                        seeds.append({"mask": mask.copy(), "model_type": "tabicl",
                                      "hp": {"depth":6,"lr":0.1,"n_est":200}, "brier": 1.0})
            except: pass
        population = seeds[:CONFIG["population_size"]]
        while len(population) < CONFIG["population_size"]:
            population.append(random_individual())
        best_ever = min((ind["brier"] for ind in population if ind["brier"] < 1.0), default=1.0)
        iteration = 0

    print(f"\n{'='*70}")
    print(f"  NBA QUANT AI — KARPATHY LOOP (Modal GPU)")
    print(f"  Pop={CONFIG['population_size']} | Max iterations={max_iterations}")
    print(f"  ATR: 0.21837 | Current best: {best_ever:.5f}")
    print(f"{'='*70}\n")

    session_start = time.time()
    for _ in range(max_iterations):
        iteration += 1
        t0 = time.time()
        n_evals = 0

        for ind in population:
            if ind["brier"] >= 0.99:
                ind["brier"] = evaluate(ind["mask"], ind["model_type"], ind["hp"])
                n_evals += 1
                if time.time() - t0 > CONFIG["iteration_budget_sec"]: break

        population.sort(key=lambda x: x["brier"])
        improved = population[0]["brier"] < best_ever
        if improved: best_ever = population[0]["brier"]

        elite_size = max(2, CONFIG["population_size"] // 5)
        elite = population[:elite_size]
        offspring = []
        while len(offspring) < CONFIG["population_size"] - elite_size:
            if random.random() < CONFIG["crossover_rate"]:
                p1, p2 = random.sample(elite, 2)
                child = mutate(crossover(p1, p2))
            else:
                child = mutate(random.choice(elite))
            offspring.append(child)
        population = elite + offspring

        dur = time.time() - t0
        tag = "*** NEW BEST ***" if improved else ""
        elapsed = (time.time() - session_start) / 60
        print(f"Iter {iteration}: best={best_ever:.5f} ({population[0]['model_type']}, "
              f"{int(np.sum(population[0]['mask']))}f) | {n_evals} evals {dur:.0f}s | "
              f"{elapsed:.0f}min {tag}")

        with open(log_file, "a") as f:
            f.write(json.dumps({"iter": iteration, "best": best_ever, "improved": improved,
                               "ts": datetime.now().isoformat()}) + "\n")

        s = {"population": [{"mask": ind["mask"].tolist(), "model_type": ind["model_type"],
              "hp": ind["hp"], "brier": ind["brier"]} for ind in population],
             "best_ever": best_ever, "iteration": iteration}
        state_file.write_text(json.dumps(s))
        vol.commit()  # Persist to Modal volume

        gc.collect()

    print(f"\nDONE: {iteration} iterations, best={best_ever:.5f}")
    return {"best_brier": best_ever, "iterations": iteration}


@app.function(image=image, volumes={STATE_DIR: vol})
def status():
    """Check current evolution status."""
    import json
    from pathlib import Path
    state_file = Path(f"{STATE_DIR}/karpathy_state.json")
    if state_file.exists():
        state = json.loads(state_file.read_text())
        print(f"Iteration: {state.get('iteration', '?')}")
        print(f"Best Brier: {state.get('best_ever', '?')}")
        return state
    print("No state found. Run evolution first.")
    return None


@app.local_entrypoint()
def main(iterations: int = 200, budget: int = 300):
    """CLI entrypoint: modal run scripts/modal_karpathy.py -- --iterations 200"""
    result = run_evolution.remote(max_iterations=iterations, budget_sec=budget)
    print(f"\nResult: {result}")
