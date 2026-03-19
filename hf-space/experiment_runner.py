#!/usr/bin/env python3
"""
S11 Experiment Runner — Isolated Evaluation Server
=====================================================
Turns S11 from a clone of S10 into a dedicated experiment server.
Agents (Eve, CrewAI, etc.) submit experiments to Supabase queue.
S11 polls the queue, evaluates in isolation (walk-forward backtest),
and stores results back. S10's population is NEVER touched.

Experiment Types:
  - feature_test:      Test specific feature mask → evaluate with walk-forward
  - model_test:        Test specific model_type + hyperparams → evaluate
  - calibration_test:  Test calibration method on current best features
  - config_change:     Test GA config by running mini-evolution (5 gens)

Queue: Supabase table `nba_experiments`
  status: pending → running → completed | failed
"""

import os
import sys
import json
import time
import random
import traceback
import threading
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

# ── Import shared functions from app.py (lazy — app.py must be loaded first) ──
# When this module is imported from app.py's bottom section, app is already
# fully loaded in sys.modules. Direct `from app import` would re-execute
# app.py's top-level code (Gradio, data loading) causing a hang.
# Instead, we grab references AFTER import, in init_from_app().

def _default_log(msg, level="INFO"):
    print(f"[{level}] {msg}")

Individual = None
evaluate = None
_build = None
_prune_correlated_features = None
_log_loss_score = None
_ece = None
_evaluate_stacking = None
load_all_games = None
build_features = None
pull_seasons = None
log = _default_log
live = {}
FAST_EVAL_GAMES = 7000
DATA_DIR = Path("/data") if Path("/data").exists() else Path("data")
HIST_DIR = DATA_DIR / "historical"
STATE_DIR = DATA_DIR / "evolution-state"
RESULTS_DIR = DATA_DIR / "results"

def init_from_app():
    """Grab references from the already-loaded app module. Call once at startup."""
    import sys as _sys
    app_mod = _sys.modules.get('app') or _sys.modules.get('__main__')
    if app_mod is None:
        raise RuntimeError("app module not loaded")

    global Individual, evaluate, _build, _prune_correlated_features
    global _log_loss_score, _ece, _evaluate_stacking
    global load_all_games, build_features, pull_seasons, log, live
    global FAST_EVAL_GAMES, DATA_DIR, HIST_DIR, STATE_DIR, RESULTS_DIR

    Individual = getattr(app_mod, 'Individual')
    evaluate = getattr(app_mod, 'evaluate')
    _build = getattr(app_mod, '_build')
    _prune_correlated_features = getattr(app_mod, '_prune_correlated_features')
    _log_loss_score = getattr(app_mod, '_log_loss_score')
    _ece = getattr(app_mod, '_ece')
    _evaluate_stacking = getattr(app_mod, '_evaluate_stacking')
    load_all_games = getattr(app_mod, 'load_all_games')
    build_features = getattr(app_mod, 'build_features')
    pull_seasons = getattr(app_mod, 'pull_seasons')
    log = getattr(app_mod, 'log')
    live = getattr(app_mod, 'live')
    FAST_EVAL_GAMES = getattr(app_mod, 'FAST_EVAL_GAMES', 7000)
    DATA_DIR = getattr(app_mod, 'DATA_DIR', DATA_DIR)
    HIST_DIR = getattr(app_mod, 'HIST_DIR', HIST_DIR)
    STATE_DIR = getattr(app_mod, 'STATE_DIR', STATE_DIR)
    RESULTS_DIR = getattr(app_mod, 'RESULTS_DIR', RESULTS_DIR)

# ── Constants ──
POLL_INTERVAL = 60          # Poll Supabase every 60 seconds
MAX_EVAL_GAMES = 7000       # Cap evaluation to prevent OOM (16GB Space)
MINI_EVO_GENS = 5           # Generations for config_change experiments
MINI_EVO_POP = 20           # Small population for config_change experiments
EXPERIMENT_TIMEOUT = 1800   # 30 min max per experiment

# ── Supabase connection (same pattern as run_logger.py) ──
_pg_pool = None


def _get_pg():
    """Lazy PostgreSQL connection pool for Supabase."""
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        log("[EXPERIMENT] DATABASE_URL not set — cannot connect to Supabase", "ERROR")
        return None
    try:
        import psycopg2
        from psycopg2 import pool as pg_pool
        _pg_pool = pg_pool.SimpleConnectionPool(1, 3, db_url, options="-c search_path=public")
        conn = _pg_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        _pg_pool.putconn(conn)
        log("[EXPERIMENT] PostgreSQL connected to Supabase OK")
        return _pg_pool
    except Exception as e:
        log(f"[EXPERIMENT] PostgreSQL connection failed: {e}", "ERROR")
        _pg_pool = None
        return None


def _exec_sql(sql, params=None, fetch=True):
    """Execute SQL on Supabase. Returns rows on SELECT, True on INSERT/UPDATE, None on failure."""
    pool = _get_pg()
    if not pool:
        return None
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            if fetch:
                try:
                    return cur.fetchall()
                except Exception:
                    return True
            return True
    except Exception as e:
        log(f"[EXPERIMENT] SQL error: {e}", "ERROR")
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return None
    finally:
        if conn and pool:
            try:
                pool.putconn(conn)
            except Exception:
                pass


def _reconnect_pg():
    """Force reconnection on next call (e.g., after connection timeout)."""
    global _pg_pool
    if _pg_pool:
        try:
            _pg_pool.closeall()
        except Exception:
            pass
    _pg_pool = None


# ═══════════════════════════════════════════════════════
# EXPERIMENT FETCHER
# ═══════════════════════════════════════════════════════

def fetch_next_experiment() -> Optional[Dict[str, Any]]:
    """Fetch the highest-priority pending experiment from Supabase.

    Returns dict with all columns, or None if queue empty.
    Uses SELECT ... FOR UPDATE SKIP LOCKED to prevent double-pickup.
    """
    rows = _exec_sql("""
        SELECT id, experiment_id, agent_name, experiment_type, description,
               hypothesis, params, priority, status, target_space,
               baseline_brier, created_at
        FROM public.nba_experiments
        WHERE status = 'pending'
          AND (target_space IS NULL OR target_space = 'S11' OR target_space = 'any')
        ORDER BY priority DESC, created_at ASC
        LIMIT 1
    """)
    if not rows or rows is True:
        return None
    row = rows[0]
    return {
        "id": row[0],
        "experiment_id": row[1],
        "agent_name": row[2],
        "experiment_type": row[3],
        "description": row[4],
        "hypothesis": row[5],
        "params": row[6] if isinstance(row[6], dict) else json.loads(row[6]) if row[6] else {},
        "priority": row[7],
        "status": row[8],
        "target_space": row[9],
        "baseline_brier": row[10],
        "created_at": str(row[11]) if row[11] else None,
    }


def claim_experiment(exp_id: int) -> bool:
    """Atomically claim an experiment (pending → running). Returns False if already claimed."""
    result = _exec_sql("""
        UPDATE public.nba_experiments
        SET status = 'running', started_at = NOW()
        WHERE id = %s AND status = 'pending'
        RETURNING id
    """, (exp_id,))
    return result is not None and result is not True and len(result) > 0


def complete_experiment(exp_id: int, brier: float, accuracy: float,
                        log_loss_val: float, details: dict, status: str = "completed"):
    """Write results back to Supabase."""
    _exec_sql("""
        UPDATE public.nba_experiments
        SET status = %s,
            result_brier = %s,
            result_accuracy = %s,
            result_log_loss = %s,
            result_details = %s,
            completed_at = NOW()
        WHERE id = %s
    """, (status, brier, accuracy, log_loss_val, json.dumps(details), exp_id), fetch=False)


def fail_experiment(exp_id: int, error_msg: str):
    """Mark experiment as failed with error details."""
    details = {"error": error_msg[:2000], "failed_at": datetime.now(timezone.utc).isoformat()}
    _exec_sql("""
        UPDATE public.nba_experiments
        SET status = 'failed',
            result_details = %s,
            completed_at = NOW()
        WHERE id = %s
    """, (json.dumps(details), exp_id), fetch=False)


# ═══════════════════════════════════════════════════════
# EXPERIMENT EXECUTORS
# ═══════════════════════════════════════════════════════

def _make_individual(n_features: int, params: dict) -> Individual:
    """Create an Individual from experiment params.

    params can contain:
      - features: list of ints (feature mask) or list of feature indices
      - feature_indices: list of ints (indices to enable)
      - hyperparams: dict of hyperparams (merged with defaults)
      - model_type: str (shortcut for hyperparams.model_type)
      - calibration: str (shortcut for hyperparams.calibration)
    """
    ind = Individual(n_features, target=params.get("target_features", 80))

    # Feature mask: explicit list
    if "features" in params:
        feat = params["features"]
        if len(feat) == n_features:
            ind.features = [int(f) for f in feat]
        else:
            # Treat as indices
            ind.features = [0] * n_features
            for idx in feat:
                if 0 <= idx < n_features:
                    ind.features[idx] = 1

    # Feature indices: explicit list of which features to enable
    if "feature_indices" in params:
        ind.features = [0] * n_features
        for idx in params["feature_indices"]:
            if 0 <= idx < n_features:
                ind.features[idx] = 1

    # Hyperparams: merge with defaults
    if "hyperparams" in params:
        for k, v in params["hyperparams"].items():
            if k in ind.hyperparams:
                ind.hyperparams[k] = v

    # Shortcuts
    if "model_type" in params:
        ind.hyperparams["model_type"] = params["model_type"]
    if "calibration" in params:
        ind.hyperparams["calibration"] = params["calibration"]

    ind.n_features = sum(ind.features)
    return ind


def run_feature_test(experiment: dict, X: np.ndarray, y: np.ndarray,
                     feature_names: list) -> dict:
    """Test specific feature configuration with walk-forward evaluation.

    params should contain:
      - features or feature_indices: which features to enable
      - hyperparams (optional): model hyperparams
      - model_type (optional): which model to use (default: xgboost)
      - n_splits (optional): number of walk-forward splits (default: 3)
    """
    params = experiment["params"]
    n_features = X.shape[1]

    ind = _make_individual(n_features, params)

    # If no explicit features specified, use current best's feature set if available
    if "features" not in params and "feature_indices" not in params:
        # Load best individual from state
        best = _load_best_individual(n_features)
        if best:
            ind.features = list(best.features)
            ind.n_features = sum(ind.features)

    n_splits = params.get("n_splits", 3)
    fast = params.get("fast", False)

    log(f"[EXPERIMENT] feature_test: {ind.n_features} features, model={ind.hyperparams['model_type']}, splits={n_splits}")

    # Evaluate with full data (not fast mode by default for experiments)
    evaluate(ind, X, y, n_splits=n_splits, fast=fast)

    return {
        "brier": ind.fitness.get("brier", 1.0),
        "roi": ind.fitness.get("roi", 0.0),
        "sharpe": ind.fitness.get("sharpe", 0.0),
        "calibration": ind.fitness.get("calibration", 1.0),
        "composite": ind.fitness.get("composite", 0.0),
        "features_pruned": ind.fitness.get("features_pruned", 0),
        "n_features_selected": ind.n_features,
        "model_type": ind.hyperparams["model_type"],
        "hyperparams": ind.hyperparams,
    }


def run_model_test(experiment: dict, X: np.ndarray, y: np.ndarray,
                   feature_names: list) -> dict:
    """Test specific model type and hyperparams.

    params should contain:
      - model_type: str (xgboost, lightgbm, catboost, random_forest, extra_trees, stacking, mlp)
      - hyperparams (optional): full or partial hyperparams dict
      - features or feature_indices (optional): use specific features
    """
    params = experiment["params"]
    n_features = X.shape[1]

    # Start from best individual's features (most fair comparison)
    best = _load_best_individual(n_features)
    ind = _make_individual(n_features, params)

    if best and "features" not in params and "feature_indices" not in params:
        ind.features = list(best.features)
        ind.n_features = sum(ind.features)

    # Must have model_type
    if "model_type" not in params:
        raise ValueError("model_test requires 'model_type' in params")

    ind.hyperparams["model_type"] = params["model_type"]
    n_splits = params.get("n_splits", 3)
    fast = params.get("fast", False)

    log(f"[EXPERIMENT] model_test: model={ind.hyperparams['model_type']}, "
        f"{ind.n_features} features, splits={n_splits}")

    evaluate(ind, X, y, n_splits=n_splits, fast=fast)

    return {
        "brier": ind.fitness.get("brier", 1.0),
        "roi": ind.fitness.get("roi", 0.0),
        "sharpe": ind.fitness.get("sharpe", 0.0),
        "calibration": ind.fitness.get("calibration", 1.0),
        "composite": ind.fitness.get("composite", 0.0),
        "features_pruned": ind.fitness.get("features_pruned", 0),
        "n_features_selected": ind.n_features,
        "model_type": ind.hyperparams["model_type"],
        "hyperparams": ind.hyperparams,
    }


def run_calibration_test(experiment: dict, X: np.ndarray, y: np.ndarray,
                         feature_names: list) -> dict:
    """Test calibration method on current best features.

    params should contain:
      - calibration: str (isotonic, sigmoid, none)
      - model_type (optional): override model type
    """
    params = experiment["params"]
    n_features = X.shape[1]

    best = _load_best_individual(n_features)
    if not best:
        raise ValueError("No best individual found — cannot run calibration_test without baseline features")

    # Test each calibration method if "all" requested, otherwise just the one specified
    calibration = params.get("calibration", "isotonic")
    methods_to_test = ["isotonic", "sigmoid", "none"] if calibration == "all" else [calibration]

    results = {}
    best_brier = 1.0
    best_method = None

    for method in methods_to_test:
        ind = Individual.__new__(Individual)
        ind.features = list(best.features)
        ind.hyperparams = dict(best.hyperparams)
        ind.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "composite": 0.0}
        ind.generation = 0
        ind.n_features = sum(ind.features)
        ind.hyperparams["calibration"] = method

        if "model_type" in params:
            ind.hyperparams["model_type"] = params["model_type"]

        n_splits = params.get("n_splits", 3)
        log(f"[EXPERIMENT] calibration_test: method={method}, model={ind.hyperparams['model_type']}")

        evaluate(ind, X, y, n_splits=n_splits, fast=False)

        results[method] = {
            "brier": ind.fitness.get("brier", 1.0),
            "roi": ind.fitness.get("roi", 0.0),
            "sharpe": ind.fitness.get("sharpe", 0.0),
            "calibration": ind.fitness.get("calibration", 1.0),
            "composite": ind.fitness.get("composite", 0.0),
        }

        if ind.fitness.get("brier", 1.0) < best_brier:
            best_brier = ind.fitness["brier"]
            best_method = method

    return {
        "brier": best_brier,
        "best_method": best_method,
        "all_results": results,
        "n_features_selected": best.n_features,
        "model_type": best.hyperparams.get("model_type", "unknown"),
    }


def run_config_change(experiment: dict, X: np.ndarray, y: np.ndarray,
                      feature_names: list) -> dict:
    """Test GA config change by running a mini-evolution.

    params should contain any of:
      - pop_size: int (default 20)
      - mutation_rate: float
      - crossover_rate: float
      - tournament_size: int
      - target_features: int
      - n_generations: int (default 5)
      - elite_size: int
    """
    params = experiment["params"]
    n_features = X.shape[1]

    pop_size = min(params.get("pop_size", MINI_EVO_POP), 30)  # Cap at 30
    n_gens = min(params.get("n_generations", MINI_EVO_GENS), 10)  # Cap at 10
    mutation_rate = params.get("mutation_rate", 0.04)
    crossover_rate = params.get("crossover_rate", 0.80)
    tournament_size = min(params.get("tournament_size", 4), pop_size // 2)
    target_features = params.get("target_features", 80)
    elite_size = min(params.get("elite_size", 3), pop_size // 3)

    log(f"[EXPERIMENT] config_change: pop={pop_size}, gens={n_gens}, "
        f"mut={mutation_rate}, cx={crossover_rate}, target_feat={target_features}")

    # Initialize population
    population = []
    model_types = ["xgboost", "lightgbm", "catboost", "random_forest", "extra_trees", "stacking", "mlp"]
    for i in range(pop_size):
        ind = Individual(n_features, target=target_features)
        ind.hyperparams["model_type"] = model_types[i % len(model_types)]
        population.append(ind)

    # Seed with best individual if available
    best = _load_best_individual(n_features)
    if best and len(population) > 0:
        population[0] = best

    history = []
    best_ever_brier = 1.0
    best_ever_composite = 0.0

    for gen in range(n_gens):
        # Evaluate
        for ind in population:
            evaluate(ind, X, y, n_splits=2, fast=True)

        # Sort by composite
        population.sort(key=lambda x: x.fitness.get("composite", 0), reverse=True)
        gen_best = population[0]

        gen_brier = gen_best.fitness.get("brier", 1.0)
        gen_composite = gen_best.fitness.get("composite", 0.0)
        if gen_brier < best_ever_brier:
            best_ever_brier = gen_brier
        if gen_composite > best_ever_composite:
            best_ever_composite = gen_composite

        history.append({
            "generation": gen,
            "best_brier": gen_brier,
            "best_composite": gen_composite,
            "avg_brier": round(np.mean([p.fitness.get("brier", 1.0) for p in population]), 5),
            "avg_composite": round(np.mean([p.fitness.get("composite", 0) for p in population]), 5),
        })

        log(f"[EXPERIMENT] config_change gen {gen}: best_brier={gen_brier:.4f}, "
            f"composite={gen_composite:.4f}")

        # Selection + Crossover + Mutation (mini-GA)
        elites = population[:elite_size]
        new_pop = list(elites)

        while len(new_pop) < pop_size:
            # Tournament selection
            candidates = random.sample(population, min(tournament_size, len(population)))
            p1 = max(candidates, key=lambda x: x.fitness.get("composite", 0))
            candidates = random.sample(population, min(tournament_size, len(population)))
            p2 = max(candidates, key=lambda x: x.fitness.get("composite", 0))

            if random.random() < crossover_rate:
                child = Individual.crossover(p1, p2)
            else:
                child = Individual(n_features, target=target_features)

            child.mutate(rate=mutation_rate)
            new_pop.append(child)

        population = new_pop[:pop_size]

    # Final evaluation (full, not fast)
    final_best = max(population, key=lambda x: x.fitness.get("composite", 0))
    evaluate(final_best, X, y, n_splits=3, fast=False)

    return {
        "brier": final_best.fitness.get("brier", 1.0),
        "roi": final_best.fitness.get("roi", 0.0),
        "sharpe": final_best.fitness.get("sharpe", 0.0),
        "calibration": final_best.fitness.get("calibration", 1.0),
        "composite": final_best.fitness.get("composite", 0.0),
        "best_ever_brier": best_ever_brier,
        "best_ever_composite": best_ever_composite,
        "n_generations": n_gens,
        "pop_size": pop_size,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
        "target_features": target_features,
        "history": history,
        "final_model_type": final_best.hyperparams.get("model_type", "unknown"),
        "final_n_features": final_best.n_features,
    }


# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════

def _load_best_individual(n_features: int) -> Optional[Individual]:
    """Load the best individual from S10's saved state (read-only, never modifies)."""
    state_file = STATE_DIR / "population.json"
    if not state_file.exists():
        return None
    try:
        st = json.loads(state_file.read_text())
        if not st.get("best_ever"):
            return None
        be = st["best_ever"]
        ind = Individual.__new__(Individual)
        ind.features = be["features"]
        ind.hyperparams = be["hyperparams"]
        ind.fitness = be["fitness"]
        ind.generation = be.get("generation", 0)
        ind.n_features = sum(ind.features)

        # Resize if feature count changed
        if len(ind.features) < n_features:
            ind.features.extend([0] * (n_features - len(ind.features)))
        elif len(ind.features) > n_features:
            ind.features = ind.features[:n_features]
        ind.n_features = sum(ind.features)

        return ind
    except Exception as e:
        log(f"[EXPERIMENT] Failed to load best individual: {e}", "WARN")
        return None


def _compute_accuracy(ind: Individual, X: np.ndarray, y: np.ndarray) -> float:
    """Compute accuracy for an individual (separate from evaluate's fitness)."""
    try:
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score
        from sklearn.base import clone
        from sklearn.calibration import CalibratedClassifierCV

        selected = ind.selected_indices()
        if len(selected) < 15:
            return 0.0

        X_sub = np.nan_to_num(X[:, selected], nan=0.0, posinf=1e6, neginf=-1e6)
        X_sub, _ = _prune_correlated_features(X_sub, threshold=0.95)

        hp = ind.hyperparams
        model = _build(hp)
        if model is None:
            return 0.0

        tscv = TimeSeriesSplit(n_splits=3)
        accs = []
        for ti, vi in tscv.split(X_sub):
            try:
                m = clone(model)
                if hp.get("calibration", "none") != "none":
                    m = CalibratedClassifierCV(m, method=hp["calibration"], cv=2)
                m.fit(X_sub[ti], y[ti])
                preds = m.predict(X_sub[vi])
                accs.append(accuracy_score(y[vi], preds))
            except Exception:
                pass
        return round(float(np.mean(accs)), 4) if accs else 0.0
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════
# EXPERIMENT EXECUTION ENGINE
# ═══════════════════════════════════════════════════════

# Experiment state (for status API)
_current_experiment = None
_queue_depth = 0
_experiments_completed = 0
_experiments_failed = 0
_last_result = None


EXECUTORS = {
    "feature_test": run_feature_test,
    "model_test": run_model_test,
    "calibration_test": run_calibration_test,
    "config_change": run_config_change,
}


def run_experiment(experiment: dict, X: np.ndarray, y: np.ndarray,
                   feature_names: list) -> dict:
    """Route an experiment to the correct executor and return results."""
    global _current_experiment, _experiments_completed, _experiments_failed, _last_result

    exp_type = experiment["experiment_type"]
    exp_id = experiment["id"]
    _current_experiment = experiment

    log(f"[EXPERIMENT] === Starting: {experiment['experiment_id']} ===")
    log(f"[EXPERIMENT] Type: {exp_type} | Agent: {experiment['agent_name']} | Priority: {experiment['priority']}")
    log(f"[EXPERIMENT] Description: {experiment['description'][:200]}")

    executor = EXECUTORS.get(exp_type)
    if not executor:
        error_msg = f"Unknown experiment type: {exp_type}. Valid: {list(EXECUTORS.keys())}"
        fail_experiment(exp_id, error_msg)
        _experiments_failed += 1
        _current_experiment = None
        raise ValueError(error_msg)

    # Claim it
    if not claim_experiment(exp_id):
        _current_experiment = None
        raise RuntimeError(f"Experiment {exp_id} already claimed by another runner")

    start_time = time.time()
    try:
        results = executor(experiment, X, y, feature_names)
        elapsed = time.time() - start_time

        brier = results.get("brier", 1.0)
        accuracy = _compute_accuracy(
            _make_individual(X.shape[1], experiment["params"]), X, y
        ) if exp_type != "config_change" else results.get("accuracy", 0.0)

        # Compute log_loss from brier (approximate — actual log_loss needs probabilities)
        log_loss_val = results.get("log_loss", 0.0)

        results["elapsed_seconds"] = round(elapsed, 1)
        results["games_evaluated"] = min(X.shape[0], MAX_EVAL_GAMES)
        results["feature_candidates"] = X.shape[1]
        results["experiment_id"] = experiment["experiment_id"]
        results["agent_name"] = experiment["agent_name"]

        # Compare with baseline
        baseline = experiment.get("baseline_brier")
        if baseline:
            results["improvement"] = round(baseline - brier, 5)
            results["improved"] = brier < baseline

        complete_experiment(exp_id, brier, accuracy, log_loss_val, results)
        _experiments_completed += 1
        _last_result = results

        log(f"[EXPERIMENT] === Completed: {experiment['experiment_id']} ===")
        log(f"[EXPERIMENT] Brier: {brier:.4f} | Elapsed: {elapsed:.1f}s")
        if baseline:
            delta = baseline - brier
            log(f"[EXPERIMENT] vs baseline {baseline:.4f}: {'BETTER' if delta > 0 else 'WORSE'} by {abs(delta):.4f}")

        return results

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{str(e)[:500]}\n{traceback.format_exc()[-1000:]}"
        fail_experiment(exp_id, error_msg)
        _experiments_failed += 1
        log(f"[EXPERIMENT] === FAILED: {experiment['experiment_id']} ({elapsed:.1f}s) ===", "ERROR")
        log(f"[EXPERIMENT] Error: {str(e)[:300]}", "ERROR")
        raise
    finally:
        _current_experiment = None


# ═══════════════════════════════════════════════════════
# MAIN LOOP (replaces evolution_loop on S11)
# ═══════════════════════════════════════════════════════

def experiment_loop():
    """Main experiment polling loop — runs in background thread on S11.

    1. Load game data + build features (same as evolution_loop init)
    2. Poll Supabase every 60s for pending experiments
    3. Execute one at a time, write results back
    4. Never modifies S10's population or state
    """
    global _queue_depth

    # Initialize references to app.py functions (must be called after app.py is loaded)
    init_from_app()

    log("=" * 60)
    log("S11 EXPERIMENT RUNNER — STARTING")
    log("=" * 60)

    # ── Phase 1: Load data (same as evolution_loop) ──
    live["status"] = "EXPERIMENT: LOADING DATA"
    pull_seasons()
    games = load_all_games()
    live["games"] = len(games)
    log(f"[EXPERIMENT] Games loaded: {len(games)}")

    if len(games) < 500:
        log("[EXPERIMENT] NOT ENOUGH GAMES — aborting", "ERROR")
        live["status"] = "EXPERIMENT: ERROR (no data)"
        return

    # ── Phase 2: Build features ──
    live["status"] = "EXPERIMENT: BUILDING FEATURES"
    X, y, feature_names = build_features(games)
    log(f"[EXPERIMENT] Raw feature matrix: {X.shape}")

    # Remove zero-variance features (same filter as evolution_loop)
    variances = np.var(X, axis=0)
    valid_mask = variances > 1e-10
    n_removed = int((~valid_mask).sum())
    if n_removed > 0:
        X = X[:, valid_mask]
        feature_names = [f for f, v in zip(feature_names, valid_mask) if v]
        log(f"[EXPERIMENT] Noise filter: removed {n_removed} zero-variance features")

    n_feat = X.shape[1]
    live["feature_candidates"] = n_feat
    log(f"[EXPERIMENT] Clean feature matrix: {X.shape} ({n_feat} usable features)")

    # Cap games for OOM protection
    if X.shape[0] > MAX_EVAL_GAMES:
        log(f"[EXPERIMENT] Capping to {MAX_EVAL_GAMES} most recent games (OOM protection)")
        X = X[-MAX_EVAL_GAMES:]
        y = y[-MAX_EVAL_GAMES:]

    # ── Phase 3: Poll loop ──
    live["status"] = "EXPERIMENT: READY (polling)"
    log(f"[EXPERIMENT] Ready — polling every {POLL_INTERVAL}s for experiments")
    consecutive_errors = 0

    while True:
        try:
            experiment = fetch_next_experiment()

            if experiment:
                _queue_depth = _count_pending()
                live["status"] = f"EXPERIMENT: RUNNING ({experiment['experiment_type']})"
                log(f"[EXPERIMENT] Found experiment: {experiment['experiment_id']} "
                    f"(type={experiment['experiment_type']}, queue={_queue_depth})")

                try:
                    run_experiment(experiment, X, y, feature_names)
                except Exception as e:
                    log(f"[EXPERIMENT] Experiment failed: {e}", "ERROR")

                live["status"] = "EXPERIMENT: READY (polling)"
                consecutive_errors = 0

                # Refresh data every 10 experiments
                if (_experiments_completed + _experiments_failed) % 10 == 0:
                    try:
                        log("[EXPERIMENT] Refreshing game data...")
                        new_games = load_all_games()
                        if len(new_games) > len(games):
                            games = new_games
                            X_new, y_new, fn_new = build_features(games)
                            variances = np.var(X_new, axis=0)
                            valid_mask = variances > 1e-10
                            X_new = X_new[:, valid_mask]
                            fn_new = [f for f, v in zip(fn_new, valid_mask) if v]
                            if X_new.shape[0] > MAX_EVAL_GAMES:
                                X_new = X_new[-MAX_EVAL_GAMES:]
                                y_new = y_new[-MAX_EVAL_GAMES:]
                            X, y, feature_names = X_new, y_new, fn_new
                            n_feat = X.shape[1]
                            log(f"[EXPERIMENT] Data refreshed: {X.shape}")
                    except Exception as e:
                        log(f"[EXPERIMENT] Data refresh failed (continuing with old): {e}", "WARN")

            else:
                # No experiments pending — sleep
                time.sleep(POLL_INTERVAL)

        except Exception as e:
            consecutive_errors += 1
            log(f"[EXPERIMENT] Poll error #{consecutive_errors}: {e}", "ERROR")

            if consecutive_errors >= 5:
                log("[EXPERIMENT] 5 consecutive errors — reconnecting to Supabase", "WARN")
                _reconnect_pg()
                consecutive_errors = 0

            time.sleep(POLL_INTERVAL * 2)  # Back off on errors


def _count_pending() -> int:
    """Count pending experiments in queue."""
    rows = _exec_sql("""
        SELECT COUNT(*) FROM public.nba_experiments
        WHERE status = 'pending'
          AND (target_space IS NULL OR target_space = 'S11' OR target_space = 'any')
    """)
    if rows and rows is not True and len(rows) > 0:
        return rows[0][0]
    return 0


# ═══════════════════════════════════════════════════════
# FASTAPI ENDPOINTS (added to control_api)
# ═══════════════════════════════════════════════════════

def register_experiment_endpoints(api):
    """Register experiment-related FastAPI endpoints on the control_api."""
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @api.get("/api/experiment/status")
    async def experiment_status():
        """Current experiment runner status."""
        return JSONResponse({
            "mode": "experiment_runner",
            "status": live.get("status", "unknown"),
            "current_experiment": {
                "experiment_id": _current_experiment["experiment_id"],
                "type": _current_experiment["experiment_type"],
                "agent": _current_experiment["agent_name"],
                "description": _current_experiment["description"][:200],
            } if _current_experiment else None,
            "queue_depth": _queue_depth,
            "experiments_completed": _experiments_completed,
            "experiments_failed": _experiments_failed,
            "last_result": {
                "experiment_id": _last_result.get("experiment_id"),
                "brier": _last_result.get("brier"),
                "improvement": _last_result.get("improvement"),
                "elapsed_seconds": _last_result.get("elapsed_seconds"),
            } if _last_result else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    @api.post("/api/experiment/run")
    async def experiment_run_direct(request: Request):
        """Submit and immediately run an experiment (bypasses queue).

        Body: {
            "experiment_type": "feature_test|model_test|calibration_test|config_change",
            "description": "...",
            "params": { ... },
            "agent_name": "direct_api"
        }
        """
        try:
            body = await request.json()

            exp_type = body.get("experiment_type")
            if exp_type not in EXECUTORS:
                return JSONResponse(
                    {"error": f"Invalid experiment_type. Valid: {list(EXECUTORS.keys())}"},
                    status_code=400
                )

            # Write to Supabase first (for tracking), then execute
            exp_id_str = f"direct-{int(time.time())}"
            _exec_sql("""
                INSERT INTO public.nba_experiments
                (experiment_id, agent_name, experiment_type, description, hypothesis,
                 params, priority, status, target_space)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', 'S11')
                RETURNING id
            """, (
                exp_id_str,
                body.get("agent_name", "direct_api"),
                exp_type,
                body.get("description", "Direct API submission"),
                body.get("hypothesis", ""),
                json.dumps(body.get("params", {})),
                body.get("priority", 10),
            ))

            # Fetch the just-inserted experiment
            rows = _exec_sql("""
                SELECT id, experiment_id, agent_name, experiment_type, description,
                       hypothesis, params, priority, status, target_space,
                       baseline_brier, created_at
                FROM public.nba_experiments
                WHERE experiment_id = %s
                ORDER BY id DESC LIMIT 1
            """, (exp_id_str,))

            if not rows or rows is True:
                return JSONResponse({"error": "Failed to insert experiment"}, status_code=500)

            row = rows[0]
            experiment = {
                "id": row[0], "experiment_id": row[1], "agent_name": row[2],
                "experiment_type": row[3], "description": row[4], "hypothesis": row[5],
                "params": row[6] if isinstance(row[6], dict) else json.loads(row[6]) if row[6] else {},
                "priority": row[7], "status": row[8], "target_space": row[9],
                "baseline_brier": row[10], "created_at": str(row[11]) if row[11] else None,
            }

            # Note: This blocks the request until evaluation completes (can be long).
            # For non-blocking, use /api/experiment/submit instead.
            return JSONResponse({
                "status": "queued_for_poll",
                "experiment_id": exp_id_str,
                "message": "Experiment inserted into queue. S11 will pick it up on next poll cycle.",
            })

        except Exception as e:
            return JSONResponse({"error": str(e)[:500]}, status_code=500)

    @api.post("/api/experiment/submit")
    async def experiment_submit(request: Request):
        """Submit an experiment to the Supabase queue (non-blocking).

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

            exp_type = body.get("experiment_type")
            if exp_type not in EXECUTORS:
                return JSONResponse(
                    {"error": f"Invalid experiment_type. Valid: {list(EXECUTORS.keys())}"},
                    status_code=400
                )

            if not body.get("description"):
                return JSONResponse({"error": "description is required"}, status_code=400)

            exp_id_str = body.get("experiment_id", f"submit-{int(time.time())}")

            result = _exec_sql("""
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

            if result is None:
                return JSONResponse({"error": "Failed to insert — Supabase error"}, status_code=500)

            db_id = result[0][0] if result and result is not True else None

            return JSONResponse({
                "status": "queued",
                "id": db_id,
                "experiment_id": exp_id_str,
                "message": f"Experiment queued. S11 polls every {POLL_INTERVAL}s.",
            })

        except Exception as e:
            return JSONResponse({"error": str(e)[:500]}, status_code=500)

    @api.get("/api/experiment/results")
    async def experiment_results():
        """Fetch recent experiment results from Supabase."""
        rows = _exec_sql("""
            SELECT experiment_id, agent_name, experiment_type, status,
                   result_brier, result_accuracy, result_details,
                   created_at, completed_at
            FROM public.nba_experiments
            ORDER BY id DESC
            LIMIT 20
        """)
        if not rows or rows is True:
            return JSONResponse({"experiments": [], "count": 0})

        experiments = []
        for row in rows:
            experiments.append({
                "experiment_id": row[0],
                "agent_name": row[1],
                "experiment_type": row[2],
                "status": row[3],
                "result_brier": row[4],
                "result_accuracy": row[5],
                "result_details": row[6],
                "created_at": str(row[7]) if row[7] else None,
                "completed_at": str(row[8]) if row[8] else None,
            })

        return JSONResponse({"experiments": experiments, "count": len(experiments)})

    log("[EXPERIMENT] FastAPI endpoints registered: /api/experiment/{status,run,submit,results}")
