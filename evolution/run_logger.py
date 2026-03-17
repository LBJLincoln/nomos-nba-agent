#!/usr/bin/env python3
"""
Run Logger & Auto-Cut — Hedge Fund Grade Evolution Monitoring
================================================================
Logs EVERY generation, cycle, and eval to Supabase.
Auto-cuts evolution when regression detected or stagnation exceeds threshold.

Tables (auto-created):
  - nba_evolution_runs    : one row per cycle (summary)
  - nba_evolution_gens    : one row per generation (detailed)
  - nba_evolution_evals   : one row per individual evaluation
  - nba_evolution_cuts    : log of auto-cut events

Auto-Cut Rules:
  1. REGRESSION CUT: If best Brier increases by > 0.005 for 3 consecutive gens → rollback
  2. STAGNATION CUT: If no improvement for 20 gens → emergency diversify + log
  3. ROI CUT: If ROI drops below -15% → pause betting, continue evolving
  4. DIVERSITY CUT: If population diversity < 0.05 → inject fresh individuals
  5. FEATURE CUT: If selected features < 40 → expand target_features

Designed for real-time monitoring via Supabase dashboard or Telegram.
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ── Supabase connection ──
_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_DATABASE_URL = os.environ.get("DATABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY", os.environ.get("SUPABASE_ANON_KEY", ""))

_pg_pool = None

def _get_pg():
    """Lazy PostgreSQL connection pool (Supabase = PostgreSQL)."""
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    db_url = _DATABASE_URL
    if not db_url:
        return None
    try:
        import psycopg2
        from psycopg2 import pool as pg_pool
        _pg_pool = pg_pool.SimpleConnectionPool(1, 3, db_url)
        return _pg_pool
    except Exception as e:
        print(f"[RUN-LOGGER] PostgreSQL connection failed: {e}")
        return None


def _exec_sql(sql, params=None):
    """Execute SQL on Supabase PostgreSQL. Best-effort, never crashes."""
    pool = _get_pg()
    if not pool:
        return None
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            try:
                return cur.fetchall()
            except Exception:
                return []
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[RUN-LOGGER] SQL error: {e}")
        return None
    finally:
        if conn and pool:
            pool.putconn(conn)


def _ensure_tables():
    """Create logging tables if they don't exist."""
    sqls = [
        """CREATE TABLE IF NOT EXISTS nba_evolution_runs (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMPTZ DEFAULT NOW(),
            cycle INT,
            generation INT,
            best_brier FLOAT,
            best_roi FLOAT,
            best_sharpe FLOAT,
            best_calibration FLOAT,
            best_composite FLOAT,
            best_features INT,
            best_model_type TEXT,
            pop_size INT,
            mutation_rate FLOAT,
            crossover_rate FLOAT,
            stagnation INT,
            games INT,
            feature_candidates INT,
            cycle_duration_s FLOAT,
            avg_composite FLOAT,
            pop_diversity FLOAT,
            top5 JSONB,
            selected_features JSONB
        )""",
        """CREATE TABLE IF NOT EXISTS nba_evolution_gens (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMPTZ DEFAULT NOW(),
            cycle INT,
            generation INT,
            best_brier FLOAT,
            best_roi FLOAT,
            best_sharpe FLOAT,
            best_composite FLOAT,
            n_features INT,
            model_type TEXT,
            mutation_rate FLOAT,
            avg_composite FLOAT,
            pop_diversity FLOAT,
            gen_duration_s FLOAT,
            improved BOOLEAN DEFAULT FALSE
        )""",
        """CREATE TABLE IF NOT EXISTS nba_evolution_cuts (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMPTZ DEFAULT NOW(),
            cut_type TEXT,
            reason TEXT,
            brier_before FLOAT,
            brier_after FLOAT,
            action_taken TEXT,
            params_applied JSONB
        )""",
        """CREATE TABLE IF NOT EXISTS nba_evolution_evals (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMPTZ DEFAULT NOW(),
            generation INT,
            individual_rank INT,
            brier FLOAT,
            roi FLOAT,
            sharpe FLOAT,
            composite FLOAT,
            n_features INT,
            model_type TEXT
        )""",
    ]
    for sql in sqls:
        _exec_sql(sql)


# ── Auto-initialize tables on import ──
_tables_ready = False


class RunLogger:
    """Logs evolution runs to Supabase + local files. Never crashes the main loop."""

    def __init__(self, local_dir=None):
        global _tables_ready
        self.local_dir = Path(local_dir or "/data/run-logs")
        self.local_dir.mkdir(parents=True, exist_ok=True)

        # Auto-cut state
        self.brier_history = []      # last N best Brier values
        self.regression_count = 0    # consecutive regressions
        self.stagnation_count = 0
        self.last_best_brier = 1.0
        self.last_best_composite = 0.0
        self.cuts_applied = 0

        # Ensure Supabase tables exist
        if not _tables_ready:
            _ensure_tables()
            _tables_ready = True
            print("[RUN-LOGGER] Supabase tables ready")

    # ══════════════════════════════════════════
    #  LOG — Record events
    # ══════════════════════════════════════════

    def log_generation(self, cycle, generation, best, mutation_rate, avg_composite, pop_diversity, duration_s):
        """Log one generation result."""
        improved = best["brier"] < self.last_best_brier - 0.0001

        # Supabase
        _exec_sql("""INSERT INTO nba_evolution_gens
            (cycle, generation, best_brier, best_roi, best_sharpe, best_composite,
             n_features, model_type, mutation_rate, avg_composite, pop_diversity,
             gen_duration_s, improved)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (cycle, generation, best["brier"], best["roi"], best["sharpe"],
             best["composite"], best.get("n_features", 0), best.get("model_type", "?"),
             mutation_rate, avg_composite, pop_diversity, duration_s, improved))

        # Track for auto-cut
        self.brier_history.append(best["brier"])
        if len(self.brier_history) > 50:
            self.brier_history = self.brier_history[-50:]

        return improved

    def log_cycle(self, cycle, generation, best, pop_size, mutation_rate, crossover_rate,
                  stagnation, games, feature_candidates, cycle_duration_s,
                  avg_composite, pop_diversity, top5=None, selected_features=None):
        """Log one full cycle (multiple generations) result."""
        _exec_sql("""INSERT INTO nba_evolution_runs
            (cycle, generation, best_brier, best_roi, best_sharpe, best_calibration,
             best_composite, best_features, best_model_type, pop_size, mutation_rate,
             crossover_rate, stagnation, games, feature_candidates, cycle_duration_s,
             avg_composite, pop_diversity, top5, selected_features)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (cycle, generation, best["brier"], best["roi"], best["sharpe"],
             best.get("calibration", 0), best["composite"], best.get("n_features", 0),
             best.get("model_type", "?"), pop_size, mutation_rate, crossover_rate,
             stagnation, games, feature_candidates, cycle_duration_s,
             avg_composite, pop_diversity,
             json.dumps(top5 or [], default=str),
             json.dumps(selected_features or [], default=str)))

        # Local file backup
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "cycle": cycle, "generation": generation,
            "best": best, "pop_size": pop_size,
            "mutation_rate": mutation_rate, "stagnation": stagnation,
        }
        log_file = self.local_dir / f"cycle-{cycle:04d}.json"
        log_file.write_text(json.dumps(entry, indent=2, default=str))

    def log_top_evals(self, generation, top_individuals):
        """Log top 10 individuals for this generation."""
        for rank, ind in enumerate(top_individuals[:10]):
            _exec_sql("""INSERT INTO nba_evolution_evals
                (generation, individual_rank, brier, roi, sharpe, composite, n_features, model_type)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (generation, rank + 1,
                 ind.get("brier", ind.get("fitness", {}).get("brier", 0)),
                 ind.get("roi", ind.get("fitness", {}).get("roi", 0)),
                 ind.get("sharpe", ind.get("fitness", {}).get("sharpe", 0)),
                 ind.get("composite", ind.get("fitness", {}).get("composite", 0)),
                 ind.get("n_features", 0),
                 ind.get("model_type", ind.get("hyperparams", {}).get("model_type", "?"))))

    def log_cut(self, cut_type, reason, brier_before, brier_after, action, params=None):
        """Log an auto-cut event."""
        _exec_sql("""INSERT INTO nba_evolution_cuts
            (cut_type, reason, brier_before, brier_after, action_taken, params_applied)
            VALUES (%s,%s,%s,%s,%s,%s)""",
            (cut_type, reason, brier_before, brier_after, action,
             json.dumps(params or {}, default=str)))
        self.cuts_applied += 1
        print(f"[AUTO-CUT] {cut_type}: {reason} → {action}")

    # ══════════════════════════════════════════
    #  AUTO-CUT — Automatic regression/stagnation handling
    # ══════════════════════════════════════════

    def check_auto_cut(self, current_best, engine_state):
        """
        Check if an auto-cut should be applied.
        Returns: list of actions to take, or empty list.

        Actions are dicts: {"type": "...", "params": {...}}
        The caller (evolution loop) is responsible for executing them.
        """
        actions = []
        brier = current_best.get("brier", 1.0)
        composite = current_best.get("composite", 0)

        # ── RULE 1: REGRESSION CUT ──
        # If Brier is getting worse for 3+ consecutive generations
        if len(self.brier_history) >= 3:
            last3 = self.brier_history[-3:]
            if all(last3[i] > last3[i-1] + 0.0003 for i in range(1, len(last3))):
                self.regression_count += 1
                if self.regression_count >= 2:
                    self.log_cut("REGRESSION", f"Brier increasing 3+ gens: {[f'{b:.4f}' for b in last3]}",
                                 last3[0], last3[-1], "rollback_mutation",
                                 {"mutation_rate": max(0.03, engine_state.get("mutation_rate", 0.1) * 0.5)})
                    actions.append({
                        "type": "config",
                        "params": {"mutation_rate": max(0.03, engine_state.get("mutation_rate", 0.1) * 0.5)},
                    })
                    self.regression_count = 0
            else:
                self.regression_count = 0

        # ── RULE 2: STAGNATION CUT ──
        stagnation = engine_state.get("stagnation", 0)
        if stagnation >= 20:
            self.log_cut("STAGNATION", f"No improvement for {stagnation} generations",
                         brier, brier, "emergency_diversify",
                         {"pop_size": 200, "mutation_rate": 0.20, "target_features": 300})
            actions.append({
                "type": "emergency_diversify",
                "params": {"pop_size": 200, "mutation_rate": 0.20, "target_features": 300},
            })

        # ── RULE 3: ROI CUT ──
        roi = current_best.get("roi", 0)
        if roi < -0.15:
            self.log_cut("ROI_THRESHOLD", f"ROI dropped to {roi:.1%} — betting paused",
                         brier, brier, "pause_betting")
            # Don't stop evolution, just flag for betting logic
            actions.append({"type": "flag", "params": {"pause_betting": True}})

        # ── RULE 4: DIVERSITY CUT ──
        diversity = engine_state.get("pop_diversity", 0)
        if diversity < 3.0 and engine_state.get("pop_size", 0) > 20:
            self.log_cut("DIVERSITY", f"Population diversity {diversity:.1f} too low",
                         brier, brier, "inject_random",
                         {"inject_count": max(10, engine_state.get("pop_size", 50) // 4)})
            actions.append({
                "type": "inject",
                "params": {"count": max(10, engine_state.get("pop_size", 50) // 4)},
            })

        # ── RULE 5: FEATURE CUT ──
        n_features = current_best.get("n_features", 0)
        if 0 < n_features < 40:
            self.log_cut("LOW_FEATURES", f"Only {n_features} features selected",
                         brier, brier, "expand_target",
                         {"target_features": 200})
            actions.append({
                "type": "config",
                "params": {"target_features": 200},
            })

        # ── RULE 6: BRIER FLOOR ──
        # If Brier is stuck above 0.24 for 30+ gens, force aggressive exploration
        if len(self.brier_history) >= 30:
            if all(b > 0.24 for b in self.brier_history[-30:]):
                self.log_cut("BRIER_FLOOR", "Brier stuck above 0.24 for 30+ gens",
                             brier, brier, "full_reset",
                             {"mutation_rate": 0.25, "pop_size": 250, "target_features": 400})
                actions.append({
                    "type": "full_reset",
                    "params": {"mutation_rate": 0.25, "pop_size": 250, "target_features": 400},
                })

        # Update tracking
        if brier < self.last_best_brier:
            self.last_best_brier = brier
        self.last_best_composite = composite

        return actions

    # ══════════════════════════════════════════
    #  QUERY — Read logged data
    # ══════════════════════════════════════════

    def get_recent_runs(self, limit=20):
        """Get recent cycle logs from Supabase."""
        rows = _exec_sql(
            "SELECT * FROM nba_evolution_runs ORDER BY ts DESC LIMIT %s", (limit,))
        return rows or []

    def get_recent_cuts(self, limit=10):
        rows = _exec_sql(
            "SELECT * FROM nba_evolution_cuts ORDER BY ts DESC LIMIT %s", (limit,))
        return rows or []

    def get_brier_trend(self, last_n=50):
        rows = _exec_sql(
            "SELECT generation, best_brier FROM nba_evolution_gens ORDER BY ts DESC LIMIT %s",
            (last_n,))
        if rows:
            return [(r[0], r[1]) for r in reversed(rows)]
        return self.brier_history[-last_n:]

    def get_stats(self):
        """Summary stats for dashboard."""
        total_gens = _exec_sql("SELECT COUNT(*) FROM nba_evolution_gens")
        total_runs = _exec_sql("SELECT COUNT(*) FROM nba_evolution_runs")
        total_cuts = _exec_sql("SELECT COUNT(*) FROM nba_evolution_cuts")
        best_ever = _exec_sql(
            "SELECT MIN(best_brier) FROM nba_evolution_runs")

        return {
            "total_generations": total_gens[0][0] if total_gens else 0,
            "total_cycles": total_runs[0][0] if total_runs else 0,
            "total_cuts": total_cuts[0][0] if total_cuts else 0,
            "best_brier_ever": best_ever[0][0] if best_ever and best_ever[0][0] else None,
            "local_cuts_applied": self.cuts_applied,
            "regression_count": self.regression_count,
            "brier_history_len": len(self.brier_history),
        }
