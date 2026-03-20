#!/usr/bin/env python3
"""
NBA Quant — Genetic Evolution Loop (Karpathy-style)
=====================================================
Real evolutionary optimization, not "ask LLM to suggest improvements."

Architecture:
  1. POPULATION: N model configurations (feature subsets + hyperparams)
  2. FITNESS: Walk-forward backtest ROI + Brier score + Sharpe ratio
  3. SELECTION: Tournament selection (top performers survive)
  4. CROSSOVER: Combine feature sets from two parents
  5. MUTATION: Add/remove features, tweak hyperparams
  6. RESEARCH: LLM agent searches latest papers, suggests new features
  7. SELF-DIAGNOSTIC: System detects its own weaknesses

Inspired by:
  - Karpathy's training loop (measure everything, iterate fast)
  - Starlizard's 500+ feature genetic selection
  - Becker's Kalshi microstructure analysis (market data = alpha)
  - OpenAI's evolutionary strategies

THIS RUNS ON HF SPACES ONLY — NOT ON VM.
"""

import os, sys, json, time, random, math
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.engine import NBAFeatureEngine, genetic_feature_selection


class EvolutionConfig:
    """Configuration for evolution loop."""
    POPULATION_SIZE = 500       # 5 islands x 100 individuals
    N_ISLANDS = 5               # Number of sub-populations
    ISLAND_SIZE = 100            # Individuals per island
    N_GENERATIONS = 100         # Generations per cycle
    MUTATION_RATE = 0.15        # Start high, decay to 0.05
    MUT_DECAY_RATE = 0.995      # Per-generation decay
    MUT_FLOOR = 0.05            # Minimum mutation rate
    CROSSOVER_RATE = 0.85       # Constant high recombination
    ELITE_SIZE = 25             # Top 5% survive unchanged
    TARGET_FEATURES = 200       # Target feature count
    MIN_FEATURES = 50           # Minimum features
    MAX_FEATURES = 350          # Maximum features
    BACKTEST_SEASONS = 3        # Walk-forward test seasons
    MIGRATION_INTERVAL = 10     # Migrate between islands every N gens
    MIGRANTS_PER_ISLAND = 5     # Best N individuals migrate
    FITNESS_WEIGHTS = {
        "brier": 0.4,           # Prediction accuracy
        "roi": 0.25,            # Return on investment
        "sharpe": 0.2,          # Risk-adjusted return
        "calibration": 0.15,    # Probability calibration
    }
    RESEARCH_INTERVAL = 10      # Run research every N generations
    DIAGNOSTIC_INTERVAL = 5     # Self-diagnostic every N generations
    # Model types the GA can evolve
    MODEL_TYPES = [
        "xgboost", "lightgbm", "catboost", "rf", "logistic",
        "lstm", "transformer", "tabnet", "ft_transformer",
        "deep_ensemble", "autogluon",
    ]


class Individual:
    """One model configuration in the population."""

    def __init__(self, n_features, target=200, model_type=None):
        # Feature selection mask
        prob = target / n_features
        self.features = [1 if random.random() < prob else 0 for _ in range(n_features)]

        # Hyperparameters (evolved)
        model_types = EvolutionConfig.MODEL_TYPES
        self.hyperparams = {
            "n_estimators": random.randint(100, 600),
            "max_depth": random.randint(3, 10),
            "learning_rate": 10 ** random.uniform(-2.5, -0.5),
            "subsample": random.uniform(0.5, 1.0),
            "colsample_bytree": random.uniform(0.3, 1.0),
            "min_child_weight": random.randint(1, 15),
            "reg_alpha": 10 ** random.uniform(-6, 1),
            "reg_lambda": 10 ** random.uniform(-6, 1),
            "model_type": model_type or random.choice(model_types),
            "calibration": random.choice(["isotonic", "sigmoid", "none"]),
            "stacking": random.choice([True, False]),
            # Neural net hyperparams
            "nn_hidden_dims": random.choice([64, 128, 256]),
            "nn_n_layers": random.randint(2, 4),
            "nn_dropout": random.uniform(0.1, 0.5),
            "nn_epochs": random.randint(20, 100),
        }

        # Fitness scores
        self.fitness = {
            "brier": 1.0,
            "roi": 0.0,
            "sharpe": 0.0,
            "calibration": 1.0,
            "composite": 0.0,
        }
        self.pareto_rank = 999
        self.crowding_dist = 0.0
        self.island_id = -1
        self.generation = 0
        self.n_features = sum(self.features)

    def selected_indices(self):
        return [i for i, bit in enumerate(self.features) if bit]

    def to_dict(self):
        return {
            "n_features": self.n_features,
            "hyperparams": self.hyperparams,
            "fitness": self.fitness,
            "generation": self.generation,
        }

    @staticmethod
    def crossover(parent1, parent2):
        """Two-point crossover on features + blend hyperparams."""
        child = Individual.__new__(Individual)
        n = len(parent1.features)

        # Feature crossover (two-point)
        pt1 = random.randint(0, n - 1)
        pt2 = random.randint(pt1, n - 1)
        child.features = parent1.features[:pt1] + parent2.features[pt1:pt2] + parent1.features[pt2:]

        # Hyperparameter blending
        child.hyperparams = {}
        for key in parent1.hyperparams:
            if isinstance(parent1.hyperparams[key], (int, float)):
                # Blend with random weight
                w = random.random()
                val = w * parent1.hyperparams[key] + (1 - w) * parent2.hyperparams[key]
                if isinstance(parent1.hyperparams[key], int):
                    val = int(round(val))
                child.hyperparams[key] = val
            else:
                # Categorical: random pick
                child.hyperparams[key] = random.choice([parent1.hyperparams[key], parent2.hyperparams[key]])

        child.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "composite": 0.0}
        child.generation = max(parent1.generation, parent2.generation) + 1
        child.n_features = sum(child.features)
        return child

    def mutate(self, rate=0.03):
        """Mutate features and hyperparams."""
        # Feature mutations
        for i in range(len(self.features)):
            if random.random() < rate:
                self.features[i] = 1 - self.features[i]

        # Hyperparameter mutations
        model_types = EvolutionConfig.MODEL_TYPES
        if random.random() < 0.1:
            self.hyperparams["n_estimators"] = max(50, self.hyperparams["n_estimators"] + random.randint(-100, 100))
        if random.random() < 0.1:
            self.hyperparams["max_depth"] = max(2, min(12, self.hyperparams["max_depth"] + random.randint(-2, 2)))
        if random.random() < 0.1:
            self.hyperparams["learning_rate"] *= 10 ** random.uniform(-0.3, 0.3)
        if random.random() < 0.08:
            self.hyperparams["model_type"] = random.choice(model_types)
        if random.random() < 0.05:
            self.hyperparams["calibration"] = random.choice(["isotonic", "sigmoid", "none"])
        # Neural net hyperparams
        if random.random() < 0.10:
            self.hyperparams["nn_hidden_dims"] = random.choice([64, 128, 256, 512])
        if random.random() < 0.10:
            self.hyperparams["nn_n_layers"] = max(1, min(6, self.hyperparams.get("nn_n_layers", 2) + random.randint(-1, 1)))

        self.n_features = sum(self.features)


class EvolutionLoop:
    """
    Main evolution loop — runs 24/7 on HF Space.

    Each generation:
      1. Evaluate population fitness (walk-forward backtest)
      2. Select parents (tournament)
      3. Crossover + mutate to create offspring
      4. Periodically: research agent + self-diagnostic
      5. Save best model + feature set
    """

    def __init__(self, data_dir=None, results_dir=None):
        self.config = EvolutionConfig()
        self.data_dir = Path(data_dir or "/app/data")
        self.results_dir = Path(results_dir or "/app/data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.feature_engine = NBAFeatureEngine(include_market=False)
        self.n_feature_candidates = len(self.feature_engine.feature_names)

        self.population = []
        self.generation = 0
        self.best_ever = None
        self.history = []  # (gen, best_brier, best_roi, n_features)
        self.research_findings = []

        print(f"Evolution Loop initialized: {self.n_feature_candidates} feature candidates")

    def initialize_population(self):
        """Create initial random population."""
        self.population = [
            Individual(self.n_feature_candidates, self.config.TARGET_FEATURES)
            for _ in range(self.config.POPULATION_SIZE)
        ]
        print(f"Population: {self.config.POPULATION_SIZE} individuals, "
              f"~{self.config.TARGET_FEATURES} features each")

    def load_data(self):
        """Load all game data."""
        hist_dir = self.data_dir / "historical"
        games = []
        for f in sorted(hist_dir.glob("games-*.json")):
            data = json.loads(f.read_text())
            items = data if isinstance(data, list) else data.get("games", [])
            games.extend(items)
        games.sort(key=lambda g: g.get("game_date", g.get("date", "")))
        return games

    def evaluate(self, individual, X, y):
        """
        Evaluate individual fitness using walk-forward backtest.

        Fitness = weighted combination of:
          - Brier score (lower = better)
          - ROI from value bets (higher = better)
          - Sharpe ratio (higher = better)
          - Calibration error (lower = better)
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import brier_score_loss
        from sklearn.calibration import CalibratedClassifierCV

        selected = individual.selected_indices()
        if len(selected) < self.config.MIN_FEATURES:
            individual.fitness["composite"] = -1.0
            return

        X_sub = X[:, selected]
        tscv = TimeSeriesSplit(n_splits=self.config.BACKTEST_SEASONS)

        # Build model based on hyperparams
        model = self._build_model(individual.hyperparams)
        if model is None:
            individual.fitness["composite"] = -1.0
            return

        briers, rois, preds_all, y_all = [], [], [], []

        for ti, vi in tscv.split(X_sub):
            try:
                m = self._clone_model(model)

                # Calibration wrapper
                if individual.hyperparams["calibration"] != "none":
                    m = CalibratedClassifierCV(m, method=individual.hyperparams["calibration"], cv=3)

                m.fit(X_sub[ti], y[ti])
                probs = m.predict_proba(X_sub[vi])[:, 1]

                # Brier score
                brier = brier_score_loss(y[vi], probs)
                briers.append(brier)

                # Simulated betting ROI
                roi = self._simulate_betting(probs, y[vi])
                rois.append(roi)

                preds_all.extend(probs)
                y_all.extend(y[vi])

            except Exception:
                briers.append(0.30)
                rois.append(-0.10)

        # Calculate fitness components
        avg_brier = np.mean(briers)
        avg_roi = np.mean(rois)

        # Sharpe ratio of per-fold ROIs
        sharpe = np.mean(rois) / max(np.std(rois), 0.01)

        # Calibration: how well do predicted probs match reality?
        cal_error = self._calibration_error(np.array(preds_all), np.array(y_all))

        individual.fitness = {
            "brier": avg_brier,
            "roi": avg_roi,
            "sharpe": sharpe,
            "calibration": cal_error,
            "composite": (
                self.config.FITNESS_WEIGHTS["brier"] * (1 - avg_brier) +
                self.config.FITNESS_WEIGHTS["roi"] * max(0, avg_roi) +
                self.config.FITNESS_WEIGHTS["sharpe"] * max(0, sharpe / 3) +
                self.config.FITNESS_WEIGHTS["calibration"] * (1 - cal_error)
            ),
        }

    def _build_model(self, hp):
        """Build ML model from hyperparameters."""
        try:
            if hp["model_type"] == "xgboost":
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=hp["n_estimators"],
                    max_depth=hp["max_depth"],
                    learning_rate=hp["learning_rate"],
                    subsample=hp["subsample"],
                    colsample_bytree=hp["colsample_bytree"],
                    min_child_weight=hp["min_child_weight"],
                    reg_alpha=hp["reg_alpha"],
                    reg_lambda=hp["reg_lambda"],
                    eval_metric="logloss", random_state=42, n_jobs=-1,
                )
            elif hp["model_type"] == "lightgbm":
                import lightgbm as lgbm
                return lgbm.LGBMClassifier(
                    n_estimators=hp["n_estimators"],
                    max_depth=hp["max_depth"],
                    learning_rate=hp["learning_rate"],
                    subsample=hp["subsample"],
                    num_leaves=min(2**hp["max_depth"] - 1, 127),
                    reg_alpha=hp["reg_alpha"],
                    reg_lambda=hp["reg_lambda"],
                    verbose=-1, random_state=42, n_jobs=-1,
                )
            elif hp["model_type"] == "catboost":
                from catboost import CatBoostClassifier
                return CatBoostClassifier(
                    iterations=hp["n_estimators"],
                    depth=min(hp["max_depth"], 10),
                    learning_rate=hp["learning_rate"],
                    l2_leaf_reg=hp["reg_lambda"],
                    verbose=0, random_state=42,
                )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                learning_rate=hp["learning_rate"],
                random_state=42,
            )
        return None

    def _clone_model(self, model):
        """Create a fresh copy of the model."""
        return type(model)(**model.get_params())

    def _simulate_betting(self, probs, actuals, edge_threshold=0.05):
        """
        Simulate flat betting on games where model has edge.

        Only bet when model probability differs from implied 50/50 by > threshold.
        """
        bankroll = 1000
        stake = 10  # Flat $10 bets
        profit = 0

        for prob, actual in zip(probs, actuals):
            if prob > 0.5 + edge_threshold:
                # Bet on home (model says home wins with edge)
                fair_odds = 1 / prob
                if actual == 1:
                    profit += stake * (fair_odds - 1)
                else:
                    profit -= stake
            elif prob < 0.5 - edge_threshold:
                # Bet on away
                fair_odds = 1 / (1 - prob)
                if actual == 0:
                    profit += stake * (fair_odds - 1)
                else:
                    profit -= stake

        n_bets = sum(1 for p in probs if abs(p - 0.5) > edge_threshold)
        if n_bets == 0:
            return 0.0
        return profit / (n_bets * stake)

    def _calibration_error(self, probs, actuals, n_bins=10):
        """Expected Calibration Error (ECE)."""
        if len(probs) == 0:
            return 1.0
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i+1])
            if mask.sum() == 0:
                continue
            avg_pred = probs[mask].mean()
            avg_actual = actuals[mask].mean()
            ece += mask.sum() / len(probs) * abs(avg_pred - avg_actual)
        return ece

    def evolve_generation(self, X, y):
        """Run one generation of evolution."""
        self.generation += 1

        # 1. Evaluate all individuals
        for i, ind in enumerate(self.population):
            self.evaluate(ind, X, y)

        # 2. Sort by composite fitness
        self.population.sort(key=lambda x: x.fitness["composite"], reverse=True)
        best = self.population[0]

        # Track best ever
        if self.best_ever is None or best.fitness["composite"] > self.best_ever.fitness["composite"]:
            self.best_ever = Individual.__new__(Individual)
            self.best_ever.features = best.features[:]
            self.best_ever.hyperparams = dict(best.hyperparams)
            self.best_ever.fitness = dict(best.fitness)
            self.best_ever.n_features = best.n_features
            self.best_ever.generation = self.generation

        self.history.append({
            "generation": self.generation,
            "best_brier": best.fitness["brier"],
            "best_roi": best.fitness["roi"],
            "best_sharpe": best.fitness["sharpe"],
            "best_composite": best.fitness["composite"],
            "n_features": best.n_features,
            "model_type": best.hyperparams["model_type"],
            "avg_fitness": np.mean([ind.fitness["composite"] for ind in self.population]),
        })

        print(f"Gen {self.generation}: "
              f"Brier={best.fitness['brier']:.4f} "
              f"ROI={best.fitness['roi']:.1%} "
              f"Features={best.n_features} "
              f"Model={best.hyperparams['model_type']} "
              f"Composite={best.fitness['composite']:.4f}")

        # 3. Create new population
        new_pop = []

        # Elitism: keep top N unchanged
        for i in range(self.config.ELITE_SIZE):
            elite = Individual.__new__(Individual)
            elite.features = self.population[i].features[:]
            elite.hyperparams = dict(self.population[i].hyperparams)
            elite.fitness = dict(self.population[i].fitness)
            elite.n_features = self.population[i].n_features
            elite.generation = self.population[i].generation
            new_pop.append(elite)

        # Fill rest with crossover + mutation
        while len(new_pop) < self.config.POPULATION_SIZE:
            # Tournament selection
            p1 = self._tournament_select(5)
            p2 = self._tournament_select(5)

            if random.random() < self.config.CROSSOVER_RATE:
                child = Individual.crossover(p1, p2)
            else:
                child = Individual.__new__(Individual)
                child.features = p1.features[:]
                child.hyperparams = dict(p1.hyperparams)
                child.fitness = dict(p1.fitness)
                child.n_features = p1.n_features
                child.generation = self.generation

            child.mutate(self.config.MUTATION_RATE)
            new_pop.append(child)

        self.population = new_pop

    def _tournament_select(self, k=5):
        """Tournament selection: pick best of k random individuals."""
        contestants = random.sample(self.population, min(k, len(self.population)))
        return max(contestants, key=lambda x: x.fitness["composite"])

    def run(self, n_generations=None):
        """
        Main evolution loop.

        Runs continuously, saving results after each generation.
        """
        n_gens = n_generations or self.config.N_GENERATIONS

        print(f"\n{'='*60}")
        print(f"EVOLUTION LOOP — {n_gens} generations")
        print(f"{'='*60}")

        # Load data
        games = self.load_data()
        print(f"Games loaded: {len(games)}")

        if len(games) < 500:
            print("Not enough games for evolution!")
            return

        # Build full feature matrix
        print("Building feature matrix...")
        X, y, feature_names = self.feature_engine.build(games)
        print(f"Feature matrix: {X.shape}")

        # Initialize population
        if not self.population:
            self.initialize_population()

        # Evolution loop
        for gen in range(n_gens):
            start = time.time()

            self.evolve_generation(X, y)

            elapsed = time.time() - start
            print(f"  ({elapsed:.1f}s)")

            # Periodic research (every N generations)
            if gen > 0 and gen % self.config.RESEARCH_INTERVAL == 0:
                self._run_research()

            # Periodic self-diagnostic
            if gen > 0 and gen % self.config.DIAGNOSTIC_INTERVAL == 0:
                self._run_diagnostic()

            # Save results
            self.save_results()

        print(f"\n{'='*60}")
        print(f"EVOLUTION COMPLETE — {n_gens} generations")
        print(f"Best: Brier={self.best_ever.fitness['brier']:.4f} "
              f"ROI={self.best_ever.fitness['roi']:.1%} "
              f"Features={self.best_ever.n_features}")
        print(f"{'='*60}")

    def _run_research(self):
        """Call LLM to research latest papers and techniques."""
        print(f"  [Research Agent] Searching latest NBA quant papers...")
        # This will be called via LiteLLM in the HF Space version
        # For now, log the intent
        self.research_findings.append({
            "generation": self.generation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        })

    def _run_diagnostic(self):
        """Self-diagnostic: find weaknesses in current best model."""
        if not self.best_ever:
            return
        best = self.best_ever
        issues = []

        if best.fitness["brier"] > 0.24:
            issues.append("Brier score > 0.24 — model barely better than baseline")
        if best.fitness["roi"] < 0.0:
            issues.append("Negative ROI — model loses money on value bets")
        if best.fitness["calibration"] > 0.05:
            issues.append("Calibration error > 5% — probabilities are unreliable")
        if best.n_features < 100:
            issues.append(f"Only {best.n_features} features — try wider search")
        if best.n_features > 300:
            issues.append(f"{best.n_features} features — may be overfitting")

        # Check feature diversity
        selected = best.selected_indices()
        feature_names = self.feature_engine.feature_names
        categories = defaultdict(int)
        for idx in selected:
            name = feature_names[idx]
            if "market" in name or "spread" in name or "clv" in name:
                categories["market"] += 1
            elif "rest" in name or "travel" in name or "fatigue" in name:
                categories["schedule"] += 1
            elif "elo" in name or "h2h" in name:
                categories["matchup"] += 1
            elif "efg" in name or "ortg" in name or "pace" in name:
                categories["efficiency"] += 1
            else:
                categories["other"] += 1

        if categories["market"] == 0:
            issues.append("ZERO market features — missing microstructure alpha!")
        if categories["schedule"] < 5:
            issues.append("Few schedule features — missing fatigue/travel edge")

        if issues:
            print(f"  [Self-Diagnostic] {len(issues)} issues found:")
            for issue in issues:
                print(f"    - {issue}")

    def save_results(self):
        """Save current state to disk."""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation": self.generation,
            "population_size": len(self.population),
            "feature_candidates": self.n_feature_candidates,
            "best": self.best_ever.to_dict() if self.best_ever else None,
            "top5": [ind.to_dict() for ind in sorted(
                self.population, key=lambda x: x.fitness["composite"], reverse=True
            )[:5]],
            "history": self.history[-50:],  # Last 50 generations
            "research_findings": self.research_findings[-10:],
        }

        out = self.results_dir / "evolution-status.json"
        out.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    loop = EvolutionLoop(
        data_dir=Path("data"),
        results_dir=Path("data/results"),
    )
    loop.run(n_generations=20)  # Quick test
