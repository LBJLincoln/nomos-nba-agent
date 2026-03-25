# %% [markdown]
# # NBA Quant AI — Ultimate GPU Evolution with TabICLv2 + TabPFN
#
# **Goal: Beat all-time best Brier (0.21976) using GPU-only models + evolved features**
#
# **What's new**:
# - TabICLv2 (in-context learning, MIT) as evolvable model type
# - TabPFN-2.5 (Prior-fitted Networks) as evolvable model type
# - Population seeded from S10-S15 best evolved individuals (not random)
# - 5-fold walk-forward (more robust than HF Spaces 2-fold)
# - XGBoost on CUDA, all tree models + ICL models competing
#
# **Instructions**:
# 1. Runtime → Change runtime type → **GPU (T4 free)**
# 2. Colab Secrets (key icon left sidebar):
#    - `DATABASE_URL`: Supabase postgres connection string
#    - `HF_TOKEN`: HuggingFace token (for TabPFN model download)
# 3. Run all cells

# %% [markdown]
# ## 1. Clone Repo & Install Dependencies

# %%
import subprocess, sys, os, time

# Clone/update repo
if not os.path.exists("/content/nomos-nba-agent"):
    subprocess.run(["git", "clone", "https://github.com/LBJLincoln/nomos-nba-agent.git",
                    "/content/nomos-nba-agent"], check=True)
else:
    subprocess.run(["git", "-C", "/content/nomos-nba-agent", "pull"], check=True)

os.chdir("/content/nomos-nba-agent")
sys.path.insert(0, "/content/nomos-nba-agent")

# Install all dependencies (tree + ICL models)
_pkgs = [
    "psycopg2-binary", "xgboost", "lightgbm", "catboost",
    "scikit-learn>=1.3", "pandas", "numpy", "scipy", "nba_api",
    "tabicl",           # TabICLv2 — MIT, soda-inria
    "tabpfn",           # TabPFN-2.5 — Prior-Labs
    "huggingface_hub",  # For model downloads
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + _pkgs)

import torch
print(f"PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")

# Verify TabICLv2 + TabPFN
try:
    from tabicl import TabICLClassifier
    print("TabICLv2: OK")
except ImportError as e:
    print(f"TabICLv2: FAILED ({e})")

try:
    from tabpfn import TabPFNClassifier
    print("TabPFN-2.5: OK")
except ImportError as e:
    print(f"TabPFN-2.5: FAILED ({e})")

# %% [markdown]
# ## 2. Load Secrets & Pre-download Models

# %%
# Supabase connection
try:
    from google.colab import userdata
    os.environ.setdefault("DATABASE_URL", userdata.get("DATABASE_URL"))
    print("DATABASE_URL loaded from Colab secrets")
except Exception:
    print("Set DATABASE_URL manually: os.environ['DATABASE_URL'] = 'postgresql://...'")

# HuggingFace token — needed for TabPFN model download
try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        # Login to huggingface_hub
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("HF_TOKEN loaded — TabPFN download should work")
    else:
        print("HF_TOKEN not set — TabPFN may fail to download")
except Exception:
    print("HF_TOKEN not available — TabPFN may fail to download")

# Pre-download TabPFN checkpoint (cache it before evolution starts)
print("\nPre-downloading model checkpoints...")
try:
    from tabpfn import TabPFNClassifier
    import numpy as np
    _m = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
    _m.fit(np.random.randn(50, 5), np.random.randint(0, 2, 50))
    _m.predict_proba(np.random.randn(10, 5))
    print("  TabPFN-2.5: checkpoint cached OK")
    del _m
except Exception as e:
    print(f"  TabPFN-2.5: cache failed ({e})")
    print("  Manual fix: download from https://huggingface.co/Prior-Labs/TabPFN-v2-clf")
    print("  Place at: /root/.cache/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt")

try:
    from tabicl import TabICLClassifier
    import numpy as np
    _m = TabICLClassifier()
    _m.fit(np.random.randn(50, 5), np.random.randint(0, 2, 50))
    _m.predict_proba(np.random.randn(10, 5))
    print("  TabICLv2: checkpoint cached OK")
    del _m
except Exception as e:
    print(f"  TabICLv2: cache failed ({e})")

import gc; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
# ## 3. Seed Population from HF Space Islands
#
# Instead of random initialization, we fetch the best evolved individuals from
# all 6 HF Space islands (S10-S15) and use them as seeds. This gives the GPU
# evolution a massive head start — we're evolving from Brier ~0.22 instead of ~0.25.

# %%
import requests, json, numpy as np

ISLAND_URLS = {
    "S10": "https://lbjlincoln-nomos-nba-quant.hf.space",
    "S11": "https://lbjlincoln-nomos-nba-quant-2.hf.space",
    "S12": "https://lbjlincoln26-nba-evo-3.hf.space",
    "S13": "https://lbjlincoln26-nba-evo-4.hf.space",
    "S14": "https://nomos42-nba-evo-5.hf.space",
    "S15": "https://nomos42-nba-evo-6.hf.space",
}

def fetch_island_seeds():
    """Fetch best individuals from all 6 HF Space islands."""
    seeds = []
    for name, url in ISLAND_URLS.items():
        try:
            resp = requests.get(f"{url}/api/results", timeout=15)
            if resp.status_code != 200:
                print(f"  {name}: HTTP {resp.status_code}")
                continue
            data = resp.json()
            best = data.get("best", {})
            brier = best.get("brier", 1.0)
            features = best.get("selected_features", [])
            model_type = best.get("model_type", "xgboost")
            n_feat = best.get("n_features", len(features))

            # Also grab top5 if available
            top5 = data.get("top5", [])

            seeds.append({
                "source": name,
                "brier": brier,
                "features": features,
                "model_type": model_type,
                "n_features": n_feat,
            })
            print(f"  {name}: brier={brier:.5f}, model={model_type}, features={n_feat}")

            # Add top5 individuals too (diversified seeds)
            for i, ind in enumerate(top5[:3]):  # top 3 from each island
                seeds.append({
                    "source": f"{name}_top{i+1}",
                    "brier": ind.get("brier", 1.0),
                    "features": ind.get("selected_features", features),
                    "model_type": ind.get("model_type", model_type),
                    "n_features": ind.get("n_features", n_feat),
                })
        except Exception as e:
            print(f"  {name}: {e}")
    return seeds

print("Fetching seeds from 6 HF Space islands...")
island_seeds = fetch_island_seeds()
print(f"\nTotal seeds: {len(island_seeds)}")

# %% [markdown]
# ## 4. Run GPU Evolution with TabICLv2 + TabPFN
#
# **Config**:
# - Pop: 100 (5 islands × 20)
# - Models: xgboost, xgboost_brier, lightgbm, catboost, random_forest, extra_trees, **tabicl**, **tabpfn**
# - 5-fold walk-forward
# - 10 gens/cycle × 100 cycles = 1000 generations
# - Population seeded from island bests

# %%
from evolution.genetic_loop_v3 import (
    run_continuous, GeneticEvolutionEngine, Individual,
    build_features, pull_seasons, load_all_games,
    GPU_MODEL_TYPES, RESULTS_DIR, STATE_DIR, _HAS_LOGGER
)
try:
    from evolution.run_logger import RunLogger
except ImportError:
    RunLogger = None

import numpy as np

print("=" * 70)
print("  NBA QUANT AI — ULTIMATE GPU EVOLUTION")
print(f"  Models: {GPU_MODEL_TYPES}")
print("=" * 70)

# 1. Pull data
print("\n[PHASE 1] Loading data...")
pull_seasons()
games = load_all_games()
print(f"  {len(games)} games loaded")

# 2. Build features
print("\n[PHASE 2] Building features...")
X, y, feature_names = build_features(games)
n_features = X.shape[1]
print(f"  Feature matrix: {X.shape} ({len(feature_names)} features)")

# 3. Initialize engine with seeded population
print("\n[PHASE 3] Initializing engine with seeded population...")

POP_SIZE = 100
N_ISLANDS = 5
TARGET_FEATURES = 80
N_SPLITS = 5

engine = GeneticEvolutionEngine(
    pop_size=POP_SIZE,
    elite_size=max(5, POP_SIZE // 20),
    mutation_rate=0.10,       # Start moderate (seeds are already good)
    crossover_rate=0.80,
    target_features=TARGET_FEATURES,
    n_splits=N_SPLITS,
    n_islands=N_ISLANDS,
    migration_interval=8,
    migrants_per_island=3,
)

# Try restore first (survive Colab disconnects)
if not engine.restore_state():
    # Seed from island bests + fill remaining with random
    engine.n_features = n_features
    engine.population = []

    # Create seeded individuals from island bests
    feature_name_to_idx = {name: i for i, name in enumerate(feature_names)}

    for seed in island_seeds:
        ind = Individual.__new__(Individual)
        ind.features = [0] * n_features

        # Map seed feature names to indices
        seed_feats = seed.get("features", [])
        if isinstance(seed_feats, list) and len(seed_feats) > 0:
            if isinstance(seed_feats[0], str):
                # Feature names — map to indices
                for fname in seed_feats:
                    idx = feature_name_to_idx.get(fname)
                    if idx is not None:
                        ind.features[idx] = 1
            elif isinstance(seed_feats[0], int):
                # Already indices
                for idx in seed_feats:
                    if 0 <= idx < n_features:
                        ind.features[idx] = 1

        # If no features mapped, random init
        if sum(ind.features) < 15:
            prob = TARGET_FEATURES / max(n_features, 1)
            ind.features = [1 if np.random.random() < prob else 0 for _ in range(n_features)]

        ind.hyperparams = {
            "n_estimators": np.random.randint(100, 400),
            "max_depth": np.random.randint(4, 9),
            "learning_rate": 10 ** np.random.uniform(-2.0, -0.7),
            "subsample": np.random.uniform(0.6, 1.0),
            "colsample_bytree": np.random.uniform(0.4, 1.0),
            "min_child_weight": np.random.randint(1, 10),
            "reg_alpha": 10 ** np.random.uniform(-4, 0),
            "reg_lambda": 10 ** np.random.uniform(-4, 0),
            "model_type": seed.get("model_type", np.random.choice(GPU_MODEL_TYPES)),
            "calibration": np.random.choice(["none", "none", "sigmoid"]),  # 67% none
            "nn_hidden_dims": np.random.choice([64, 128, 256]),
            "nn_n_layers": np.random.randint(2, 4),
            "nn_dropout": np.random.uniform(0.1, 0.5),
            "nn_epochs": np.random.randint(20, 100),
            "nn_batch_size": np.random.choice([32, 64, 128]),
        }
        ind.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "calibration_error": 1.0, "composite": 0.0}
        ind.generation = 0
        ind.birth_generation = 0
        ind.pareto_rank = 999
        ind.crowding_dist = 0.0
        ind.island_id = -1
        ind.n_features = sum(ind.features)
        ind._enforce_feature_cap = lambda self=ind: Individual._enforce_feature_cap(self)
        Individual._enforce_feature_cap(ind)
        engine.population.append(ind)

    # Also create TabICLv2 variants of each seed (same features, different model)
    for seed in island_seeds[:10]:  # Top 10 seeds as TabICLv2
        ind = Individual.__new__(Individual)
        ind.features = [0] * n_features
        seed_feats = seed.get("features", [])
        if isinstance(seed_feats, list) and len(seed_feats) > 0:
            if isinstance(seed_feats[0], str):
                for fname in seed_feats:
                    idx = feature_name_to_idx.get(fname)
                    if idx is not None:
                        ind.features[idx] = 1
            elif isinstance(seed_feats[0], int):
                for idx in seed_feats:
                    if 0 <= idx < n_features:
                        ind.features[idx] = 1
        if sum(ind.features) < 15:
            prob = TARGET_FEATURES / max(n_features, 1)
            ind.features = [1 if np.random.random() < prob else 0 for _ in range(n_features)]
        ind.hyperparams = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "model_type": "tabicl",  # Force TabICLv2
            "calibration": "none",
            "nn_hidden_dims": 128,
            "nn_n_layers": 2,
            "nn_dropout": 0.3,
            "nn_epochs": 50,
            "nn_batch_size": 64,
        }
        ind.fitness = {"brier": 1.0, "roi": 0.0, "sharpe": 0.0, "calibration": 1.0, "calibration_error": 1.0, "composite": 0.0}
        ind.generation = 0
        ind.birth_generation = 0
        ind.pareto_rank = 999
        ind.crowding_dist = 0.0
        ind.island_id = -1
        ind.n_features = sum(ind.features)
        Individual._enforce_feature_cap(ind)
        engine.population.append(ind)

    # Fill remaining slots with random individuals
    remaining = POP_SIZE - len(engine.population)
    if remaining > 0:
        for _ in range(remaining):
            ind = Individual(n_features, TARGET_FEATURES,
                             model_type=np.random.choice(GPU_MODEL_TYPES))
            engine.population.append(ind)
    elif len(engine.population) > POP_SIZE:
        engine.population = engine.population[:POP_SIZE]

    print(f"  Seeded: {len(island_seeds)} from islands + TabICLv2 variants")
    print(f"  Total population: {len(engine.population)}")

    # Model type distribution
    from collections import Counter
    mt_counts = Counter(ind.hyperparams["model_type"] for ind in engine.population)
    print(f"  Model types: {dict(mt_counts)}")
else:
    engine.resize_population_features(n_features)

# 4. Run logger
run_logger = None
if RunLogger:
    try:
        run_logger = RunLogger(local_dir=str(RESULTS_DIR / "run-logs"))
        print("[RUN-LOGGER] Supabase logging ACTIVE")
    except Exception as e:
        print(f"[RUN-LOGGER] Init failed: {e}")

# 5. RUN EVOLUTION
print(f"\n{'='*70}")
print("  STARTING GPU EVOLUTION — Target: Brier < 0.20")
print(f"  Pop={POP_SIZE}, Islands={N_ISLANDS}, Folds={N_SPLITS}")
print(f"  Models: {GPU_MODEL_TYPES}")
print(f"{'='*70}\n")

# Use the engine's run_generation method for each generation
TOTAL_CYCLES = 100
GENS_PER_CYCLE = 10

for cycle in range(1, TOTAL_CYCLES + 1):
    cycle_start = time.time()
    print(f"\n{'='*60}")
    print(f"  CYCLE {cycle}/{TOTAL_CYCLES} — {GENS_PER_CYCLE} generations")
    print(f"{'='*60}")

    for gen_i in range(GENS_PER_CYCLE):
        engine.generation += 1
        gen_start = time.time()

        # Evaluate all unevaluated individuals
        n_eval = 0
        for ind in engine.population:
            if ind.fitness.get("composite", 0.0) <= 0.0 or ind.fitness.get("brier", 1.0) >= 0.99:
                from evolution.genetic_loop_v3 import evaluate_individual
                evaluate_individual(ind, X, y, n_splits=N_SPLITS, use_gpu=engine.use_gpu)
                n_eval += 1

        # Sort by composite fitness (descending)
        engine.population.sort(key=lambda x: x.fitness.get("composite", 0.0), reverse=True)

        # Track best
        gen_best = engine.population[0]
        if engine.best_ever is None or gen_best.fitness["brier"] < engine.best_ever.fitness["brier"]:
            engine.best_ever = gen_best
            improved = True
        else:
            improved = False

        best_b = engine.best_ever.fitness["brier"]
        gen_b = gen_best.fitness["brier"]
        gen_mt = gen_best.hyperparams["model_type"]
        gen_nf = gen_best.n_features
        gen_dur = time.time() - gen_start

        marker = " *** NEW BEST ***" if improved else ""
        print(f"  Gen {engine.generation}: best={gen_b:.5f} ({gen_mt}, {gen_nf}f) "
              f"| all-time={best_b:.5f} | {n_eval} evals in {gen_dur:.0f}s{marker}")

        # Log to Supabase
        if run_logger:
            try:
                run_logger.log_generation(
                    cycle=cycle,
                    generation=engine.generation,
                    best_brier=gen_b,
                    best_roi=gen_best.fitness.get("roi", 0),
                    best_sharpe=gen_best.fitness.get("sharpe", 0),
                    best_composite=gen_best.fitness.get("composite", 0),
                    n_features=gen_nf,
                    model_type=gen_mt,
                    mutation_rate=engine.mutation_rate,
                    avg_composite=np.mean([ind.fitness.get("composite", 0) for ind in engine.population]),
                    pop_diversity=0.0,
                    gen_duration_s=gen_dur,
                    improved=improved,
                )
            except Exception:
                pass

        # Selection + Crossover + Mutation → next generation
        import random
        elite_size = max(5, POP_SIZE // 20)
        elite = engine.population[:elite_size]

        # Tournament selection
        def tournament(pop, k=5):
            contestants = random.sample(pop, min(k, len(pop)))
            return min(contestants, key=lambda x: x.fitness.get("brier", 1.0))

        children = list(elite)  # Keep elite unchanged
        while len(children) < POP_SIZE:
            if random.random() < engine.crossover_rate:
                p1 = tournament(engine.population)
                p2 = tournament(engine.population)
                child = Individual.crossover(p1, p2)
            else:
                child = tournament(engine.population)
                child = Individual.crossover(child, child)  # clone via crossover

            child.mutate(engine.mutation_rate)

            # Occasionally force TabICLv2/TabPFN model type (ensure representation)
            if random.random() < 0.15:
                child.hyperparams["model_type"] = random.choice(["tabicl", "tabpfn"])

            children.append(child)

        engine.population = children[:POP_SIZE]

        # Adaptive mutation decay
        engine.mutation_rate = max(engine.mut_floor,
                                    engine.mutation_rate * engine.mut_decay)

        # Memory management
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cycle_dur = time.time() - cycle_start
    print(f"\n  Cycle {cycle} done in {cycle_dur:.0f}s — Best ever: {engine.best_ever.fitness['brier']:.5f} "
          f"({engine.best_ever.hyperparams['model_type']}, {engine.best_ever.n_features}f)")

    # Model type distribution
    mt_counts = Counter(ind.hyperparams["model_type"] for ind in engine.population)
    print(f"  Model distribution: {dict(mt_counts)}")

    # Save state (survive disconnects)
    try:
        state = {
            "generation": engine.generation,
            "n_features": n_features,
            "mutation_rate": engine.mutation_rate,
            "stagnation_counter": engine.stagnation_counter,
            "population": [
                {
                    "features": ind.features,
                    "hyperparams": {k: (v if not callable(v) else str(v)) for k, v in ind.hyperparams.items()},
                    "fitness": ind.fitness,
                    "generation": getattr(ind, "generation", 0),
                    "birth_generation": getattr(ind, "birth_generation", 0),
                }
                for ind in engine.population
            ],
            "best_ever": {
                "features": engine.best_ever.features,
                "hyperparams": {k: (v if not callable(v) else str(v)) for k, v in engine.best_ever.hyperparams.items()},
                "fitness": engine.best_ever.fitness,
                "generation": getattr(engine.best_ever, "generation", 0),
            } if engine.best_ever else None,
        }
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / "population.json").write_text(json.dumps(state, default=str))
    except Exception as e:
        print(f"  [SAVE] Failed: {e}")

    # Save best results
    if engine.best_ever:
        results = {
            "best": {
                "brier": engine.best_ever.fitness["brier"],
                "roi": engine.best_ever.fitness.get("roi", 0),
                "sharpe": engine.best_ever.fitness.get("sharpe", 0),
                "composite": engine.best_ever.fitness.get("composite", 0),
                "model_type": engine.best_ever.hyperparams["model_type"],
                "n_features": engine.best_ever.n_features,
                "selected_features": [feature_names[i] for i in engine.best_ever.selected_indices()],
            },
            "generation": engine.generation,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "evolution-latest.json").write_text(json.dumps(results, indent=2))

# %% [markdown]
# ## 5. Final Results

# %%
print("\n" + "=" * 70)
print("  FINAL RESULTS")
print("=" * 70)

if engine.best_ever:
    b = engine.best_ever
    print(f"  Best Brier:    {b.fitness['brier']:.5f}")
    print(f"  ROI:           {b.fitness.get('roi', 0):.4f}")
    print(f"  Sharpe:        {b.fitness.get('sharpe', 0):.4f}")
    print(f"  Model:         {b.hyperparams['model_type']}")
    print(f"  Features:      {b.n_features}")
    print(f"  Generation:    {engine.generation}")
    print(f"  Calibration:   {b.hyperparams.get('calibration', 'none')}")

    # Top 5 by Brier
    top5 = sorted(engine.population, key=lambda x: x.fitness.get("brier", 1.0))[:5]
    print(f"\n  Top 5:")
    for i, ind in enumerate(top5):
        print(f"    #{i+1}: brier={ind.fitness['brier']:.5f} | {ind.hyperparams['model_type']} | {ind.n_features}f")

    # Model type breakdown in top 20
    top20 = sorted(engine.population, key=lambda x: x.fitness.get("brier", 1.0))[:20]
    mt_top20 = Counter(ind.hyperparams["model_type"] for ind in top20)
    print(f"\n  Model types in top 20: {dict(mt_top20)}")

    # Compare vs all-time best
    print(f"\n  All-time best (MOVDA-era): 0.22041")
    print(f"  All-time best (ever):      0.21976")
    delta = b.fitness['brier'] - 0.21976
    print(f"  Delta vs all-time:         {delta:+.5f} ({'NEW RECORD!' if delta < 0 else 'keep evolving'})")
else:
    print("  No results yet.")
