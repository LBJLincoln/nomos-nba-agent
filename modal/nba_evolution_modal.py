"""
NBA Quant AI — Modal GPU Evolution
===================================
Architecture: local evolution loop dispatches individual evals to Modal T4 GPUs.
Each eval is a Modal function call (stateless), results collected locally.
Persistent state + feature cache stored on Modal Volume.

Usage:
    modal run nba_evolution_modal.py                       # fresh run (T4, 800 gens)
    modal run nba_evolution_modal.py --resume              # resume from local state JSON
    modal run nba_evolution_modal.py --gens 200            # custom gen count
    modal run nba_evolution_modal.py --platform modal_a10g # A10G GPU, larger pop
    modal run nba_evolution_modal.py --rebuild-cache       # force rebuild feature cache
    modal deploy nba_evolution_modal.py                    # deploy as persistent app

Modal free tier: $30/mo compute credits (~hundreds of T4-hours).
Per-second billing — no idle cost when not running evals.

Requirements:
    pip install modal
    modal token new   # authenticate once
    # Set secrets in Modal dashboard or via CLI:
    modal secret create nba-secrets DATABASE_URL=<pooler_url> HF_TOKEN=<token>
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import modal

# ── Modal app + volume ──────────────────────────────────────────────────────
app = modal.App("nba-evolution")

# Persistent volume: feature cache (large .npz) + evolution state (JSON)
vol = modal.Volume.from_name("nba-features", create_if_missing=True)

# Image: GPU-capable, all deps, repo cloned at build time
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        # Core ML
        "torch",
        "xgboost>=2.0",
        "lightgbm",
        "catboost",
        "scikit-learn",
        "tabicl",
        # Data
        "numpy",
        "pandas",
        "scipy",
        # Utilities
        "requests",
        "huggingface_hub",
        "psycopg2-binary",
    )
    # Clone feature engine into image so it's available without network at eval time.
    # The full data (historical JSON) is NOT cloned here — it lives on the Volume.
    .run_commands(
        "git clone --depth=1 https://github.com/LBJLincoln/nomos-nba-agent.git /repo",
        "echo 'Repo cloned'",
    )
    .env({"PYTHONPATH": "/repo/hf-space"})
)

# ── Constants / paths inside Volume ────────────────────────────────────────
VOLUME_MOUNT = "/data"
CACHE_FILE   = f"{VOLUME_MOUNT}/features_cache_v38.npz"
STATE_FILE   = f"{VOLUME_MOUNT}/evolution_state_modal.json"
BEST_FILE    = f"{VOLUME_MOUNT}/best_gpu_features.json"

# ── Evolution hyper-params ──────────────────────────────────────────────────
PLATFORM_CONFIGS: dict[str, dict] = {
    "colab_free":  {"POP": 24, "FOLDS": 2, "GENS": 500, "ELITE": 4, "TIMEOUT": 90},
    "colab_pro":   {"POP": 40, "FOLDS": 3, "GENS": 500, "ELITE": 6, "TIMEOUT": 120},
    "modal_t4":    {"POP": 32, "FOLDS": 2, "GENS": 800, "ELITE": 6, "TIMEOUT": 120},
    "modal_a10g":  {"POP": 48, "FOLDS": 3, "GENS": 800, "ELITE": 8, "TIMEOUT": 150},
}

TARGET_FEATURES  = 60
MAX_FEATURES     = 200
CROSSOVER_RATE   = 0.80
MUT_FLOOR        = 0.05
MUT_DECAY        = 0.998
SUBSAMPLE        = 6000
PURGE_GAP        = 5

MODEL_WEIGHTS = {
    "tabicl":     50,
    "xgboost":    15,
    "catboost":   10,
    "lightgbm":   10,
    "extra_trees":15,
}

ISLANDS = {
    "S10": "https://nomos42-nba-quant.hf.space",
    "S11": "https://nomos42-nba-quant-2.hf.space",
    "S12": "https://nomos42-nba-evo-3.hf.space",
    "S13": "https://nomos42-nba-evo-4.hf.space",
    "S14": "https://nomos42-nba-evo-5.hf.space",
    "S15": "https://nomos42-nba-evo-6.hf.space",
}


# ══════════════════════════════════════════════════════════════════════════════
# REMOTE FUNCTIONS (run on Modal GPU workers)
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    gpu="T4",
    image=image,
    volumes={VOLUME_MOUNT: vol},
    timeout=600,
    retries=1,
)
def build_feature_cache() -> dict:
    """Build feature cache from historical JSON files and save to Volume.

    Run once (or when engine version changes). Returns shape info.
    The historical game files must already be on the Volume at
    /data/historical/games-*.json  OR  inside the cloned repo.
    """
    import gc
    import json
    import sys
    import time
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, "/repo/hf-space")

    cache_path = Path(CACHE_FILE)
    if cache_path.exists():
        cached = np.load(str(cache_path), allow_pickle=True)
        shape = tuple(cached["X"].shape)
        n_feat = len(cached["feature_names"])
        print(f"Cache already exists: {shape}, {n_feat} features")
        return {"cached": True, "shape": shape, "n_features": n_feat}

    print("Building features from historical JSON files…")
    t0 = time.time()

    # Try Volume path first, fall back to repo
    data_dirs = [
        Path(f"{VOLUME_MOUNT}/historical"),
        Path("/repo/hf-space/data/historical"),
        Path("/repo/data/historical"),
    ]
    data_dir = next((d for d in data_dirs if d.exists()), None)
    if data_dir is None:
        raise RuntimeError(
            "No historical data found. Copy games-*.json to /data/historical/ on the Volume."
        )

    games: list[dict] = []
    for f in sorted(data_dir.glob("games-*.json")):
        raw = json.loads(f.read_text())
        if isinstance(raw, list):
            games.extend(raw)
        elif isinstance(raw, dict) and "games" in raw:
            games.extend(raw["games"])
    print(f"Games loaded: {len(games)}")

    from features.engine import NBAFeatureEngine

    engine = NBAFeatureEngine()
    X_raw, y_raw, feature_names = engine.build(games)

    X = np.nan_to_num(np.array(X_raw, dtype=np.float32))
    y = np.array(y_raw, dtype=np.int32)

    # Drop zero-variance features
    variances = np.var(X, axis=0)
    valid = variances > 1e-10
    X = X[:, valid]
    feature_names = [fn for fn, v in zip(feature_names, valid) if v]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), X=X, y=y, feature_names=np.array(feature_names))
    vol.commit()  # flush to persistent storage

    elapsed = time.time() - t0
    print(f"Cache built: {X.shape} in {elapsed:.0f}s → {CACHE_FILE}")
    return {"cached": False, "shape": tuple(X.shape), "n_features": len(feature_names), "elapsed": elapsed}


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: vol},
    timeout=60,
)
def get_feature_names() -> tuple[list[str], int]:
    """Return (feature_names, n_rows) from the cached feature matrix.

    Runs on a cheap CPU worker — no GPU needed.
    Used by the local entrypoint to build/restore the population without
    loading the full (potentially GBs-sized) numpy array locally.
    """
    import numpy as np

    vol.reload()
    cached = np.load(CACHE_FILE, allow_pickle=True)
    names  = cached["feature_names"].tolist()
    n_rows = int(cached["X"].shape[0])
    return names, n_rows


@app.function(
    gpu="T4",          # swap to "A10G" or "A100" for larger runs
    image=image,
    volumes={VOLUME_MOUNT: vol},
    timeout=300,
    retries=0,         # no retries — bad genome should stay penalised, not rerun
)
def evaluate_individual(
    features_indices: list[int],
    model_type: str,
    hp: dict[str, Any],
    n_features_total: int,
    n_folds: int = 2,
) -> float:
    """Evaluate one individual on GPU.

    Parameters
    ----------
    features_indices : sorted list of active feature column indices
    model_type       : one of tabicl / xgboost / lightgbm / catboost / extra_trees
    hp               : hyperparameter dict (n_estimators, max_depth, learning_rate, …)
    n_features_total : total number of columns in feature matrix (for mask reconstruction)
    n_folds          : number of TimeSeriesSplit folds (2 = fast, 3 = thorough)

    Returns
    -------
    float : mean Brier score across folds (lower = better); 0.30 on error/timeout
    """
    import gc
    import signal
    import sys
    from pathlib import Path

    import numpy as np
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import TimeSeriesSplit

    sys.path.insert(0, "/repo/hf-space")

    # ── load feature cache ──
    cache_path = Path(CACHE_FILE)
    if not cache_path.exists():
        raise RuntimeError("Feature cache missing — run build_feature_cache() first.")

    vol.reload()  # ensure latest version
    cached = np.load(str(cache_path), allow_pickle=True)
    X_full = cached["X"]
    y_full = cached["y"]

    # Subsample: last SUBSAMPLE games (more recent = more relevant)
    if X_full.shape[0] > SUBSAMPLE:
        X = X_full[-SUBSAMPLE:]
        y = y_full[-SUBSAMPLE:]
    else:
        X = X_full
        y = y_full

    # ── guard rails ──
    if len(features_indices) < 5:
        return 1.0
    indices = features_indices[:MAX_FEATURES]  # hard cap
    X_sub = X[:, indices].astype(np.float32)

    # ── walk-forward splits with purge gap ──
    tscv = TimeSeriesSplit(n_splits=n_folds)
    splits = [
        (tr[:-PURGE_GAP] if len(tr) > PURGE_GAP + 50 else tr, te)
        for tr, te in tscv.split(X)
    ]

    # ── model factory ──
    import torch

    _xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_model(mtype: str, params: dict):
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import ExtraTreesClassifier

        n_est = min(params.get("n_estimators", 200), 300)
        depth = params.get("max_depth", 6)
        lr    = params.get("learning_rate", 0.05)
        sub   = params.get("subsample", 0.8)
        cst   = params.get("colsample_bytree", 0.8)

        if mtype == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=n_est, max_depth=depth,
                learning_rate=lr, subsample=sub, colsample_bytree=cst,
                tree_method="hist", device=_xgb_device,
                random_state=42, verbosity=0,
                objective="binary:logistic", eval_metric="logloss",
            )
        elif mtype == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=n_est, max_depth=depth,
                learning_rate=lr, subsample=sub, colsample_bytree=cst,
                random_state=42, verbose=-1,
            )
        elif mtype == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=n_est, depth=min(depth, 10),
                learning_rate=lr, random_state=42, verbose=0,
            )
        elif mtype == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=n_est, max_depth=min(depth, 12),
                min_samples_leaf=5, random_state=42, n_jobs=-1,
            )
        elif mtype == "tabicl":
            from tabicl import TabICLClassifier
            return TabICLClassifier()
        else:
            return ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    # ── timeout helper (SIGALRM works on Linux) ──
    class _Timeout(Exception):
        pass

    def _handler(sig, frame):
        raise _Timeout()

    # Timeout pulled from hp dict (set by caller); default 120s
    EVAL_TIMEOUT_SECONDS = int(hp.get("_timeout", 120))

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(EVAL_TIMEOUT_SECONDS)

    try:
        fold_briers: list[float] = []
        for train_idx, test_idx in splits:
            model = make_model(model_type, hp)
            model.fit(X_sub[train_idx], y[train_idx])
            probs = model.predict_proba(X_sub[test_idx])[:, 1]
            fold_briers.append(brier_score_loss(y[test_idx], probs))
            del model

        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return float(np.mean(fold_briers))

    except _Timeout:
        signal.signal(signal.SIGALRM, old_handler)
        gc.collect()
        return 0.30  # timeout penalty

    except Exception as exc:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        print(f"Eval error ({model_type}): {exc}")
        return 0.30


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL HELPERS  (run on the machine executing `modal run`)
# ══════════════════════════════════════════════════════════════════════════════

def _weighted_model_type() -> str:
    types  = list(MODEL_WEIGHTS.keys())
    weights = list(MODEL_WEIGHTS.values())
    return random.choices(types, weights=weights, k=1)[0]


def _random_hp() -> dict:
    return {
        "n_estimators": random.randint(100, 300),
        "max_depth":    random.randint(4, 9),
        "learning_rate": 10 ** random.uniform(-2.0, -0.7),
        "subsample":     random.uniform(0.6, 1.0),
        "colsample_bytree": random.uniform(0.4, 1.0),
    }


class Individual:
    """Genome = binary feature mask + model type + hyperparameters."""

    def __init__(
        self,
        n_features: int,
        features: list[int] | None = None,
        model_type: str | None = None,
        hp: dict | None = None,
    ):
        self.n_features = n_features
        if features is None:
            prob = TARGET_FEATURES / max(n_features, 1)
            self.features = [1 if random.random() < prob else 0 for _ in range(n_features)]
        else:
            self.features = list(features)
        self.model_type = model_type or _weighted_model_type()
        self.hp = hp or _random_hp()
        self.brier: float = 1.0
        self._enforce_cap()

    # ── properties ──

    @property
    def indices(self) -> list[int]:
        return [i for i, v in enumerate(self.features) if v]

    @property
    def n_feat(self) -> int:
        return sum(self.features)

    # ── genome ops ──

    def _enforce_cap(self) -> None:
        idx = self.indices
        if len(idx) > MAX_FEATURES:
            for i in random.sample(idx, len(idx) - MAX_FEATURES):
                self.features[i] = 0

    def mutate(self, rate: float) -> None:
        for i in range(self.n_features):
            if random.random() < rate:
                self.features[i] = 1 - self.features[i]
        if random.random() < 0.25:
            self.hp["n_estimators"] = max(50, min(300, self.hp["n_estimators"] + random.randint(-50, 50)))
        if random.random() < 0.25:
            self.hp["max_depth"] = max(2, min(10, self.hp["max_depth"] + random.randint(-2, 2)))
        if random.random() < 0.25:
            self.hp["learning_rate"] = max(0.001, min(0.3, self.hp["learning_rate"] * 10 ** random.uniform(-0.3, 0.3)))
        if random.random() < 0.08:
            self.model_type = _weighted_model_type()
        self._enforce_cap()
        self.brier = 1.0  # needs re-evaluation

    @staticmethod
    def crossover(p1: "Individual", p2: "Individual") -> "Individual":
        n = p1.n_features
        point = random.randint(1, n - 1)
        child_feat = p1.features[:point] + p2.features[point:]
        child_mt = p1.model_type if random.random() < 0.5 else p2.model_type
        child_hp = {k: p1.hp[k] if random.random() < 0.5 else p2.hp[k] for k in p1.hp}
        return Individual(n, features=child_feat, model_type=child_mt, hp=child_hp)

    # ── serialisation ──

    def to_dict(self, feature_names: list[str]) -> dict:
        return {
            "features": [feature_names[i] for i, v in enumerate(self.features) if v],
            "model_type": self.model_type,
            "hp": self.hp,
            "brier": self.brier,
        }

    @classmethod
    def from_dict(cls, d: dict, feature_names: list[str], n_features: int) -> "Individual":
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feat = [0] * n_features
        for fname in d.get("features", []):
            if isinstance(fname, str) and fname in name_to_idx:
                feat[name_to_idx[fname]] = 1
        return cls(n_features, features=feat, model_type=d.get("model_type"), hp=d.get("hp"))


def _tournament(pop: list[Individual], k: int = 4) -> Individual:
    return min(random.sample(pop, min(k, len(pop))), key=lambda x: x.brier)


def _fetch_island_seeds() -> list[dict]:
    """Pull best individuals from all 6 HF Spaces (best + top5[:2])."""
    import requests

    seeds: list[dict] = []
    print("\nFetching seeds from HF islands…")
    for name, url in ISLANDS.items():
        try:
            resp = requests.get(f"{url}/api/results", timeout=10)
            if resp.status_code != 200:
                print(f"  {name}: HTTP {resp.status_code}")
                continue
            data = resp.json()
            best = data.get("best", {})
            seeds.append({
                "source": name,
                "brier":  best.get("brier", 1.0),
                "features": best.get("selected_features", []),
                "model_type": best.get("model_type", "xgboost"),
            })
            print(
                f"  {name}: brier={best.get('brier', '?'):.5f}  "
                f"model={best.get('model_type', '?')}  "
                f"feat={best.get('n_features', '?')}"
            )
            for i, ind in enumerate(data.get("top5", [])[:2]):
                seeds.append({
                    "source": f"{name}_top{i+1}",
                    "brier":  ind.get("brier", 1.0),
                    "features": ind.get("selected_features", []),
                    "model_type": ind.get("model_type", "xgboost"),
                })
        except Exception as exc:
            print(f"  {name}: {exc}")
    print(f"Seeds collected: {len(seeds)}")
    return seeds


def _save_state(
    path: str,
    gen: int,
    population: list[Individual],
    feature_names: list[str],
    best_brier: float,
    best_info: dict | None,
    mut_rate: float,
) -> None:
    """Save evolution state to local JSON every generation.

    Written locally so the loop can resume after a network hiccup.
    The Volume-side copy (STATE_FILE) is synced every 10 gens via
    _upload_state_to_volume() to avoid hammering Modal's storage API.
    """
    pop_data = [ind.to_dict(feature_names) for ind in population]
    state = {
        "generation": gen + 1,
        "best_brier": best_brier,
        "best_info": best_info,
        "mutation_rate": mut_rate,
        "population": pop_data,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path(path).write_text(json.dumps(state, default=str))


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: vol},
    timeout=60,
)
def upload_state_to_volume(state_json: str, best_json: str) -> None:
    """Write state + best JSON blobs to the Modal Volume for crash recovery.

    Called every 10 gens from the local entrypoint.  Cheap CPU function —
    no GPU needed.  Ensures that if the local machine dies, evolution can
    resume from a recent checkpoint by fetching STATE_FILE from the Volume.
    """
    from pathlib import Path

    Path(STATE_FILE).write_text(state_json)
    if best_json:
        Path(BEST_FILE).write_text(best_json)
    vol.commit()
    print(f"Volume checkpoint saved (gen embedded in state JSON)")


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: vol},
    timeout=30,
)
def download_state_from_volume() -> tuple[str, str]:
    """Return (state_json, best_json) from Volume for cross-machine resume.

    Returns empty strings if files don't exist yet.
    """
    from pathlib import Path

    vol.reload()
    state_json = Path(STATE_FILE).read_text() if Path(STATE_FILE).exists() else ""
    best_json  = Path(BEST_FILE).read_text()  if Path(BEST_FILE).exists()  else ""
    return state_json, best_json


@app.local_entrypoint()
def main(
    resume: bool = False,
    volume_resume: bool = False,
    gens: int = 0,
    platform: str = "modal_t4",
    rebuild_cache: bool = False,
) -> None:
    """Main evolution loop — runs locally, dispatches GPU evals to Modal.

    Arguments
    ---------
    --resume         : resume from local evolution_state_modal.json
    --volume-resume  : pull state from Modal Volume (use on a new machine)
    --gens N         : override total generation count
    --platform STR   : one of modal_t4, modal_a10g, colab_free, colab_pro
    --rebuild-cache  : force rebuild feature cache even if it exists
    """
    import numpy as np

    cfg = PLATFORM_CONFIGS.get(platform, PLATFORM_CONFIGS["modal_t4"])
    POP_SIZE    = cfg["POP"]
    N_FOLDS     = cfg["FOLDS"]
    TOTAL_GENS  = gens if gens > 0 else cfg["GENS"]
    ELITE_SIZE  = cfg["ELITE"]
    EVAL_TIMEOUT = cfg["TIMEOUT"]

    MUTATION_RATE = 0.10

    print("=" * 70)
    print("  NBA QUANT AI — Modal GPU Evolution")
    print(f"  Platform={platform}  Pop={POP_SIZE}  Folds={N_FOLDS}  Gens={TOTAL_GENS}")
    print("  ATR to beat: 0.21837 (S13 CatBoost gen 815)")
    print("=" * 70)

    # ── 1. Ensure feature cache exists on Volume ──────────────────────────
    if rebuild_cache:
        print("Rebuilding feature cache on GPU worker…")
        result = build_feature_cache.remote()
        print(f"Cache build result: {result}")
    else:
        print("Checking feature cache…")
        result = build_feature_cache.remote()
        print(f"Cache status: {result}")

    # ── 2. Load feature metadata locally (names only, not the full matrix) ──
    # We need feature_names locally to serialise/deserialise individuals.
    # The actual X,y are loaded inside evaluate_individual on the GPU worker.
    feature_names, n_rows = get_feature_names.remote()
    n_features = len(feature_names)
    print(f"Feature space: {n_features} features, {n_rows} games in cache")

    # ── 3. Fetch island seeds ─────────────────────────────────────────────
    island_seeds = _fetch_island_seeds()

    # ── 4. Build / restore population ────────────────────────────────────
    population: list[Individual] = []
    best_ever_brier: float = 1.0
    best_ever_info: dict | None = None
    start_gen: int = 0
    mut_rate: float = MUTATION_RATE

    # Try to load saved state
    local_state = Path("evolution_state_modal.json")

    # Pull from Volume when running on a new machine
    if volume_resume and not local_state.exists():
        print("Pulling state from Modal Volume…")
        state_json, best_json = download_state_from_volume.remote()
        if state_json:
            local_state.write_text(state_json)
            print("Volume state downloaded to local disk.")
        if best_json:
            Path("best_gpu_features.json").write_text(best_json)

    if (resume or volume_resume) and local_state.exists():
        try:
            state = json.loads(local_state.read_text())
            start_gen = state.get("generation", 0)
            best_ever_brier = state.get("best_brier", 1.0)
            best_ever_info  = state.get("best_info")
            mut_rate = state.get("mutation_rate", MUTATION_RATE)
            for saved in state.get("population", []):
                ind = Individual.from_dict(saved, feature_names, n_features)
                ind.brier = saved.get("brier", 1.0)
                population.append(ind)
            print(f"RESUMED: gen={start_gen}, best={best_ever_brier:.5f}, pop={len(population)}")
        except Exception as exc:
            print(f"State load failed ({exc}), starting fresh")
            population = []

    # Seed from islands if population is thin
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    if len(population) < POP_SIZE:
        for seed in island_seeds:
            if len(population) >= POP_SIZE:
                break
            feat = [0] * n_features
            for fname in seed.get("features", []):
                if isinstance(fname, str) and fname in name_to_idx:
                    feat[name_to_idx[fname]] = 1
            if sum(feat) < 15:
                prob = TARGET_FEATURES / max(n_features, 1)
                feat = [1 if random.random() < prob else 0 for _ in range(n_features)]
            # TabICL variant — the GPU advantage
            population.append(Individual(n_features, features=feat, model_type="tabicl"))
            if len(population) < POP_SIZE:
                population.append(
                    Individual(n_features, features=list(feat), model_type=seed.get("model_type", "xgboost"))
                )

        while len(population) < POP_SIZE:
            population.append(Individual(n_features))
        population = population[:POP_SIZE]

    mt_counts = Counter(ind.model_type for ind in population)
    print(f"Population ready: {len(population)} | Models: {dict(mt_counts)}")

    # ── 5. Evolution loop ─────────────────────────────────────────────────
    session_start = time.time()
    gens_this_session = 0

    for gen in range(start_gen, TOTAL_GENS):
        gen_start = time.time()

        # Identify individuals that need evaluation
        to_eval = [ind for ind in population if ind.brier >= 0.99]

        if to_eval:
            # ── Dispatch all evals to Modal GPU workers in parallel ──
            print(f"Gen {gen + 1}/{TOTAL_GENS}: dispatching {len(to_eval)} evals to GPU…", end="", flush=True)

            # Prepare serialisable args for each individual
            eval_args = [
                (
                    ind.indices,
                    ind.model_type,
                    {**ind.hp, "_timeout": EVAL_TIMEOUT},
                    n_features,
                    N_FOLDS,
                )
                for ind in to_eval
            ]

            # starmap → returns list[float] in same order
            brier_scores = list(
                evaluate_individual.starmap(eval_args)
            )

            for ind, score in zip(to_eval, brier_scores):
                ind.brier = score
        else:
            print(f"Gen {gen + 1}/{TOTAL_GENS}: all pre-evaluated", end="", flush=True)

        # Sort population (best first)
        population.sort(key=lambda x: x.brier)

        # Track best ever
        gen_best = population[0]
        improved = gen_best.brier < best_ever_brier
        if improved:
            best_ever_brier = gen_best.brier
            best_ever_info = {
                "brier":      gen_best.brier,
                "model_type": gen_best.model_type,
                "n_features": gen_best.n_feat,
                "generation": gen + 1,
                "features":   [feature_names[i] for i, v in enumerate(gen_best.features) if v],
                "hp":         gen_best.hp,
                "platform":   platform,
                "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            # Save best result immediately to local disk
            Path("best_gpu_features.json").write_text(
                json.dumps(best_ever_info, indent=2, default=str)
            )

        gen_dur = time.time() - gen_start
        elapsed = (time.time() - session_start) / 60
        gens_this_session += 1
        rate = gens_this_session / max(elapsed, 0.1) * 60

        marker = "  *** NEW BEST ***" if improved else ""
        print(
            f"  best={gen_best.brier:.5f} ({gen_best.model_type}, {gen_best.n_feat}f)"
            f" | ever={best_ever_brier:.5f} | {len(to_eval)} evals {gen_dur:.0f}s"
            f" | {elapsed:.0f}min {rate:.0f}g/h{marker}"
        )

        # Detailed log every 10 gens
        if (gen + 1) % 10 == 0:
            mt = Counter(ind.model_type for ind in population)
            top5 = [(f"{ind.brier:.5f}", ind.model_type, ind.n_feat) for ind in population[:5]]
            print(f"  Models: {dict(mt)} | Top5: {top5}")

        # Save state locally every gen (cheap, survives crashes)
        _save_state(
            str(local_state), gen, population, feature_names,
            best_ever_brier, best_ever_info, mut_rate,
        )

        # Sync to Modal Volume every 10 gens (async — fire and forget)
        if (gen + 1) % 10 == 0:
            state_json = local_state.read_text()
            best_json = (
                json.dumps(best_ever_info, indent=2, default=str)
                if best_ever_info else ""
            )
            upload_state_to_volume.spawn(state_json, best_json)

        # ── Selection + Reproduction ──
        elite = population[:ELITE_SIZE]
        children: list[Individual] = []

        # Elites carry over unchanged
        for e in elite:
            c = Individual(n_features, features=list(e.features), model_type=e.model_type, hp=dict(e.hp))
            c.brier = e.brier
            children.append(c)

        # Fill rest via crossover / mutation
        while len(children) < POP_SIZE:
            if random.random() < CROSSOVER_RATE:
                child = Individual.crossover(_tournament(population), _tournament(population))
            else:
                p = _tournament(population)
                child = Individual(n_features, features=list(p.features), model_type=p.model_type, hp=dict(p.hp))
            child.mutate(mut_rate)
            # Force TabICL representation ~40% of new children (GPU advantage)
            if random.random() < 0.40:
                child.model_type = "tabicl"
            children.append(child)

        population = children[:POP_SIZE]
        # Adaptive mutation — decay with floor cap (mirrors HF island fix)
        mut_rate = max(MUT_FLOOR, min(0.15, mut_rate * MUT_DECAY))

    # ── 6. Final summary ──────────────────────────────────────────────────
    _save_state(
        str(local_state), TOTAL_GENS - 1, population, feature_names,
        best_ever_brier, best_ever_info, mut_rate,
    )

    total_time = (time.time() - session_start) / 60
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)

    if best_ever_info:
        delta = best_ever_info["brier"] - 0.21837
        print(f"  Best Brier:   {best_ever_info['brier']:.5f}")
        print(f"  Model:        {best_ever_info['model_type']}")
        print(f"  Features:     {best_ever_info['n_features']}")
        print(f"  Generation:   {best_ever_info['generation']}")
        print(f"  Session gens: {gens_this_session}")
        print(f"  Session time: {total_time:.0f} min")
        print(f"\n  ATR:          0.21837 (S13 CatBoost)")
        print(f"  Delta vs ATR: {delta:+.5f}  {'NEW RECORD!' if delta < 0 else ''}")
        print(f"\n  Best features saved → best_gpu_features.json")
        print(f"  Inject into HF islands via POST /api/config on S10–S15")

    population.sort(key=lambda x: x.brier)
    print("\n  Top 10:")
    for i, ind in enumerate(population[:10]):
        print(f"    #{i+1}: brier={ind.brier:.5f} | {ind.model_type} | {ind.n_feat}f")
