# %% [markdown]
# # NBA Quant AI — Genetic Evolution with GPU (Google Colab)
#
# **One-click notebook**: Clone repo → install deps → run genetic evolution on T4 GPU.
#
# **Instructions**:
# 1. Open in Colab: Runtime → Change runtime type → GPU (T4 free)
# 2. Add Colab Secrets (key icon left sidebar):
#    - `DATABASE_URL`: your Supabase postgres URL
# 3. Run all cells — evolution starts automatically

# %% [markdown]
# ## 1. Clone Repo & Install

# %%
import subprocess, sys, os

if not os.path.exists("/content/nomos-nba-agent"):
    subprocess.run(["git", "clone", "https://github.com/LBJLincoln/nomos-nba-agent.git",
                    "/content/nomos-nba-agent"], check=True)
else:
    subprocess.run(["git", "-C", "/content/nomos-nba-agent", "pull"], check=True)

os.chdir("/content/nomos-nba-agent")
sys.path.insert(0, "/content/nomos-nba-agent")

_pkgs = [
    "psycopg2-binary", "xgboost", "lightgbm", "catboost",
    "pytorch-tabnet", "scikit-learn>=1.3", "pandas", "numpy",
]
for pkg in _pkgs:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import torch
print(f"PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")

# %% [markdown]
# ## 2. Load Secrets

# %%
try:
    from google.colab import userdata
    os.environ.setdefault("DATABASE_URL", userdata.get("DATABASE_URL"))
    print("DATABASE_URL loaded from Colab secrets")
except Exception:
    print("Set DATABASE_URL manually: os.environ['DATABASE_URL'] = 'postgresql://...'")

# %% [markdown]
# ## 3. Run Genetic Evolution (GPU)
#
# - **Pop: 100** (5 islands × 20) — fast iterations
# - **10 gens/cycle** × 50 cycles = **500 generations total**
# - All model types: XGBoost, LightGBM, CatBoost, TabNet, MLP, LSTM
# - Walk-forward 3-fold, Supabase logging

# %%
from evolution.genetic_loop_v3 import run_continuous

# GPU evolution: 100 pop, 10 gens/cycle, 50 cycles
run_continuous(
    generations_per_cycle=10,
    total_cycles=50,
    pop_size=100,
    target_features=100,
    n_splits=3,
    cool_down=10,
)

# %% [markdown]
# ## 4. Check Results

# %%
import json
from pathlib import Path

f = Path("/content/nomos-nba-agent/data/results/evolution-latest.json")
if f.exists():
    r = json.loads(f.read_text())
    b = r.get("best", {})
    print(f"Best Brier: {b.get('brier')} | ROI: {b.get('roi')} | "
          f"Sharpe: {b.get('sharpe')} | Features: {b.get('n_features')} | "
          f"Model: {b.get('model_type')} | Gen: {r.get('generation')}")
else:
    print("No results yet.")
