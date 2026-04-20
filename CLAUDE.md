# Nomos42 — NBA Quant AI

> Architecture v15 — Post 2026-04-17 island cull | Updated: 2026-04-20

## Mission
Build the best NBA prediction AI in the world.
**Best:** Brier 0.21514 (Colab TabICL, 186f, iter 129) | Fleet best: 0.22073 (S22 venn_abers_fusion, gen 39, checkpointed 2026-04-19) | **Target:** Brier < 0.20, ROI > 5%, Sharpe > 1.5

## Architecture

```
CLOUD BRAIN (Sonnet 4.6, every 4h at :00)
    ├── Monitor 6 NBA survivors (S13-S22) via public /api/status
    ├── Research via Claude Code subagents (every 4h)
    ├── Decide: tune GA / diversify / inject features / checkpoint
    ├── Act on islands via POST /api/config
    └── Write health-status.json + push
    Trigger: trig_01BS3ixBvt2uKHY9p5EemcgD

VM MUSCLE (cron, every 4h at :30)
    ├── Run predict_today.py (if NBA games)
    ├── Execute brain recommendations
    ├── Push results to git
    └── Auto-restart data server

HF SPACES (6 NBA survivors after 2026-04-17 cull; S10/S11/S12/S16/S19/S20/S21 eliminated)
    S13 Nomos42/nba-evo-4                — catboost          gen 130  brier 0.22749
    S14 Nomos42/nba-evo-5                — lightgbm          gen 554  brier 0.22186
    S15 Nomos42/nba-evo-6                — wide search       gen 127  brier 0.22418
    S17 LBJLincoln26/nba-evo-s17         — ensemble          gen 78   brier 0.22340
    S18 TESTforge42/nba-evo-s18          — catboost_spec     gen 1030 brier 0.22114
    S22 TESTforge42/nba-evo-s22          — venn_abers_fusion gen 39   brier 0.22073  ★ FLEET BEST

GOOGLE COLAB (GPU, on-demand)
    └── colab/nba_evolution_gpu.ipynb    — T4 GPU evolution (neural models)
```

## Key Files

| File | Role |
|------|------|
| `features/engine.py` | Canonical feature engine v3.1-54cat = 54 categories, 7213 raw features |
| `hf-space/features/engine.py` | MUST equal root engine (deploy_island.py checks parity) |
| `hf-space/app.py` | Evolution loop v4 + Gradio UI + API endpoints (all 6 spaces) |
| `hf-space/deploy_island.py` | Deploy any island: `python3 hf-space/deploy_island.py SPACE ROLE TOKEN` |
| `predict_today.py` | Daily predictions (rank-based fusion across all 6 islands) |
| `calibration/isotonic_calibrator.py` | Probability calibration (stub — fit on HF Space) |

## Fleet Public API (no auth — any surviving island)

| Endpoint | Method | Returns |
|----------|--------|---------|
| `/api/status` | GET | generation, best_brier, stagnation, pop_size, gpu |
| `/api/brier-trend` | GET | Last 50 generations Brier scores |
| `/api/recent-runs` | GET | Recent cycle summaries |
| `/api/run-stats` | GET | Aggregate stats from Supabase |
| `/api/remote-log` | GET | Pending params, commands, injected features |
| `/api/config` | POST | Queue GA parameter changes |
| `/api/command` | POST | diversify, boost_mutation, reset |
| `/api/inject-features` | POST | Force features into GA population |
| `/api/checkpoint` | POST | Save current best model |
| `/api/predict` | POST | Get evolved model predictions |

## Rules

1. **ZERO ML on VM** — 1 vCPU / 969 MB RAM. ALL training on HF Spaces or Colab
2. **Feature engine parity** — `features/engine.py` = `hf-space/features/engine.py` always
3. **1 fix per iteration** — never multiple simultaneous changes
4. **All experiments tagged** with `feature_engine_version` in Supabase
5. **ENGINE_VERSION** = `v3.1-66cat` (verified 2026-04-20, see `features/engine.py` header)
6. **MAX_FEATURES=200** — hard cap enforced in all spaces
7. **Mutation cap** — adaptive mutation capped at 0.15
8. **CPU-only** — no neural models on CPU islands, stacking removed

## Supabase Tables

| Table | Purpose |
|-------|---------|
| `nba_experiments` | All experiment results + `feature_engine_version` |
| `research_proposals` | Karpathy loop proposals (proposed/testing/rejected/live) |


## Forge v19 — 3 Layers × 8 Departments (2026-04-03T20:12:19Z)

```
L1 STRATEGIC:  Claude Code CLI + User (vision, milestones, decisions)
L2 APPLICATION: D1 Research | D2 Engineering | D3 Evolution | D4 Product | D5 Business | D6 Evaluation
L3 LOGISTICS:   D7 Infra | D8 Finance
```

Each department runs a Karpathy autoresearch loop:
- SCAN → PROPOSE → EXECUTE (5-min) → EVALUATE → KEEP/REVERT
- Council state: data/departments/council-<dept>.json
- Metrics log: data/departments/<dept>/metrics.jsonl
- Runner: scripts/councils/department-council.sh <dept>

Shared infra: VM (control tower) + Laptop (local models) + HF Spaces (3 accounts) + GPU burst

