# Nomos42 — NBA Quant AI

> Architecture v14 — Claude Code 2026 | Updated: 2026-03-24

## Mission
Build the best NBA prediction AI in the world.
**Current:** Brier 0.2200 (Stacking, 96 features) | **Target:** Brier < 0.20, ROI > 5%, Sharpe > 1.5

## Architecture

```
Cloud Brain (Sonnet 4.6, every 4h)     VM Muscle (cron, every 4h)
├── Monitor S10/S11 via APIs            ├── Crew research cycle (4 agents)
├── Analyze trends + stagnation         ├── Daily predictions
├── Decide: tune/diversify/inject       ├── Push results to git
└── Write health report + push          └── Auto-restart data server

S10 (HF Space) — 24/7 genetic evolution (5 islands × 100, NSGA-II)
S11 (HF Space) — Experiment queue runner (polls Supabase)
```

## Key Files

| File | Role |
|------|------|
| `features/engine.py` | Canonical feature engine v3.0-35cat, 6000+ candidates |
| `hf-space/features/engine.py` | MUST equal root engine (deploy.py checks) |
| `hf-space/app.py` | S10 Gradio app + evolution loop + API endpoints |
| `hf-space/experiment_runner.py` | S11 experiment queue processor |
| `evolution/genetic_loop_v3.py` | GA v3 with real engine import |
| `predict_today.py` | Daily predictions (60% S10 evolved + 40% ensemble) |
| `agents/nba_crew.py` | 4-agent CrewAI swarm (Research, Market, Feature, Evolution) |
| `agents/key_rotator.py` | Multi-provider LLM routing (6 providers, 16 keys) |

## S10 Public API (no auth)

| Endpoint | Method | Returns |
|----------|--------|---------|
| `/api/status` | GET | cycle, generation, best_brier, stagnation, pop_size |
| `/api/brier-trend` | GET | Last 50 generations Brier scores |
| `/api/recent-runs` | GET | Recent cycle summaries |
| `/api/run-stats` | GET | Aggregate stats from Supabase |
| `/api/remote-log` | GET | Pending params, commands, injected features |
| `/api/config` | POST | Queue GA parameter changes |
| `/api/command` | POST | diversify, boost_mutation, reset |
| `/api/inject-features` | POST | Force features into GA population |
| `/api/checkpoint` | POST | Save current best model |
| `/api/experiment/submit` | POST | Queue experiment for S11 |
| `/api/predict` | POST | Get evolved model predictions |

## Rules

1. **ZERO ML on VM** — 1 vCPU / 969 MB RAM. ALL training on HF Spaces
2. **Feature engine parity** — `features/engine.py` = `hf-space/features/engine.py` always
3. **1 fix per iteration** — never multiple simultaneous changes
4. **All experiments tagged** with `feature_engine_version` in Supabase
5. **ENGINE_VERSION** = `v3.0-35cat` (6000+ candidates, 35 categories)

## Supabase Tables

| Table | Purpose |
|-------|---------|
| `nba_experiments` | All experiment results + `feature_engine_version` |
| `research_proposals` | Karpathy loop proposals (proposed/testing/rejected/live) |
