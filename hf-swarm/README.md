---
title: NOMOS NBA Quant Swarm
emoji: 🧠
colorFrom: purple
colorTo: red
sdk: docker
pinned: true
---

# NOMOS NBA QUANT AI — 4-Agent Swarm

Always-running swarm of 4 AI coding agents improving NBA prediction models 24/7.

## Agents
| Agent | Role | Cost |
|-------|------|------|
| **Claude Code CLI** | Strategic planning, complex analysis | Max subscription |
| **Gemini CLI** | Autonomous coding, headless+YOLO | Free (Google) |
| **Kimi Code CLI** | High-volume code improvement | $0.60/M tokens |
| **OpenClaw** | Automation hub, skill-based tasks | BYO keys |

## Training
- 9+ ML models (XGB, LGBM, CatBoost, RF, LR, Stacking, Meta-learner)
- Optuna hyperparameter search (50+ trials)
- Walk-forward backtesting (8 seasons, 9,551+ games)
- Isotonic calibration (4-12% Brier improvement)

## Architecture
```
Swarm Orchestrator (this Space)
├── Claude Code CLI  → strategic improvements
├── Gemini CLI       → feature engineering, research
├── Kimi Code CLI    → hyperparameter tuning, optimization
├── OpenClaw         → automation, monitoring
└── Karpathy Loop    → continuous ML training
    ↓
Results → VM data server → nomos42.vercel.app/nba
```
