---
title: NOMOS NBA Quant AI
emoji: 🏀
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: true
---

# NOMOS NBA Quant AI — Continuous Training

Always-running agentic loop for NBA quantitative prediction models.

- **5 tree-based models** evolved via NSGA-II genetic algorithm (CPU-optimized)
- **8 seasons** of NBA data (9,551+ games, 2018-2026)
- **Up to 200 features** (island-specific: 55-80) from v3.0-37cat engine with MOVDA
- **Walk-forward backtesting** with Kelly criterion sizing
- **Feature engine**: v3.0 + Cat36 EWMA + Cat37 MOVDA (deployed 2026-03-25)
- **MAX_FEATURES=200** hard cap, mutation capped at 0.15, xgboost_brier fixed
