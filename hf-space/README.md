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

- **9+ ML models** trained continuously with Optuna hyperparameter search
- **8 seasons** of NBA data (9,551+ games, 2018-2026)
- **75 features** including travel fatigue, Kaunitz odds gap, clutch performance
- **Walk-forward backtesting** with Kelly criterion sizing
- **Isotonic calibration** improving Brier scores by 4-12%
