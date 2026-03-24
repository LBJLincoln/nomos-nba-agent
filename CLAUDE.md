# NBA Quant AI Agent

You are an autonomous agent improving the NBA Quant AI prediction model.
Do NOT modify this file to change the agent name — this file is agent-agnostic by design.
You have Claude Code Max OAuth credentials (no API key needed).
You are working on the REAL git repository. Your changes will be committed and pushed.

## MISSION
Build the best NBA prediction AI in the world. Beat the best hedge funds.
Current best: Brier 0.2205 | Target: Brier < 0.20, ROI > 5%, Sharpe > 1.5

## KEY FILES
- features/engine.py — 580+ feature candidates, 94 selected
- evolution/loop.py — Genetic algorithm (population 50+, multi-objective fitness)
- models/ — XGBoost, LightGBM, CatBoost, Stacking
- colab/nba_gpu_runner.ipynb — GPU training (MLP, LSTM, FT-Transformer, etc.)
- predict_today.py — Daily prediction pipeline

## RULES
1. NEVER run ML training here — submit experiments to Supabase for GPU runners (Colab)
2. Keep changes minimal and focused — 1 fix per commit
3. ALWAYS commit and push when done
4. Read existing code BEFORE modifying
5. Run tests if available
6. Do NOT create README.md or documentation files

## ENDPOINTS
- S10 Evolution: https://lbjlincoln-nomos-nba-quant.hf.space/api/status
- S11 Parallel: https://lbjlincoln-nomos-nba-quant-2.hf.space/api/status
- ESPN Scores: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard

## SUPABASE (for experiment submission)
Use the nba_experiments table. Insert with status='pending', target_space='gpu'.
DATABASE_URL is in the environment.
