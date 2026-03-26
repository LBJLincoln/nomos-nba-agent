---
name: submit-experiment
description: Submit a GPU experiment to Google Colab for neural model training.
---

# Submit GPU Experiment

Submit a GPU experiment to Google Colab for neural model training.

Available model types: ft_transformer, tabnet, node, saint, mc_dropout_rnn
These require GPU (T4+) and cannot run on HF Spaces CPU.

Steps:
1. Check current best in Supabase: `SELECT model_type, brier_score FROM nba_experiments ORDER BY brier_score LIMIT 5`
2. Choose model type and hyperparameters
3. Update `colab/nba_evolution_gpu.ipynb` with experiment config
4. Commit and push — user will open in Colab manually

The Colab notebook uses the full NBAFeatureEngine (6129 features) and logs to Supabase automatically.
