"""
NBA Quant AI — Neural Network Models
=====================================
SOTA 2025-2026 neural architectures for NBA game prediction.

All models conform to the same interface:
  - fit(X_train, y_train, X_val, y_val)
  - predict_proba(X)
  - get_params()
  - save(path) / load(path)

Runs on HF Spaces (16 GB RAM, CPU-only PyTorch).
"""

from .neural_models import (
    LSTMSequenceModel,
    TransformerAttentionModel,
    TabNetModel,
    FTTransformerModel,
    DeepEnsemble,
    ConformalPredictionWrapper,
    AutoGluonEnsemble,
    NEURAL_MODEL_REGISTRY,
)

__all__ = [
    "LSTMSequenceModel",
    "TransformerAttentionModel",
    "TabNetModel",
    "FTTransformerModel",
    "DeepEnsemble",
    "ConformalPredictionWrapper",
    "AutoGluonEnsemble",
    "NEURAL_MODEL_REGISTRY",
]
