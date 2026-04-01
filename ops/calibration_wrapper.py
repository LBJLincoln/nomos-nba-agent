#!/usr/bin/env python3
"""
Calibration Wrapper — Post-hoc isotonic/sigmoid calibration for GA-selected models
Based on: MDPI Information 2026 "Uncertainty-Aware NBA Forecasting"
Added by Nomos42 Brain 2026-04-01
"""

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import numpy as np


def calibrate_model(base_model, X_val, y_val, model_type: str = "tree") -> object:
    """
    Wrap a fitted model with isotonic or sigmoid calibration.
    Returns the calibrated model with lower Brier score.
    
    Args:
        base_model: Fitted sklearn-compatible model
        X_val: Validation features
        y_val: Validation labels  
        model_type: "tree" for isotonic, "linear" for sigmoid, "auto" to try both
    
    Returns:
        Best calibrated model (or original if calibration doesn't improve)
    """
    base_brier = brier_score_loss(y_val, base_model.predict_proba(X_val)[:, 1])
    
    results = {"original": (base_model, base_brier)}
    
    methods = ["isotonic", "sigmoid"] if model_type == "auto" else (
        ["isotonic"] if model_type == "tree" else ["sigmoid"]
    )
    
    for method in methods:
        try:
            cal = CalibratedClassifierCV(base_model, method=method, cv="prefit")
            cal.fit(X_val, y_val)
            cal_brier = brier_score_loss(y_val, cal.predict_proba(X_val)[:, 1])
            results[method] = (cal, cal_brier)
        except Exception as e:
            print(f"[CALIBRATION] {method} failed: {e}")
    
    best_name = min(results, key=lambda k: results[k][1])
    best_model, best_brier = results[best_name]
    
    improvement = base_brier - best_brier
    print(f"[CALIBRATION] Base Brier: {base_brier:.5f} | Best ({best_name}): {best_brier:.5f} | Improvement: {improvement:+.5f}")
    
    return best_model


def compute_ece(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Expected Calibration Error — measures calibration quality.
    Lower is better. Perfect calibration = 0.0
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    
    return ece


def composite_fitness_with_calibration(brier: float, ece: float, sharpe: float) -> float:
    """
    Combined fitness metric for GA evaluation that rewards calibration.
    Lower is better (minimization).
    
    Weights: 70% Brier, 20% ECE, 10% inverse Sharpe penalty
    """
    sharpe_penalty = 1 / (1 + max(sharpe, 0))
    return 0.70 * brier + 0.20 * ece + 0.10 * sharpe_penalty
