"""
NBA Quant AI — Conformal Prediction Calibration
=================================================
Split Conformal, Adaptive Conformal Inference (ACI), and Mondrian
Conformal Prediction for NBA game probability calibration.

Reference:
  - Vovk et al. (2005) "Algorithmic Learning in a Random World"
  - Gibbs & Candes (2021) "Adaptive Conformal Inference Under Distribution Shift"
  - Vovk (2012) "Conditional Validity of Inductive Conformal Predictors"
"""

from calibration.conformal import (
    conformal_calibrate,
    ConformalCalibrator,
    SplitConformalCalibrator,
    AdaptiveConformalCalibrator,
    MondrianConformalCalibrator,
    EnsembleConformalCalibrator,
    create_conformal_experiment,
)

__all__ = [
    "conformal_calibrate",
    "ConformalCalibrator",
    "SplitConformalCalibrator",
    "AdaptiveConformalCalibrator",
    "MondrianConformalCalibrator",
    "EnsembleConformalCalibrator",
    "create_conformal_experiment",
]
