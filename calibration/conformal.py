#!/usr/bin/env python3
"""
NBA Quant AI — Conformal Prediction Calibration
=================================================
Production-ready conformal prediction for NBA binary classification.

Implements:
  1. Split Conformal Prediction (Vovk et al., 2005)
     - Distribution-free coverage guarantee
     - Non-conformity scores from base model probabilities
  2. Adaptive Conformal Inference (ACI) (Gibbs & Candes, 2021)
     - Time-varying coverage for non-stationary NBA data
     - Online learning rate gamma to adapt alpha over time
  3. Mondrian Conformal Prediction (Vovk, 2012)
     - Group-conditional calibration (home/away, rest, ELO tier, etc.)
     - Each "cell" gets its own calibration set
  4. Ensemble Conformal — combines all three methods

Integration:
  - Works with walk-forward backtesting (temporal split: cal → test)
  - ConformalCalibrator wraps any base model's predict_proba output
  - Supabase experiment creation for automated pipeline

THIS MODULE RUNS ON GPU RUNNERS (Kaggle/Colab/HF Space) — not on VM.
"""

import json
import time
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple, Union
from collections import defaultdict
import warnings


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _safe_clip(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip probabilities to (eps, 1-eps) for numerical stability."""
    return np.clip(probs, eps, 1.0 - eps)


def _quantile_with_correction(scores: np.ndarray, alpha: float) -> float:
    """
    Compute the conformal quantile with finite-sample correction.

    For n calibration scores, the conformal quantile is the
    ceil((n+1)*(1-alpha))/n-th empirical quantile. This guarantees
    marginal coverage >= 1-alpha.

    Args:
        scores: Non-conformity scores from calibration set.
        alpha: Miscoverage rate (e.g. 0.1 for 90% coverage).

    Returns:
        Conformal quantile threshold.
    """
    n = len(scores)
    if n == 0:
        return 0.5  # fallback: no information
    # Finite-sample corrected quantile level (Vovk et al.)
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level))


def _nonconformity_score_hinge(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Hinge-based non-conformity score for binary classification.

    score(x, y) = 1 - f(x)[y]

    Where f(x)[y] is the predicted probability for the true class.
    Lower probability for the true class = higher non-conformity.
    """
    prob_true_class = np.where(y == 1, prob, 1.0 - prob)
    return 1.0 - prob_true_class


def _nonconformity_score_margin(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Margin-based non-conformity score.

    score(x, y) = f(x)[1-y] - f(x)[y]

    The predicted probability of the wrong class minus the true class.
    Negative when model is correct and confident, positive when wrong.
    """
    prob_true = np.where(y == 1, prob, 1.0 - prob)
    prob_false = 1.0 - prob_true
    return prob_false - prob_true


def _nonconformity_score_log(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Negative log-likelihood non-conformity score.

    score(x, y) = -log(f(x)[y])

    More sensitive to calibration errors near 0 and 1.
    """
    prob = _safe_clip(prob)
    prob_true_class = np.where(y == 1, prob, 1.0 - prob)
    return -np.log(prob_true_class)


SCORE_FUNCTIONS = {
    "hinge": _nonconformity_score_hinge,
    "margin": _nonconformity_score_margin,
    "log": _nonconformity_score_log,
}


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score (lower = better, 0 = perfect)."""
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE).

    Partitions predictions into equal-width bins and measures the
    weighted average absolute difference between predicted and
    observed frequency.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_pred = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        ece += (count / total) * abs(avg_pred - avg_true)
    return float(ece)


# ═══════════════════════════════════════════════════════════════════
# 1. SPLIT CONFORMAL PREDICTION
# ═══════════════════════════════════════════════════════════════════

class SplitConformalCalibrator:
    """
    Split Conformal Prediction for binary classification (Vovk et al., 2005).

    Uses a held-out calibration set to compute non-conformity scores,
    then adjusts test predictions to ensure coverage guarantee.

    For NBA: Given model probabilities P(home_win), produce calibrated
    prediction sets {0}, {1}, or {0,1} with coverage >= 1-alpha.
    For point predictions: maps raw probs through the empirical CDF
    of calibration scores to produce calibrated probabilities.

    Properties:
      - Distribution-free: no parametric assumptions
      - Finite-sample valid: coverage guarantee holds for any n
      - Model-agnostic: wraps any base classifier
    """

    def __init__(
        self,
        alpha: float = 0.1,
        score_fn: str = "hinge",
    ):
        """
        Args:
            alpha: Target miscoverage rate. Default 0.1 = 90% coverage.
            score_fn: Non-conformity score function. One of "hinge", "margin", "log".
        """
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.score_fn_name = score_fn
        self._score_fn = SCORE_FUNCTIONS[score_fn]
        self._cal_scores: Optional[np.ndarray] = None
        self._quantile: Optional[float] = None
        self._cal_probs: Optional[np.ndarray] = None
        self._cal_y: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, cal_probs: np.ndarray, cal_y: np.ndarray) -> "SplitConformalCalibrator":
        """
        Fit on calibration data.

        Args:
            cal_probs: Model's P(Y=1) on calibration set. Shape (n_cal,).
            cal_y: True labels (0 or 1) on calibration set. Shape (n_cal,).
        """
        cal_probs = np.asarray(cal_probs, dtype=np.float64).ravel()
        cal_y = np.asarray(cal_y, dtype=np.int32).ravel()
        if len(cal_probs) != len(cal_y):
            raise ValueError(f"Length mismatch: probs={len(cal_probs)}, y={len(cal_y)}")
        if len(cal_probs) < 10:
            warnings.warn(f"Very small calibration set ({len(cal_probs)}). Results may be unreliable.")

        self._cal_probs = _safe_clip(cal_probs)
        self._cal_y = cal_y
        self._cal_scores = self._score_fn(self._cal_probs, self._cal_y)
        self._quantile = _quantile_with_correction(self._cal_scores, self.alpha)
        self._fitted = True
        return self

    def predict_sets(self, test_probs: np.ndarray) -> List[List[int]]:
        """
        Produce prediction sets with coverage >= 1-alpha.

        For each test point, include class y in the set if the
        non-conformity score s(x, y) <= quantile.

        Args:
            test_probs: Model's P(Y=1) on test set. Shape (n_test,).

        Returns:
            List of prediction sets, e.g. [[1], [0,1], [0], ...].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_sets()")
        test_probs = _safe_clip(np.asarray(test_probs, dtype=np.float64).ravel())
        sets = []
        for p in test_probs:
            s = set()
            # Check if y=1 is in the set
            score_1 = 1.0 - p  # hinge score for y=1
            if self.score_fn_name == "margin":
                score_1 = (1.0 - p) - p
            elif self.score_fn_name == "log":
                score_1 = -np.log(max(p, 1e-10))
            if score_1 <= self._quantile:
                s.add(1)
            # Check if y=0 is in the set
            score_0 = 1.0 - (1.0 - p)  # = p; hinge score for y=0
            if self.score_fn_name == "margin":
                score_0 = p - (1.0 - p)
            elif self.score_fn_name == "log":
                score_0 = -np.log(max(1.0 - p, 1e-10))
            if score_0 <= self._quantile:
                s.add(0)
            sets.append(sorted(s) if s else [1 if p >= 0.5 else 0])
        return sets

    def calibrate(self, test_probs: np.ndarray) -> np.ndarray:
        """
        Calibrate test probabilities using the empirical CDF of
        non-conformity scores from the calibration set.

        Maps raw model prob p to: calibrated_p = fraction of
        calibration samples whose score for y=1 exceeds the
        test point's score for y=1.

        This produces well-calibrated point predictions.

        Args:
            test_probs: Model's P(Y=1) on test set. Shape (n_test,).

        Returns:
            Calibrated P(Y=1). Shape (n_test,).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before calibrate()")
        test_probs = _safe_clip(np.asarray(test_probs, dtype=np.float64).ravel())
        n_cal = len(self._cal_scores)

        # For each test prob, compute score as if y=1, then see where it
        # ranks among cal scores. The calibrated prob is the conformal p-value.
        test_scores_for_1 = self._score_fn(test_probs, np.ones(len(test_probs), dtype=np.int32))

        calibrated = np.zeros(len(test_probs), dtype=np.float64)
        for i, ts in enumerate(test_scores_for_1):
            # p-value: fraction of cal scores >= test score
            # (plus 1 for the test point itself, Vovk's smoothed p-value)
            count_geq = np.sum(self._cal_scores >= ts) + 1
            calibrated[i] = count_geq / (n_cal + 1)

        return _safe_clip(calibrated)

    def diagnostics(self, test_probs: np.ndarray, test_y: np.ndarray) -> Dict[str, Any]:
        """Compute diagnostic metrics for the conformal calibrator."""
        cal_probs = self.calibrate(test_probs)
        test_y = np.asarray(test_y, dtype=np.int32).ravel()
        pred_sets = self.predict_sets(test_probs)

        # Coverage
        coverage = np.mean([test_y[i] in s for i, s in enumerate(pred_sets)])
        # Average set size
        avg_set_size = np.mean([len(s) for s in pred_sets])
        # Singleton rate (informative predictions)
        singleton_rate = np.mean([len(s) == 1 for s in pred_sets])

        return {
            "coverage": float(coverage),
            "avg_set_size": float(avg_set_size),
            "singleton_rate": float(singleton_rate),
            "brier_raw": brier_score(test_y, _safe_clip(np.asarray(test_probs).ravel())),
            "brier_calibrated": brier_score(test_y, cal_probs),
            "ece_raw": expected_calibration_error(test_y, _safe_clip(np.asarray(test_probs).ravel())),
            "ece_calibrated": expected_calibration_error(test_y, cal_probs),
            "n_calibration": len(self._cal_scores) if self._cal_scores is not None else 0,
            "n_test": len(test_y),
            "quantile": float(self._quantile) if self._quantile is not None else None,
            "alpha": self.alpha,
            "score_fn": self.score_fn_name,
        }


# ═══════════════════════════════════════════════════════════════════
# 2. ADAPTIVE CONFORMAL INFERENCE (ACI)
# ═══════════════════════════════════════════════════════════════════

class AdaptiveConformalCalibrator:
    """
    Adaptive Conformal Inference (Gibbs & Candes, 2021).

    Standard conformal prediction assumes exchangeability, but NBA data
    has distribution shift (teams improve/degrade, injuries, trades,
    schedule effects). ACI adapts the coverage level alpha_t over time:

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

    where err_t = 1 if y_t not in prediction set at time t, 0 otherwise.

    This ensures long-run average coverage converges to 1-alpha even
    under distribution shift.

    NBA application: Early-season models differ from late-season.
    ACI tightens/loosens bands as the data distribution changes.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.01,
        score_fn: str = "hinge",
        window: int = 500,
    ):
        """
        Args:
            alpha: Target miscoverage rate.
            gamma: Learning rate for alpha adaptation. Higher = faster adaptation.
                   Typical range: 0.005 to 0.05.
            score_fn: Non-conformity score function.
            window: Rolling window of recent calibration scores to maintain.
        """
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        self.alpha_target = alpha
        self.alpha_t = alpha  # current adaptive alpha
        self.gamma = gamma
        self.score_fn_name = score_fn
        self._score_fn = SCORE_FUNCTIONS[score_fn]
        self.window = window

        # Rolling calibration state
        self._scores_buffer: List[float] = []
        self._alpha_history: List[float] = [alpha]
        self._coverage_history: List[int] = []
        self._fitted = False

    def fit(self, cal_probs: np.ndarray, cal_y: np.ndarray) -> "AdaptiveConformalCalibrator":
        """
        Initialize with a calibration set.

        Args:
            cal_probs: Model's P(Y=1) on calibration data. Shape (n,).
            cal_y: True labels. Shape (n,).
        """
        cal_probs = _safe_clip(np.asarray(cal_probs, dtype=np.float64).ravel())
        cal_y = np.asarray(cal_y, dtype=np.int32).ravel()
        scores = self._score_fn(cal_probs, cal_y)
        self._scores_buffer = list(scores[-self.window:])
        self._fitted = True
        return self

    def update(self, prob: float, y_true: int) -> None:
        """
        Online update after observing one new data point.

        Computes non-conformity score, checks coverage, and adapts alpha.

        Args:
            prob: Model's P(Y=1) for this data point.
            y_true: True label (0 or 1).
        """
        prob = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
        score = float(self._score_fn(np.array([prob]), np.array([y_true]))[0])

        # Check coverage at current alpha
        if len(self._scores_buffer) > 0:
            quantile = _quantile_with_correction(
                np.array(self._scores_buffer), self.alpha_t
            )
            # Was the true label covered?
            covered = int(score <= quantile)
        else:
            covered = 1

        self._coverage_history.append(covered)

        # ACI update rule: alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
        err_t = 1 - covered  # 1 if miscovered, 0 if covered
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha_target - err_t)
        # Clamp to valid range
        self.alpha_t = float(np.clip(self.alpha_t, 0.001, 0.999))
        self._alpha_history.append(self.alpha_t)

        # Add score to buffer (sliding window)
        self._scores_buffer.append(score)
        if len(self._scores_buffer) > self.window:
            self._scores_buffer.pop(0)

    def update_batch(self, probs: np.ndarray, y_true: np.ndarray) -> None:
        """
        Sequential online update for a batch of observations.

        Processes each point sequentially (order matters for ACI).
        """
        probs = np.asarray(probs, dtype=np.float64).ravel()
        y_true = np.asarray(y_true, dtype=np.int32).ravel()
        for p, y in zip(probs, y_true):
            self.update(p, int(y))

    def calibrate(self, test_probs: np.ndarray) -> np.ndarray:
        """
        Calibrate test probabilities using current adaptive state.

        Uses the current scores buffer and adapted alpha to produce
        calibrated probabilities via the conformal p-value approach.

        Args:
            test_probs: Model's P(Y=1). Shape (n,).

        Returns:
            Calibrated P(Y=1). Shape (n,).
        """
        if not self._fitted and len(self._scores_buffer) == 0:
            warnings.warn("ACI not fitted. Returning raw probs.")
            return _safe_clip(np.asarray(test_probs, dtype=np.float64).ravel())

        test_probs = _safe_clip(np.asarray(test_probs, dtype=np.float64).ravel())
        cal_scores = np.array(self._scores_buffer)
        n_cal = len(cal_scores)

        if n_cal == 0:
            return test_probs

        test_scores = self._score_fn(test_probs, np.ones(len(test_probs), dtype=np.int32))
        calibrated = np.zeros(len(test_probs), dtype=np.float64)
        for i, ts in enumerate(test_scores):
            count_geq = np.sum(cal_scores >= ts) + 1
            calibrated[i] = count_geq / (n_cal + 1)

        return _safe_clip(calibrated)

    def predict_sets(self, test_probs: np.ndarray) -> List[List[int]]:
        """Prediction sets using current adaptive alpha."""
        if len(self._scores_buffer) == 0:
            return [[1] if p >= 0.5 else [0]
                    for p in np.asarray(test_probs).ravel()]

        quantile = _quantile_with_correction(
            np.array(self._scores_buffer), self.alpha_t
        )
        test_probs = _safe_clip(np.asarray(test_probs, dtype=np.float64).ravel())

        sets = []
        for p in test_probs:
            s = set()
            score_1 = float(self._score_fn(np.array([p]), np.array([1]))[0])
            score_0 = float(self._score_fn(np.array([p]), np.array([0]))[0])
            if score_1 <= quantile:
                s.add(1)
            if score_0 <= quantile:
                s.add(0)
            sets.append(sorted(s) if s else [1 if p >= 0.5 else 0])
        return sets

    def diagnostics(self, test_probs: np.ndarray, test_y: np.ndarray) -> Dict[str, Any]:
        """Compute diagnostic metrics."""
        cal_probs = self.calibrate(test_probs)
        test_y = np.asarray(test_y, dtype=np.int32).ravel()
        pred_sets = self.predict_sets(test_probs)
        coverage = np.mean([test_y[i] in s for i, s in enumerate(pred_sets)])
        avg_set_size = np.mean([len(s) for s in pred_sets])

        # Rolling empirical coverage from history
        recent_cov = (np.mean(self._coverage_history[-200:])
                      if len(self._coverage_history) >= 20 else None)

        return {
            "coverage": float(coverage),
            "avg_set_size": float(avg_set_size),
            "brier_raw": brier_score(test_y, _safe_clip(np.asarray(test_probs).ravel())),
            "brier_calibrated": brier_score(test_y, cal_probs),
            "ece_raw": expected_calibration_error(test_y, _safe_clip(np.asarray(test_probs).ravel())),
            "ece_calibrated": expected_calibration_error(test_y, cal_probs),
            "alpha_current": float(self.alpha_t),
            "alpha_target": float(self.alpha_target),
            "gamma": float(self.gamma),
            "n_updates": len(self._coverage_history),
            "rolling_coverage": float(recent_cov) if recent_cov is not None else None,
            "n_buffer": len(self._scores_buffer),
        }


# ═══════════════════════════════════════════════════════════════════
# 3. MONDRIAN CONFORMAL PREDICTION
# ═══════════════════════════════════════════════════════════════════

class MondrianConformalCalibrator:
    """
    Mondrian Conformal Prediction (Vovk, 2012).

    Standard conformal gives marginal coverage but may under-cover
    certain subgroups. Mondrian CP partitions data into groups and
    calibrates separately within each group.

    NBA groups (cells):
      - home_rest_class: B2B (<=1 day), short (2-3), normal (4+)
      - elo_tier: top_25%, middle_50%, bottom_25% by ELO difference
      - game_context: regular, rivalry (H2H < 0.4 or > 0.6), b2b_away

    Each Mondrian cell gets its own calibration quantile, ensuring
    conditional coverage: P(Y in C(X) | G=g) >= 1-alpha for all groups g.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        score_fn: str = "hinge",
        min_cell_size: int = 30,
    ):
        """
        Args:
            alpha: Target miscoverage rate.
            score_fn: Non-conformity score function.
            min_cell_size: Minimum calibration samples per cell.
                           Cells below this threshold fall back to global calibration.
        """
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.score_fn_name = score_fn
        self._score_fn = SCORE_FUNCTIONS[score_fn]
        self.min_cell_size = min_cell_size

        # Per-cell calibration data
        self._cell_scores: Dict[str, List[float]] = defaultdict(list)
        self._cell_quantiles: Dict[str, float] = {}
        self._global_scores: List[float] = []
        self._global_quantile: Optional[float] = None
        self._fitted = False

    @staticmethod
    def assign_cell(features: Optional[Dict[str, float]] = None) -> str:
        """
        Assign a game to a Mondrian cell based on context features.

        Args:
            features: Dict with keys like "h_rest", "a_rest", "elo_d",
                      "h2h_wp", "h_b2b". None = "default" cell.

        Returns:
            Cell label string, e.g. "rest_normal__elo_mid__ctx_regular".
        """
        if features is None:
            return "default"

        parts = []

        # 1. Rest classification
        h_rest = features.get("h_rest", 3.0)
        a_rest = features.get("a_rest", 3.0)
        min_rest = min(h_rest, a_rest)
        if min_rest <= 1:
            parts.append("rest_b2b")
        elif min_rest <= 3:
            parts.append("rest_short")
        else:
            parts.append("rest_normal")

        # 2. ELO tier (based on absolute ELO difference)
        elo_d = abs(features.get("elo_d", 0.0))
        if elo_d >= 150:
            parts.append("elo_mismatch")  # lopsided game
        elif elo_d >= 50:
            parts.append("elo_mid")
        else:
            parts.append("elo_close")  # competitive game

        # 3. Game context
        h_b2b = features.get("h_b2b", 0.0)
        h2h_wp = features.get("h2h_wp", 0.5)
        if h_b2b > 0.5 and a_rest >= 3:
            parts.append("ctx_b2b_disadvantage")
        elif h2h_wp < 0.35 or h2h_wp > 0.65:
            parts.append("ctx_rivalry")
        else:
            parts.append("ctx_regular")

        return "__".join(parts)

    @staticmethod
    def extract_cell_features(
        feature_row: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """
        Extract Mondrian cell features from a full feature vector.

        Looks for known feature names in the feature_names list.
        Falls back to defaults if features are not found.

        Args:
            feature_row: Single game feature vector. Shape (n_features,).
            feature_names: List of feature names matching feature_row columns.

        Returns:
            Dict of cell-relevant features.
        """
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        result = {}

        mappings = {
            "h_rest": ["h_rest"],
            "a_rest": ["a_rest"],
            "elo_d": ["elo_d", "d_elo"],
            "h2h_wp": ["h2h_wp"],
            "h_b2b": ["h_b2b"],
        }

        for key, candidates in mappings.items():
            for cand in candidates:
                if cand in name_to_idx:
                    result[key] = float(feature_row[name_to_idx[cand]])
                    break

        return result

    def fit(
        self,
        cal_probs: np.ndarray,
        cal_y: np.ndarray,
        cal_cells: Optional[List[str]] = None,
    ) -> "MondrianConformalCalibrator":
        """
        Fit Mondrian calibration on cell-partitioned calibration data.

        Args:
            cal_probs: Model's P(Y=1) on calibration set. Shape (n,).
            cal_y: True labels. Shape (n,).
            cal_cells: Cell label for each calibration point. If None, all "default".
        """
        cal_probs = _safe_clip(np.asarray(cal_probs, dtype=np.float64).ravel())
        cal_y = np.asarray(cal_y, dtype=np.int32).ravel()
        if cal_cells is None:
            cal_cells = ["default"] * len(cal_probs)
        if len(cal_probs) != len(cal_y) or len(cal_probs) != len(cal_cells):
            raise ValueError("Length mismatch between probs, y, and cells")

        scores = self._score_fn(cal_probs, cal_y)

        # Partition scores by cell
        self._cell_scores = defaultdict(list)
        self._global_scores = list(scores)
        for s, cell in zip(scores, cal_cells):
            self._cell_scores[cell].append(float(s))

        # Compute per-cell quantiles (with fallback to global)
        self._global_quantile = _quantile_with_correction(
            np.array(self._global_scores), self.alpha
        )
        self._cell_quantiles = {}
        for cell, cell_scores in self._cell_scores.items():
            if len(cell_scores) >= self.min_cell_size:
                self._cell_quantiles[cell] = _quantile_with_correction(
                    np.array(cell_scores), self.alpha
                )
            else:
                # Fall back to global calibration for small cells
                self._cell_quantiles[cell] = self._global_quantile

        self._fitted = True
        return self

    def calibrate(
        self,
        test_probs: np.ndarray,
        test_cells: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Calibrate test probabilities using cell-specific calibration.

        For each test point, use the calibration scores from its
        Mondrian cell to compute the conformal p-value.

        Args:
            test_probs: Model's P(Y=1). Shape (n,).
            test_cells: Cell labels for test points. None = "default".

        Returns:
            Calibrated P(Y=1). Shape (n,).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before calibrate()")

        test_probs = _safe_clip(np.asarray(test_probs, dtype=np.float64).ravel())
        if test_cells is None:
            test_cells = ["default"] * len(test_probs)

        test_scores = self._score_fn(test_probs, np.ones(len(test_probs), dtype=np.int32))
        calibrated = np.zeros(len(test_probs), dtype=np.float64)

        for i, (ts, cell) in enumerate(zip(test_scores, test_cells)):
            # Use cell-specific scores if available with enough data
            if cell in self._cell_scores and len(self._cell_scores[cell]) >= self.min_cell_size:
                cal_scores = np.array(self._cell_scores[cell])
            else:
                cal_scores = np.array(self._global_scores)

            n_cal = len(cal_scores)
            count_geq = np.sum(cal_scores >= ts) + 1
            calibrated[i] = count_geq / (n_cal + 1)

        return _safe_clip(calibrated)

    def predict_sets(
        self,
        test_probs: np.ndarray,
        test_cells: Optional[List[str]] = None,
    ) -> List[List[int]]:
        """Prediction sets using cell-specific quantiles."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_sets()")

        test_probs = _safe_clip(np.asarray(test_probs, dtype=np.float64).ravel())
        if test_cells is None:
            test_cells = ["default"] * len(test_probs)

        sets = []
        for p, cell in zip(test_probs, test_cells):
            q = self._cell_quantiles.get(cell, self._global_quantile)
            s = set()
            score_1 = float(self._score_fn(np.array([p]), np.array([1]))[0])
            score_0 = float(self._score_fn(np.array([p]), np.array([0]))[0])
            if score_1 <= q:
                s.add(1)
            if score_0 <= q:
                s.add(0)
            sets.append(sorted(s) if s else [1 if p >= 0.5 else 0])
        return sets

    def diagnostics(
        self,
        test_probs: np.ndarray,
        test_y: np.ndarray,
        test_cells: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Diagnostics with per-cell breakdown."""
        test_probs_arr = _safe_clip(np.asarray(test_probs, dtype=np.float64).ravel())
        test_y = np.asarray(test_y, dtype=np.int32).ravel()
        if test_cells is None:
            test_cells = ["default"] * len(test_probs_arr)

        cal_probs = self.calibrate(test_probs_arr, test_cells)
        pred_sets = self.predict_sets(test_probs_arr, test_cells)

        coverage = np.mean([test_y[i] in s for i, s in enumerate(pred_sets)])
        avg_set_size = np.mean([len(s) for s in pred_sets])

        # Per-cell metrics
        cell_metrics = {}
        cell_indices = defaultdict(list)
        for i, cell in enumerate(test_cells):
            cell_indices[cell].append(i)

        for cell, indices in cell_indices.items():
            idx = np.array(indices)
            if len(idx) < 5:
                continue
            cell_y = test_y[idx]
            cell_raw = test_probs_arr[idx]
            cell_cal = cal_probs[idx]
            cell_sets = [pred_sets[i] for i in indices]
            cell_cov = np.mean([cell_y[j] in cell_sets[j] for j in range(len(idx))])
            cell_metrics[cell] = {
                "n": int(len(idx)),
                "coverage": float(cell_cov),
                "brier_raw": brier_score(cell_y, cell_raw),
                "brier_calibrated": brier_score(cell_y, cell_cal),
                "cal_size": len(self._cell_scores.get(cell, [])),
                "uses_global": len(self._cell_scores.get(cell, [])) < self.min_cell_size,
            }

        return {
            "coverage": float(coverage),
            "avg_set_size": float(avg_set_size),
            "brier_raw": brier_score(test_y, test_probs_arr),
            "brier_calibrated": brier_score(test_y, cal_probs),
            "ece_raw": expected_calibration_error(test_y, test_probs_arr),
            "ece_calibrated": expected_calibration_error(test_y, cal_probs),
            "n_cells": len(self._cell_scores),
            "cells_with_enough_data": sum(
                1 for v in self._cell_scores.values() if len(v) >= self.min_cell_size
            ),
            "cell_metrics": cell_metrics,
            "alpha": self.alpha,
        }


# ═══════════════════════════════════════════════════════════════════
# 4. ENSEMBLE CONFORMAL CALIBRATOR
# ═══════════════════════════════════════════════════════════════════

class EnsembleConformalCalibrator:
    """
    Ensemble Conformal Calibrator.

    Combines Split, ACI, and Mondrian methods via weighted average.
    Weights are determined by validation Brier score on a held-out
    portion of the calibration set.

    The ensemble typically outperforms individual methods because:
      - Split CP is optimal when data is exchangeable
      - ACI handles distribution shift
      - Mondrian provides group fairness
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.01,
        score_fn: str = "hinge",
        min_cell_size: int = 30,
        val_fraction: float = 0.2,
    ):
        """
        Args:
            alpha: Target miscoverage rate.
            gamma: ACI learning rate.
            score_fn: Non-conformity score function.
            min_cell_size: Mondrian minimum cell size.
            val_fraction: Fraction of calibration data used for weight tuning.
        """
        self.alpha = alpha
        self.val_fraction = val_fraction

        self.split = SplitConformalCalibrator(alpha=alpha, score_fn=score_fn)
        self.aci = AdaptiveConformalCalibrator(
            alpha=alpha, gamma=gamma, score_fn=score_fn
        )
        self.mondrian = MondrianConformalCalibrator(
            alpha=alpha, score_fn=score_fn, min_cell_size=min_cell_size
        )

        self._weights: np.ndarray = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        self._fitted = False

    def fit(
        self,
        cal_probs: np.ndarray,
        cal_y: np.ndarray,
        cal_cells: Optional[List[str]] = None,
    ) -> "EnsembleConformalCalibrator":
        """
        Fit all sub-calibrators and determine ensemble weights.

        Splits calibration data into fit/val portions. Fits all methods
        on the fit portion, evaluates Brier on val portion, and assigns
        inverse-Brier weights.
        """
        cal_probs = _safe_clip(np.asarray(cal_probs, dtype=np.float64).ravel())
        cal_y = np.asarray(cal_y, dtype=np.int32).ravel()
        if cal_cells is None:
            cal_cells = ["default"] * len(cal_probs)

        n = len(cal_probs)
        n_val = max(int(n * self.val_fraction), 20)
        n_fit = n - n_val

        if n_fit < 30:
            # Not enough data for proper val split; use uniform weights
            self.split.fit(cal_probs, cal_y)
            self.aci.fit(cal_probs, cal_y)
            self.mondrian.fit(cal_probs, cal_y, cal_cells)
            self._weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
            self._fitted = True
            return self

        # Temporal split: earlier data for fitting, later for validation
        fit_probs, val_probs = cal_probs[:n_fit], cal_probs[n_fit:]
        fit_y, val_y = cal_y[:n_fit], cal_y[n_fit:]
        fit_cells, val_cells = cal_cells[:n_fit], cal_cells[n_fit:]

        # Fit each method on fit portion
        self.split.fit(fit_probs, fit_y)
        self.aci.fit(fit_probs, fit_y)
        self.mondrian.fit(fit_probs, fit_y, fit_cells)

        # Run ACI updates through fit data to build state
        self.aci.update_batch(fit_probs, fit_y)

        # Evaluate on val portion
        briers = []
        for method, cells_arg in [
            (self.split, None),
            (self.aci, None),
            (self.mondrian, val_cells),
        ]:
            if cells_arg is not None:
                cal = method.calibrate(val_probs, cells_arg)
            else:
                cal = method.calibrate(val_probs)
            b = brier_score(val_y, cal)
            briers.append(b)

        briers = np.array(briers)

        # Inverse-Brier weighting: better Brier = higher weight
        # Add small epsilon to avoid division by zero
        inv_briers = 1.0 / (briers + 1e-6)
        self._weights = inv_briers / inv_briers.sum()

        # Now refit on ALL calibration data for production use
        self.split.fit(cal_probs, cal_y)
        self.aci.fit(cal_probs, cal_y)
        self.aci.update_batch(cal_probs, cal_y)
        self.mondrian.fit(cal_probs, cal_y, cal_cells)

        self._fitted = True
        return self

    def calibrate(
        self,
        test_probs: np.ndarray,
        test_cells: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Weighted ensemble of calibrated probabilities.

        Args:
            test_probs: Model's P(Y=1). Shape (n,).
            test_cells: Mondrian cell labels. None = "default".

        Returns:
            Calibrated P(Y=1). Shape (n,).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before calibrate()")

        test_probs = _safe_clip(np.asarray(test_probs, dtype=np.float64).ravel())
        if test_cells is None:
            test_cells = ["default"] * len(test_probs)

        p_split = self.split.calibrate(test_probs)
        p_aci = self.aci.calibrate(test_probs)
        p_mondrian = self.mondrian.calibrate(test_probs, test_cells)

        ensemble = (
            self._weights[0] * p_split
            + self._weights[1] * p_aci
            + self._weights[2] * p_mondrian
        )
        return _safe_clip(ensemble)

    def diagnostics(
        self,
        test_probs: np.ndarray,
        test_y: np.ndarray,
        test_cells: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Full diagnostics including per-method and ensemble metrics."""
        test_probs_arr = _safe_clip(np.asarray(test_probs, dtype=np.float64).ravel())
        test_y = np.asarray(test_y, dtype=np.int32).ravel()
        if test_cells is None:
            test_cells = ["default"] * len(test_probs_arr)

        cal_ensemble = self.calibrate(test_probs_arr, test_cells)

        return {
            "ensemble_brier_raw": brier_score(test_y, test_probs_arr),
            "ensemble_brier_calibrated": brier_score(test_y, cal_ensemble),
            "ensemble_ece_raw": expected_calibration_error(test_y, test_probs_arr),
            "ensemble_ece_calibrated": expected_calibration_error(test_y, cal_ensemble),
            "weights": {
                "split": float(self._weights[0]),
                "aci": float(self._weights[1]),
                "mondrian": float(self._weights[2]),
            },
            "split_diagnostics": self.split.diagnostics(test_probs_arr, test_y),
            "aci_diagnostics": self.aci.diagnostics(test_probs_arr, test_y),
            "mondrian_diagnostics": self.mondrian.diagnostics(
                test_probs_arr, test_y, test_cells
            ),
        }


# ═══════════════════════════════════════════════════════════════════
# 5. UNIFIED ConformalCalibrator (WRAPPER)
# ═══════════════════════════════════════════════════════════════════

class ConformalCalibrator:
    """
    Unified wrapper that integrates conformal calibration into the
    walk-forward backtest framework.

    Usage:
        calibrator = ConformalCalibrator(method="ensemble", alpha=0.1)

        # In each walk-forward fold:
        # 1. Train base model on train set
        # 2. Get raw probs on cal + test sets
        # 3. Calibrate
        raw_probs_cal = model.predict_proba(X_cal)[:, 1]
        raw_probs_test = model.predict_proba(X_test)[:, 1]

        calibrated_test = calibrator.fit_calibrate(
            raw_probs_cal, y_cal, raw_probs_test,
            cal_features=X_cal, test_features=X_test,
            feature_names=feature_names,
        )
    """

    METHODS = ("split", "aci", "mondrian", "ensemble")

    def __init__(
        self,
        method: str = "ensemble",
        alpha: float = 0.1,
        gamma: float = 0.01,
        score_fn: str = "hinge",
        min_cell_size: int = 30,
        cal_fraction: float = 0.3,
    ):
        """
        Args:
            method: Calibration method. One of "split", "aci", "mondrian", "ensemble".
            alpha: Target miscoverage rate.
            gamma: ACI learning rate (only used if method includes ACI).
            score_fn: Non-conformity score function.
            min_cell_size: Mondrian minimum cell size.
            cal_fraction: Fraction of training data to reserve for calibration
                          when doing internal train/cal split in walk-forward.
        """
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method}")

        self.method = method
        self.alpha = alpha
        self.gamma = gamma
        self.score_fn = score_fn
        self.min_cell_size = min_cell_size
        self.cal_fraction = cal_fraction

        self._calibrator = None
        self._last_diagnostics: Optional[Dict[str, Any]] = None

    def _build_calibrator(self):
        """Instantiate the appropriate calibrator."""
        if self.method == "split":
            return SplitConformalCalibrator(alpha=self.alpha, score_fn=self.score_fn)
        elif self.method == "aci":
            return AdaptiveConformalCalibrator(
                alpha=self.alpha, gamma=self.gamma, score_fn=self.score_fn
            )
        elif self.method == "mondrian":
            return MondrianConformalCalibrator(
                alpha=self.alpha, score_fn=self.score_fn,
                min_cell_size=self.min_cell_size,
            )
        elif self.method == "ensemble":
            return EnsembleConformalCalibrator(
                alpha=self.alpha, gamma=self.gamma, score_fn=self.score_fn,
                min_cell_size=self.min_cell_size,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit_calibrate(
        self,
        cal_probs: np.ndarray,
        cal_y: np.ndarray,
        test_probs: np.ndarray,
        cal_features: Optional[np.ndarray] = None,
        test_features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        test_y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fit calibrator on calibration data and calibrate test predictions.

        This is the main entry point for walk-forward integration.

        Args:
            cal_probs: Base model's P(Y=1) on calibration set.
            cal_y: True labels for calibration set.
            test_probs: Base model's P(Y=1) on test set.
            cal_features: Feature matrix for calibration (for Mondrian cells).
            test_features: Feature matrix for test (for Mondrian cells).
            feature_names: Feature names (for Mondrian cell extraction).
            test_y: True labels for test (for diagnostics only, not used in calibration).

        Returns:
            Calibrated P(Y=1) for test set.
        """
        self._calibrator = self._build_calibrator()

        cal_probs = np.asarray(cal_probs, dtype=np.float64).ravel()
        cal_y = np.asarray(cal_y, dtype=np.int32).ravel()
        test_probs = np.asarray(test_probs, dtype=np.float64).ravel()

        # Build Mondrian cells if features provided
        cal_cells = None
        test_cells = None
        if (self.method in ("mondrian", "ensemble")
                and cal_features is not None
                and feature_names is not None):
            cal_cells = []
            for row in cal_features:
                feats = MondrianConformalCalibrator.extract_cell_features(
                    row, feature_names
                )
                cal_cells.append(MondrianConformalCalibrator.assign_cell(feats))

            if test_features is not None:
                test_cells = []
                for row in test_features:
                    feats = MondrianConformalCalibrator.extract_cell_features(
                        row, feature_names
                    )
                    test_cells.append(MondrianConformalCalibrator.assign_cell(feats))

        # Fit
        if self.method == "split":
            self._calibrator.fit(cal_probs, cal_y)
        elif self.method == "aci":
            self._calibrator.fit(cal_probs, cal_y)
            self._calibrator.update_batch(cal_probs, cal_y)
        elif self.method == "mondrian":
            self._calibrator.fit(cal_probs, cal_y, cal_cells)
        elif self.method == "ensemble":
            self._calibrator.fit(cal_probs, cal_y, cal_cells)

        # Calibrate
        if self.method == "mondrian":
            calibrated = self._calibrator.calibrate(test_probs, test_cells)
        elif self.method == "ensemble":
            calibrated = self._calibrator.calibrate(test_probs, test_cells)
        else:
            calibrated = self._calibrator.calibrate(test_probs)

        # Diagnostics if test_y provided
        if test_y is not None:
            test_y = np.asarray(test_y, dtype=np.int32).ravel()
            if self.method == "mondrian":
                self._last_diagnostics = self._calibrator.diagnostics(
                    test_probs, test_y, test_cells
                )
            elif self.method == "ensemble":
                self._last_diagnostics = self._calibrator.diagnostics(
                    test_probs, test_y, test_cells
                )
            else:
                self._last_diagnostics = self._calibrator.diagnostics(
                    test_probs, test_y
                )

        return _safe_clip(calibrated)

    def walk_forward_calibrate(
        self,
        all_probs: np.ndarray,
        all_y: np.ndarray,
        fold_indices: List[Tuple[np.ndarray, np.ndarray]],
        all_features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run conformal calibration within walk-forward backtest folds.

        For each fold:
          1. Take the training portion's last cal_fraction as calibration set
          2. Use the validation portion as test set
          3. Fit conformal calibrator on calibration, apply to test

        Args:
            all_probs: Base model's P(Y=1) for ALL data points. Shape (N,).
            all_y: True labels for ALL data points. Shape (N,).
            fold_indices: List of (train_idx, test_idx) from TimeSeriesSplit.
            all_features: Full feature matrix (N, d) for Mondrian.
            feature_names: Feature names for Mondrian cell extraction.

        Returns:
            Dict with calibrated predictions, per-fold metrics, and aggregates.
        """
        all_probs = np.asarray(all_probs, dtype=np.float64).ravel()
        all_y = np.asarray(all_y, dtype=np.int32).ravel()

        fold_results = []
        all_cal_probs = np.full(len(all_probs), np.nan)
        all_raw_probs = np.full(len(all_probs), np.nan)

        for fi, (train_idx, test_idx) in enumerate(fold_indices):
            # Split training into actual-train and calibration
            n_train = len(train_idx)
            n_cal = max(int(n_train * self.cal_fraction), 50)
            cal_start = n_train - n_cal
            cal_idx = train_idx[cal_start:]

            cal_p = all_probs[cal_idx]
            cal_y_fold = all_y[cal_idx]
            test_p = all_probs[test_idx]
            test_y_fold = all_y[test_idx]

            cal_feats = all_features[cal_idx] if all_features is not None else None
            test_feats = all_features[test_idx] if all_features is not None else None

            try:
                calibrated = self.fit_calibrate(
                    cal_p, cal_y_fold, test_p,
                    cal_features=cal_feats,
                    test_features=test_feats,
                    feature_names=feature_names,
                    test_y=test_y_fold,
                )

                brier_raw = brier_score(test_y_fold, _safe_clip(test_p))
                brier_cal = brier_score(test_y_fold, calibrated)
                ece_raw = expected_calibration_error(test_y_fold, _safe_clip(test_p))
                ece_cal = expected_calibration_error(test_y_fold, calibrated)

                all_cal_probs[test_idx] = calibrated
                all_raw_probs[test_idx] = test_p

                fold_results.append({
                    "fold": fi + 1,
                    "n_cal": len(cal_idx),
                    "n_test": len(test_idx),
                    "brier_raw": round(brier_raw, 5),
                    "brier_calibrated": round(brier_cal, 5),
                    "brier_improvement": round(brier_raw - brier_cal, 5),
                    "ece_raw": round(ece_raw, 5),
                    "ece_calibrated": round(ece_cal, 5),
                    "diagnostics": self._last_diagnostics,
                })
            except Exception as e:
                fold_results.append({
                    "fold": fi + 1,
                    "error": str(e)[:500],
                })

        # Aggregate metrics across folds (only non-NaN test indices)
        valid_mask = ~np.isnan(all_cal_probs)
        if valid_mask.sum() > 0:
            agg_brier_raw = brier_score(all_y[valid_mask], all_raw_probs[valid_mask])
            agg_brier_cal = brier_score(all_y[valid_mask], all_cal_probs[valid_mask])
            agg_ece_raw = expected_calibration_error(all_y[valid_mask], all_raw_probs[valid_mask])
            agg_ece_cal = expected_calibration_error(all_y[valid_mask], all_cal_probs[valid_mask])
        else:
            agg_brier_raw = agg_brier_cal = 1.0
            agg_ece_raw = agg_ece_cal = 1.0

        return {
            "method": self.method,
            "alpha": self.alpha,
            "aggregate_brier_raw": round(agg_brier_raw, 5),
            "aggregate_brier_calibrated": round(agg_brier_cal, 5),
            "aggregate_brier_improvement": round(agg_brier_raw - agg_brier_cal, 5),
            "aggregate_ece_raw": round(agg_ece_raw, 5),
            "aggregate_ece_calibrated": round(agg_ece_cal, 5),
            "folds": fold_results,
            "n_total_evaluated": int(valid_mask.sum()),
            "calibrated_probs": all_cal_probs[valid_mask].tolist(),
            "raw_probs": all_raw_probs[valid_mask].tolist(),
            "true_labels": all_y[valid_mask].tolist(),
        }

    @property
    def diagnostics(self) -> Optional[Dict[str, Any]]:
        """Last diagnostics from fit_calibrate."""
        return self._last_diagnostics


# ═══════════════════════════════════════════════════════════════════
# 6. CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════

def conformal_calibrate(
    model_probs: np.ndarray,
    y_true: np.ndarray,
    alpha: float = 0.1,
    method: str = "ensemble",
    cal_fraction: float = 0.3,
    score_fn: str = "hinge",
    return_diagnostics: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    One-shot conformal calibration with temporal train/calibration/test split.

    Splits data temporally:
      - First (1 - cal_fraction) of data: used to derive model_probs (assumed already done)
      - Last cal_fraction of non-test data: calibration set
      - All data: calibrated

    For the simplest use case:
      1. You have model probs and true labels for the same set
      2. This function splits them temporally, fits conformal on the first part,
         calibrates the second part, and returns all calibrated probs.

    Args:
        model_probs: Raw model probabilities P(Y=1). Shape (N,).
        y_true: True binary labels. Shape (N,).
        alpha: Miscoverage rate (default 0.1).
        method: "split", "aci", "mondrian", or "ensemble".
        cal_fraction: Fraction of data used for calibration.
        score_fn: Non-conformity score function.
        return_diagnostics: If True, return (calibrated, diagnostics) tuple.

    Returns:
        calibrated: Calibrated probabilities. Shape (N,).
        diagnostics: (only if return_diagnostics=True) Dict with metrics.
    """
    model_probs = np.asarray(model_probs, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    n = len(model_probs)

    if n < 50:
        warnings.warn(f"Very small dataset ({n} points). Conformal calibration may be unreliable.")
        if return_diagnostics:
            return model_probs, {"warning": "dataset too small", "n": n}
        return model_probs

    # Temporal split: first portion = calibration, rest = test
    n_cal = max(int(n * cal_fraction), 30)
    n_cal = min(n_cal, n - 20)  # leave at least 20 for test

    cal_probs = model_probs[:n_cal]
    cal_y = y_true[:n_cal]
    test_probs = model_probs[n_cal:]
    test_y = y_true[n_cal:]

    calibrator = ConformalCalibrator(
        method=method, alpha=alpha, score_fn=score_fn,
        cal_fraction=1.0,  # we already split externally
    )

    calibrated_test = calibrator.fit_calibrate(
        cal_probs, cal_y, test_probs, test_y=test_y,
    )

    # For calibration portion, use leave-one-out style calibration
    # (each point calibrated using all other cal points)
    calibrated_cal = _loo_calibrate(cal_probs, cal_y, score_fn)

    calibrated = np.concatenate([calibrated_cal, calibrated_test])

    if return_diagnostics:
        diag = calibrator.diagnostics or {}
        diag["n_calibration"] = int(n_cal)
        diag["n_test"] = int(len(test_probs))
        diag["overall_brier_raw"] = brier_score(y_true, model_probs)
        diag["overall_brier_calibrated"] = brier_score(y_true, calibrated)
        diag["overall_ece_raw"] = expected_calibration_error(y_true, model_probs)
        diag["overall_ece_calibrated"] = expected_calibration_error(y_true, calibrated)
        diag["brier_improvement"] = round(
            diag["overall_brier_raw"] - diag["overall_brier_calibrated"], 5
        )
        return calibrated, diag

    return calibrated


def _loo_calibrate(
    probs: np.ndarray,
    y: np.ndarray,
    score_fn: str = "hinge",
) -> np.ndarray:
    """
    Leave-one-out conformal calibration for the calibration set itself.

    For each point i, compute its conformal p-value using all other
    calibration points as the reference set. This avoids using the
    same data for both fitting and calibrating.

    Optimized to O(n log n) using sorted scores and binary search.

    Args:
        probs: Model's P(Y=1). Shape (n,).
        y: True labels. Shape (n,).
        score_fn: Non-conformity score function name.

    Returns:
        LOO-calibrated probabilities. Shape (n,).
    """
    score_func = SCORE_FUNCTIONS[score_fn]
    n = len(probs)
    if n < 5:
        return _safe_clip(probs)

    all_scores = score_func(probs, y)
    test_scores = score_func(probs, np.ones(n, dtype=np.int32))

    # Sort calibration scores for O(log n) lookup per point
    sorted_scores = np.sort(all_scores)

    calibrated = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ts = test_scores[i]
        # Total count >= ts in all_scores (including self)
        total_geq = n - int(np.searchsorted(sorted_scores, ts, side="left"))
        # Was self's calibration score >= test score? If so, subtract 1
        self_geq = 1 if all_scores[i] >= ts else 0
        count_geq = total_geq - self_geq + 1  # +1 for finite-sample correction
        calibrated[i] = count_geq / n  # (n-1 others + 1 self) = n

    return _safe_clip(calibrated)


# ═══════════════════════════════════════════════════════════════════
# 7. SUPABASE EXPERIMENT ENTRY
# ═══════════════════════════════════════════════════════════════════

def create_conformal_experiment(
    method: str = "ensemble",
    alpha: float = 0.1,
    gamma: float = 0.01,
    score_fn: str = "hinge",
    cal_fraction: float = 0.3,
    min_cell_size: int = 30,
    model_type: str = "xgboost_gpu",
    n_splits: int = 3,
    baseline_brier: float = 0.2187,
    priority: int = 8,
    target_space: str = "kaggle",
    agent_name: str = "adam_conformal",
    description: Optional[str] = None,
    hypothesis: Optional[str] = None,
    database_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Create a conformal prediction experiment in the Supabase nba_experiments table.

    This queues an experiment for the GPU runner (Kaggle/Colab) to pick up
    and execute. The runner trains a base model, then applies conformal
    calibration, measuring the Brier improvement.

    Args:
        method: Conformal method ("split", "aci", "mondrian", "ensemble").
        alpha: Target miscoverage rate.
        gamma: ACI learning rate.
        score_fn: Non-conformity score function.
        cal_fraction: Fraction of training data for calibration.
        min_cell_size: Mondrian minimum cell size.
        model_type: Base model type (e.g., "xgboost_gpu", "mlp").
        n_splits: Number of walk-forward splits.
        baseline_brier: Current best Brier score for comparison.
        priority: Experiment priority (higher = run sooner).
        target_space: Where to run ("kaggle", "colab", "S11", "any").
        agent_name: Who created this experiment.
        description: Human-readable description.
        hypothesis: Expected outcome.
        database_url: PostgreSQL connection string. Falls back to DATABASE_URL env var.

    Returns:
        Dict with experiment_id and status, or None on failure.
    """
    import os

    db_url = database_url or os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("[CONFORMAL] No DATABASE_URL — cannot create experiment")
        return None

    experiment_id = f"conformal-{method}-{score_fn}-a{int(alpha*100)}-{int(time.time())}"

    if description is None:
        description = (
            f"Conformal Prediction calibration ({method} method, "
            f"score={score_fn}, alpha={alpha}) on {model_type}. "
            f"Expected Brier improvement: -0.01 to -0.02."
        )

    if hypothesis is None:
        hypothesis = (
            f"Conformal {method} with {score_fn} scores at alpha={alpha} "
            f"will improve calibration, reducing Brier from {baseline_brier:.4f} "
            f"by 0.01-0.02 points through better probability coverage."
        )

    params = {
        "model_type": model_type,
        "conformal": {
            "method": method,
            "alpha": alpha,
            "gamma": gamma,
            "score_fn": score_fn,
            "cal_fraction": cal_fraction,
            "min_cell_size": min_cell_size,
        },
        "n_splits": n_splits,
        "experiment_category": "calibration",
    }

    try:
        import psycopg2

        conn = psycopg2.connect(db_url, connect_timeout=10, options="-c search_path=public")
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO public.nba_experiments
            (experiment_id, agent_name, experiment_type, description, hypothesis,
             params, priority, status, target_space, baseline_brier)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s)
            RETURNING id, experiment_id
            """,
            (
                experiment_id,
                agent_name,
                "calibration_test",
                description,
                hypothesis,
                json.dumps(params),
                priority,
                target_space,
                baseline_brier,
            ),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()

        result = {
            "id": row[0],
            "experiment_id": row[1],
            "status": "pending",
            "method": method,
            "alpha": alpha,
            "score_fn": score_fn,
            "target_space": target_space,
            "baseline_brier": baseline_brier,
        }
        print(f"[CONFORMAL] Experiment created: {experiment_id} (id={row[0]})")
        return result

    except Exception as e:
        print(f"[CONFORMAL] Failed to create experiment: {e}")
        return None


def create_conformal_sweep(
    baseline_brier: float = 0.2187,
    model_type: str = "xgboost_gpu",
    database_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Create a sweep of conformal experiments to find the best configuration.

    Tests multiple combinations of method, alpha, and score function.

    Args:
        baseline_brier: Current best Brier for comparison.
        model_type: Base model to calibrate.
        database_url: PostgreSQL connection string.

    Returns:
        List of created experiment dicts.
    """
    configs = [
        # Primary: ensemble with different alphas
        {"method": "ensemble", "alpha": 0.10, "score_fn": "hinge", "priority": 9},
        {"method": "ensemble", "alpha": 0.15, "score_fn": "hinge", "priority": 8},
        {"method": "ensemble", "alpha": 0.05, "score_fn": "hinge", "priority": 8},
        # Score function comparison
        {"method": "ensemble", "alpha": 0.10, "score_fn": "margin", "priority": 7},
        {"method": "ensemble", "alpha": 0.10, "score_fn": "log", "priority": 7},
        # Individual methods (best alpha)
        {"method": "split", "alpha": 0.10, "score_fn": "hinge", "priority": 7},
        {"method": "aci", "alpha": 0.10, "score_fn": "hinge", "priority": 7},
        {"method": "mondrian", "alpha": 0.10, "score_fn": "hinge", "priority": 7},
        # ACI with different gammas
        {"method": "aci", "alpha": 0.10, "score_fn": "hinge", "gamma": 0.005, "priority": 6},
        {"method": "aci", "alpha": 0.10, "score_fn": "hinge", "gamma": 0.05, "priority": 6},
        # Mondrian with different cell sizes
        {"method": "mondrian", "alpha": 0.10, "score_fn": "hinge", "min_cell_size": 20, "priority": 6},
        {"method": "mondrian", "alpha": 0.10, "score_fn": "hinge", "min_cell_size": 50, "priority": 6},
    ]

    results = []
    for cfg in configs:
        result = create_conformal_experiment(
            method=cfg["method"],
            alpha=cfg["alpha"],
            score_fn=cfg["score_fn"],
            gamma=cfg.get("gamma", 0.01),
            min_cell_size=cfg.get("min_cell_size", 30),
            model_type=model_type,
            baseline_brier=baseline_brier,
            priority=cfg.get("priority", 7),
            database_url=database_url,
            description=(
                f"Conformal sweep: {cfg['method']}/{cfg['score_fn']} "
                f"alpha={cfg['alpha']}"
            ),
        )
        if result:
            results.append(result)

    print(f"[CONFORMAL] Created {len(results)}/{len(configs)} sweep experiments")
    return results
