"""
Isotonic Post-Calibration for NBA predictions
=============================================
Lightweight piecewise-linear interpolation at inference time.
No sklearn dependency on the VM — only plain Python + json.

Fitting happens exclusively on HF Space (S10/S11).  The result is
serialised to ``calibration/isotonic_breakpoints.json`` and committed
to the repo; predict_today.py reads that file at runtime.

Classes
-------
IsotonicPostCalibrator
    Load pre-computed breakpoints and apply calibration at inference.
    fit_and_export()  — run on HF Space to produce the JSON artefact.
    from_backtest()   — query Supabase for historical data, then fit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Inference-only calibrator (zero sklearn dependency)
# ---------------------------------------------------------------------------

class IsotonicPostCalibrator:
    """Apply a pre-fitted isotonic mapping using piecewise linear interpolation.

    The calibration mapping is stored as two parallel lists:
      ``x_points`` — raw model probabilities (monotone ascending)
      ``y_points`` — calibrated probabilities (monotone ascending)

    At inference time ``calibrate(p)`` does a simple linear interpolation
    between the two bracketing breakpoints.  This is O(log n) via bisection
    and has no external dependencies beyond the Python standard library.
    """

    def __init__(self, x_points: List[float], y_points: List[float],
                 metadata: Optional[dict] = None):
        if len(x_points) != len(y_points):
            raise ValueError("x_points and y_points must have equal length")
        if len(x_points) < 2:
            raise ValueError("Need at least 2 breakpoints")
        # Ensure ascending order (isotonic guarantee)
        pairs = sorted(zip(x_points, y_points))
        self._x = [p[0] for p in pairs]
        self._y = [p[1] for p in pairs]
        self.metadata = metadata or {}

    # ------------------------------------------------------------------
    # Core inference method
    # ------------------------------------------------------------------

    def calibrate(self, prob: float) -> float:
        """Return the calibrated probability for a raw model probability.

        Uses piecewise linear interpolation; clamps to [0, 1].
        """
        prob = float(prob)
        if math.isnan(prob):
            return 0.5

        x, y = self._x, self._y

        # Clamp to breakpoint range
        if prob <= x[0]:
            return float(y[0])
        if prob >= x[-1]:
            return float(y[-1])

        # Binary search for the bracketing interval
        lo, hi = 0, len(x) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if x[mid] <= prob:
                lo = mid
            else:
                hi = mid

        # Linear interpolation
        t = (prob - x[lo]) / (x[hi] - x[lo])
        calibrated = y[lo] + t * (y[hi] - y[lo])
        return float(max(0.0, min(1.0, calibrated)))

    def is_identity(self) -> bool:
        """Return True if this calibrator applies no correction (stub mapping)."""
        return self.metadata.get("identity", False)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "x_points": self._x,
            "y_points": self._y,
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "IsotonicPostCalibrator":
        """Load from a JSON breakpoints file.  Returns an identity calibrator
        if the file does not exist (graceful degradation)."""
        path = Path(path)
        if not path.exists():
            return cls._identity()
        data = json.loads(path.read_text())
        return cls(
            x_points=data["x_points"],
            y_points=data["y_points"],
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def _identity(cls) -> "IsotonicPostCalibrator":
        """Return a pass-through calibrator (no-op)."""
        x = [round(i * 0.05, 2) for i in range(21)]  # 0.00 … 1.00
        return cls(x_points=x, y_points=list(x),
                   metadata={"identity": True, "note": "stub — run fit_and_export on HF Space"})

    # ------------------------------------------------------------------
    # Fitting methods (run on HF Space — sklearn available there)
    # ------------------------------------------------------------------

    @classmethod
    def fit_and_export(
        cls,
        raw_probs: List[float],
        outcomes: List[int],
        output_path: Path,
        n_breakpoints: int = 20,
    ) -> "IsotonicPostCalibrator":
        """Fit isotonic regression on (raw_probs, outcomes) and save breakpoints.

        MUST run on HF Space where sklearn is available.

        Parameters
        ----------
        raw_probs   : list of float  — blended final_prob values (pre-calibration)
        outcomes    : list of int    — 1 = home won, 0 = away won
        output_path : Path           — where to write isotonic_breakpoints.json
        n_breakpoints: int           — number of evenly-spaced x-grid points to keep

        Returns
        -------
        IsotonicPostCalibrator with fitted breakpoints
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "fit_and_export() requires sklearn — run this on HF Space, not the VM"
            ) from exc

        raw = np.array(raw_probs, dtype=float)
        y   = np.array(outcomes,  dtype=float)

        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(raw, y)

        # Evaluate on a fine grid so the breakpoints are compact and portable
        x_grid = np.linspace(0.0, 1.0, max(n_breakpoints, 10))
        y_grid = ir.predict(x_grid)

        # Ensure strict monotonicity (handle isotonic plateau ties)
        y_grid = np.maximum.accumulate(y_grid)

        calibrator = cls(
            x_points=x_grid.tolist(),
            y_points=y_grid.tolist(),
            metadata={
                "identity": False,
                "n_samples": int(len(raw)),
                "n_breakpoints": int(len(x_grid)),
                "fitted_on": "HF Space",
            },
        )
        calibrator.save(output_path)
        return calibrator

    @classmethod
    def from_backtest(
        cls,
        output_path: Path,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        min_samples: int = 200,
    ) -> "IsotonicPostCalibrator":
        """Query Supabase for historical blended predictions vs outcomes, fit,
        and export.

        Queries the ``nba_predictions`` table for rows where:
          - ``final_prob`` IS NOT NULL  (the blended probability)
          - ``home_won`` IS NOT NULL    (outcome is known)
          - ``model_version`` LIKE 'blended%'

        Falls back to identity calibrator if insufficient data or Supabase
        is unreachable.

        MUST run on HF Space.
        """
        import os

        url = supabase_url or os.environ.get("SUPABASE_URL", "")
        key = supabase_key or os.environ.get("SUPABASE_SERVICE_KEY", "") \
              or os.environ.get("SUPABASE_KEY", "")

        if not url or not key:
            print("[CALIBRATOR] Supabase credentials missing — returning identity")
            return cls._identity()

        try:
            from supabase import create_client
        except ImportError:
            print("[CALIBRATOR] supabase-py not installed — returning identity")
            return cls._identity()

        try:
            client = create_client(url, key)
            resp = (
                client.table("nba_predictions")
                .select("final_prob,home_won")
                .not_.is_("final_prob", "null")
                .not_.is_("home_won", "null")
                .like("model_version", "blended%")
                .execute()
            )
            rows = resp.data or []
        except Exception as exc:
            print(f"[CALIBRATOR] Supabase query failed: {exc} — returning identity")
            return cls._identity()

        if len(rows) < min_samples:
            print(
                f"[CALIBRATOR] Only {len(rows)} rows found (need {min_samples}) "
                "— returning identity calibrator"
            )
            return cls._identity()

        raw_probs = [float(r["final_prob"]) for r in rows]
        outcomes  = [int(r["home_won"])     for r in rows]

        print(f"[CALIBRATOR] Fitting on {len(rows)} historical predictions")
        calibrator = cls.fit_and_export(raw_probs, outcomes, output_path)
        print(
            f"[CALIBRATOR] Saved breakpoints to {output_path} "
            f"({len(calibrator._x)} points)"
        )
        return calibrator
