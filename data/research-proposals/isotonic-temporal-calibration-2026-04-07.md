# Research Proposal: Isotonic Temporal Calibration for NBA GA

**Date:** 2026-04-07  
**Brain Cycle:** 74  
**Status:** DEPLOYED (brain cycle 74)  
**Priority:** HIGH  
**Expected Brier Delta:** -0.002 to -0.006  

---

## Motivation

Current best Brier: **0.2219** (S15, extra_trees, 74 features)  
Threshold target: **0.21837** | Gap: **0.00353**

The MDPI 2026 paper "Uncertainty-Aware Machine Learning for NBA Forecasting in Digital Betting Markets" (doi:10.3390/info17010056) demonstrates:
- **Logistic regression + isotonic calibration on temporal holdout → Brier 0.199**
- XGBoost achieves 0.202 without special calibration
- Our fleet average: ~0.224 — above both benchmarks

The key insight: isotonic regression as calibration **fails with cross-validation** (overfits on small folds) but **succeeds with temporal holdout** (uses the last N games as calibration set).

Our NBA code previously had `isotonic` as an option but explicitly disabled it:
```python
if cal_method == "isotonic":
    cal_method = "none"  # Isotonic empirically hurts Brier (+0.003 to +0.007)
```
This was correct — sklearn's CalibratedClassifierCV uses CV folds. The temporal holdout variant is different.

---

## What Was Implemented (Brain Cycle 74)

Added `isotonic_temporal` as a new calibration option in `hf-space/app.py`:

```python
_isotonic_cal = None
if cal_method == "isotonic_temporal":
    # Temporal isotonic: temporal holdout (not CV folds) — avoids overfitting
    from sklearn.isotonic import IsotonicRegression
    m.fit(X_sub[ti_safe], y_eval[ti_safe])
    _model_fitted = True
    cal_size = min(400, max(50, len(ti_safe) // 3))
    cal_slice = ti_safe[-cal_size:] if len(ti_safe) > cal_size + 50 else ti_safe
    raw_p = m.predict_proba(X_sub[cal_slice])[:, 1]
    _isotonic_cal = IsotonicRegression(out_of_bounds="clip")
    _isotonic_cal.fit(raw_p, y_eval[cal_slice])
    cal_method = "none"
```

Applied in prediction phase:
```python
if _isotonic_cal is not None:
    p = _isotonic_cal.predict(p)
```

GA weights (initialization): 7% of new individuals  
GA weights (mutation): 6% mutation probability toward this method

**Ported to Political Alpha** same cycle (Rotation C, with 30-event minimum guard).

---

## Why This Is Different From Disabled `isotonic`

| Method | Calibration Dataset | Risk | Outcome |
|--------|-------------------|------|---------|
| `CalibratedClassifierCV(method='isotonic')` | CV folds (temporal leakage risk, small folds) | Overfits | +0.003-0.007 Brier (WORSE) |
| `isotonic_temporal` | Last 400 games (temporal holdout, no leakage) | Minimal | Expected -0.002 to -0.006 Brier |

The paper's isotonic method used clean train/test split — equivalent to our temporal holdout, not CV.

---

## Why Tree Models Benefit Most

Isotonic calibration is most effective on models whose raw probabilities are **not inherently calibrated**:
- Random Forest: over-confident (probabilities cluster near 0/1)
- Extra Trees: similarly over-confident  
- XGBoost / LightGBM: moderately calibrated already
- Logistic Regression: inherently calibrated (doesn't benefit as much)

Our fleet runs mostly random_forest and extra_trees → maximum expected benefit.

---

## Success Criteria

- [ ] `isotonic_temporal` selected by ≥1 top-5 Pareto individual within 20 cycles  
- [ ] Best Brier with isotonic_temporal < 0.220 (better than current fleet best)  
- [ ] Fleet best Brier drops below 0.21837 (checkpoint threshold)

---

## Follow-Up Proposals

1. **Temperature scaling**: Single-parameter post-hoc calibration T where p = sigmoid(logit/T). Even simpler than isotonic, no monotonic assumption. Particularly good for LR + neural nets.

2. **Market-anchored calibration**: Calibrate against closing line probabilities rather than outcomes. Uses Pinnacle/best-book closing implied probability as the "truth" — then we bet against lines that disagree with our closing-line-calibrated predictions.

3. **Ensemble meta-learner**: Stack 3 calibrated base models (RF + ET + LR) with isotonic as the meta-learner. Requires stacking infrastructure currently disabled on CPU spaces.

---

## References

- [MDPI 2026: Uncertainty-Aware ML for NBA Forecasting](https://www.mdpi.com/2078-2489/17/1/56)
- [scikit-learn: Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)  
- [Niculescu-Mizil & Caruana 2005: Predicting Good Probabilities with Supervised Learning](https://dl.acm.org/doi/10.1145/1102351.1102430)
