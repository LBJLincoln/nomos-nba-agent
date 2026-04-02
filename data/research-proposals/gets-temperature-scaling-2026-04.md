# Research Proposal: Isotonic Regression Calibration for NBA Win Probability
**Date:** 2026-04-02 (updated same day — method corrected from temperature scaling to isotonic)
**Status:** PROPOSED  
**Expected Brier improvement:** -0.005 to -0.015  
**Priority:** HIGH (fleet best 0.22182, target 0.21837, gap 0.00345)

---

## Problem

The current fleet best Brier is **0.22182** (S5 catboost, gen 370). The target is **0.21837**.
The GA evolves good feature sets but raw model outputs are often poorly calibrated.

## Method Selection

**For our dataset (9551 NBA games > 1000 threshold): use Isotonic Regression, not Temperature Scaling.**

| Method | Best For | Our Case |
|--------|----------|----------|
| Isotonic Regression | N > 1,000, tree models | **USE THIS** |
| Temperature Scaling | Neural nets, small N | Skip |
| Platt Scaling | SVMs, N < 1,000 | Skip |

Source: sklearn docs, Rohan Paul 2025 ML calibration guide, MDPI 2026 NBA study.

---

## Implementation Plan

**Step 1: Hold-out calibration split (in app.py)**

```python
from sklearn.model_selection import train_test_split

# After GA produces best genome, fit on 80% train, calibrate on 20% holdout
X_train, X_cal, y_train, y_cal = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model.fit(X_train, y_train)
```

**Step 2: Fit isotonic calibration wrapper**

```python
from sklearn.calibration import CalibratedClassifierCV

# Method A: Isotonic regression wrapper (recommended for N > 1000)
cal_model = CalibratedClassifierCV(
    estimator=model,
    method='isotonic',
    cv='prefit'  # model already fit, calibrate on holdout
)
cal_model.fit(X_cal, y_cal)
calibrated_brier = brier_score_loss(y_cal, cal_model.predict_proba(X_cal)[:, 1])
```

**Step 3: Auto-select best calibration method**

```python
from sklearn.metrics import brier_score_loss

best_brier = brier_score_loss(y_cal, model.predict_proba(X_cal)[:, 1])
best_method = 'none'

for method in ['isotonic', 'sigmoid']:
    cal = CalibratedClassifierCV(estimator=model, method=method, cv='prefit')
    cal.fit(X_cal, y_cal)
    b = brier_score_loss(y_cal, cal.predict_proba(X_cal)[:, 1])
    if b < best_brier:
        best_brier = b
        best_method = method
        best_cal_model = cal
```

**Step 4: Store calibration info in /api/status**

```python
status['calibration_method'] = best_method
status['calibrated_brier'] = float(best_brier)
status['raw_brier'] = float(raw_brier)
status['calibration_delta'] = float(raw_brier - best_brier)
```

---

## Expected Impact

| Model Type | Typical Improvement | Notes |
|------------|--------------------|---------|
| CatBoost (S5) | -0.005 to -0.010 | Often overconfident |
| XGBoost (S3) | -0.005 to -0.008 | Moderate benefit |
| Random Forest (S4) | -0.008 to -0.015 | RF notoriously miscalibrated |
| ExtraTrees | -0.008 to -0.015 | Same as RF |

Fleet average improvement: **-0.005 to -0.010 Brier**  
S5 catboost targeted: **-0.005 to -0.010** (0.22182 → ~0.212-0.217, likely breaks 0.21837 target)

---

## Cross-Project Port

Same pattern applies to `nomos-political-alpha`. Political models have 372 events — below the 1000 threshold, so **use Platt scaling (sigmoid) for political engine**, not isotonic.

```python
# Political: use sigmoid calibration (N=372 events < 1000)
from sklearn.calibration import CalibratedClassifierCV
cal_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
```

---

## Implementation Steps

1. Add calibration block to `hf-space/app.py` after GA selects best genome (~30 lines)
2. Use 80/20 train/calibration split (maintain temporal order — calibration set = most recent 20%)
3. Expose `calibration_method`, `calibrated_brier`, `calibration_delta` in `/api/status`
4. Monitor: if `calibration_delta < 0` (calibration hurts), fall back to `method='none'`
