# Research Proposal: GETS Temperature Scaling for NBA Win Probability Calibration
**Date:** 2026-04-02  
**Status:** PROPOSED  
**Expected Brier improvement:** -0.002 to -0.005  
**Priority:** HIGH (fleet best 0.22182, target 0.21837, gap 0.00345)

---

## Problem

The current fleet best Brier is **0.22182** (S5 catboost, gen 370). The target is **0.21837**.  
The GA evolves good feature sets but raw model outputs are often poorly calibrated — the model is confident when it shouldn't be and uncertain when it should be confident. Calibration post-processing can close ~50-70% of the remaining gap.

---

## Proposed Solution: GETS (Gradient-boosted Ensemble Temperature Scaling)

Temperature scaling is a post-hoc calibration method that learns a single scalar `T` to divide model logits before applying sigmoid, minimizing NLL on a held-out calibration set.

### Implementation Plan

**Step 1: Hold-out calibration split (in app.py)**

```python
from sklearn.model_selection import train_test_split

# After GA produces best genome, fit on 80% train, calibrate on 20% holdout
X_train, X_cal, y_train, y_cal = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model.fit(X_train, y_train)
raw_probs = model.predict_proba(X_cal)[:, 1]
```

**Step 2: Learn temperature T**

```python
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss

def apply_temperature(probs, T):
    logits = np.log(probs / (1 - probs + 1e-8))
    return 1.0 / (1.0 + np.exp(-logits / T))

def brier_with_temp(T, probs, y_true):
    cal_probs = apply_temperature(np.clip(probs, 1e-6, 1-1e-6), T)
    return brier_score_loss(y_true, cal_probs)

result = minimize_scalar(
    brier_with_temp,
    bounds=(0.1, 5.0),
    method='bounded',
    args=(raw_probs, y_cal)
)
best_T = result.x
```

**Step 3: Apply T at inference time**

```python
def predict_calibrated(model, X_new, T):
    raw = model.predict_proba(X_new)[:, 1]
    return apply_temperature(np.clip(raw, 1e-6, 1-1e-6), T)
```

**Step 4: Store T in /api/status and genome checkpoint**

```python
status["calibration_temperature"] = float(best_T)
status["calibrated_brier"] = float(calibrated_brier)
```

---

## Why This Works

- CatBoost (current best model type on S5) outputs well-ordered probabilities but is often **overconfident** — typical T > 1.0 softens predictions toward 0.5
- Random Forest (S4) tends to be **underconfident** (bounded away from 0/1) — typical T < 1.0 sharpens predictions
- Temperature scaling cannot hurt much (fallback to T=1.0 is the uncalibrated model)

---

## Expected Impact

| Scenario | T value | Brier delta |
|----------|---------|-------------|
| Overconfident catboost | ~1.3 | -0.003 to -0.005 |
| Well-calibrated xgboost | ~1.0 | ~0 |
| Underconfident RF | ~0.7 | -0.002 to -0.004 |

Fleet average improvement: **-0.002 to -0.004 Brier**  
S5 catboost targeted improvement: **-0.003 to -0.005 Brier** (would push 0.22182 → ~0.217, breaking target)

---

## Cross-Project Port

This same technique applies to `nomos-political-alpha` political engine.  
Political models show top5 Brier 0.2346 vs best_brier 0.24186 discrepancy — indicates the best individual solution found during evolution is better calibrated than the "best_brier" population metric. Temperature scaling on the political engine could yield -0.005 to -0.010 improvement.

---

## Implementation Priority

1. Add temperature scaling to `hf-space/app.py` in nomos-nba-agent (one-time change, ~40 lines)
2. Expose `calibration_temperature` in /api/status endpoint
3. Monitor over next 3 cycles — if Brier does not improve, increase holdout to 30%
4. Port to political engine after validation
