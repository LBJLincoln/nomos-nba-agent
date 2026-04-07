# Research Proposal: Seasonal-Recency Calibration Window for Platt/LR-Meta
**Date:** 2026-04-07
**Status:** PROPOSED
**Expected Brier improvement:** -0.002 to -0.006
**Priority:** MEDIUM (fleet best 0.2219, target 0.21837, gap 0.00353)

---

## Problem

The current `lr_platt` and `lr_meta` calibration methods use a fixed-size holdout slice
to fit the calibration layer:
```python
cal_size = min(400, max(50, len(ti_safe) // 3))
cal_slice = ti_safe[-cal_size:]
```

This uses the last N samples regardless of their temporal distance. In an NBA season,
calibration patterns drift significantly:
- **Early season** (Oct–Nov): high uncertainty, models over/underconfident near 50%
- **Mid season** (Jan–Feb): stable patterns, best calibration data
- **Late season** (Mar–Apr): tanking, load management, model drift

Using a 400-game flat window may mix early and late season signals, degrading calibration.

## Proposed Method: Season-Weighted Calibration Slice

Replace fixed `cal_size` with a **recency-decayed** window that emphasizes recent
games (last 8 weeks = ~60 games) rather than last N samples.

### Implementation (in app.py lr_platt / lr_meta blocks)

```python
# NEW: Time-aware calibration slice
# Prefer last 8 weeks of games; fall back to last 200 if insufficient
RECENT_GAMES = 60  # ~8 weeks of NBA season
if len(ti_safe) > RECENT_GAMES + 50:
    # Use most recent games for calibration (best drift signal)
    cal_slice = ti_safe[-RECENT_GAMES:]
else:
    # Fallback: use 1/3 of available training data
    cal_size = max(30, len(ti_safe) // 3)
    cal_slice = ti_safe[-cal_size:]
```

Apply identically to both `lr_platt` and `lr_meta` blocks.

### Why This Helps

- **Calibration is seasonal**: a model trained on 3 seasons uses old patterns;
  the lr-meta calibrator fitted on recent 60 games corrects for current-season drift
- **Less data, better signal**: 60 highly-relevant games > 400 stale games for calibration
- **NBA evidence**: "Back-to-back, fatigue, travel" effects are seasonally concentrated;
  recent calibration data captures these patterns better

### Risk

- Tiny calibration set (60 games) has higher variance → add a fallback:
  if calibration increases Brier by >0.005, skip calibration (`cal_method = "none"`)

```python
_lr_meta_brier = brier_score_loss(y_eval[cal_slice], _lr_meta_cal.predict_proba(
    m.predict_proba(X_sub[cal_slice])[:, 1].reshape(-1, 1))[:, 1])
_raw_brier = brier_score_loss(y_eval[cal_slice], m.predict_proba(X_sub[cal_slice])[:, 1])
if _lr_meta_brier > _raw_brier + 0.005:
    _lr_meta_cal = None  # calibration hurts, skip
```

---

## Expected Impact

| Model Type | Expected Improvement | Notes |
|------------|--------------------|-------|
| XGBoost | -0.002 to -0.004 | Moderate calibration drift |
| Random Forest | -0.003 to -0.006 | Most drift-sensitive |
| LightGBM | -0.002 to -0.004 | Good baseline calibration |
| ExtraTrees | -0.003 to -0.006 | Same as RF |

## Implementation Steps

1. Modify `lr_platt` block in `hf-space/app.py` (lines ~1363-1378)
2. Modify `lr_meta` block in `hf-space/app.py` (lines ~1347-1362)
3. Add Brier-safety check (skip calibration if it hurts validation fold)
4. Test on S10 only first; if best_brier improves, roll to all islands

## Cross-Project Port

Same fix applies to `nomos-political-alpha/hf-space/app.py`. Political engine has
fewer events (372-1180), so **use RECENT_GAMES = 30** (last 4 weeks of events).

---

## Related Findings (2026-04)
- XGBoost with proper calibration achieved Brier 0.202 on 2024 NBA test data
  (vs our 0.2219 fleet best) → gap is partially calibration quality, not feature quality
- Stacked ensemble + isotonic regression: 83.27% accuracy, AUC 0.9213 (Nature 2025)
- Temporal decay in training features already implemented (Cat35); extend to calibration
