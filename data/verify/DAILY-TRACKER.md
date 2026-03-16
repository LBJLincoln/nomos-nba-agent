# NBA Prediction Tracker — Daily Results

> Auto-updated by `ops/nba-verify-results.py`
> Last update: 2026-03-16

## Cumulative Record

| Metric | Record | Rate |
|--------|--------|------|
| **Moneyline** | 6-1 | **85.7%** |
| **Spread ATS** | 3-4 | 42.9% |

## Bankroll ($100 start)

| Strategy | P/L | Balance | ROI |
|----------|-----|---------|-----|
| Flat $5 ML (HIGH+ conf) | -$2.04 | $97.96 | -2.0% |
| Flat $5 ML (ALL picks) | **+$2.07** | **$102.07** | +2.1% |
| Flat $5 Spread ATS (-110) | -$6.35 | $93.65 | -6.4% |

## Daily Breakdown

### 2026-03-15 (7 games)

| Game | ML Pred | Conf | Result | ML | Spread Pred | Margin | ATS |
|------|---------|------|--------|-----|-------------|--------|-----|
| OKC vs MIN | OKC 85% | HIGH | OKC 116-103 | OK | -9.0 | +13 | OK |
| CLE vs DAL | CLE 85% | HIGH | DAL 130-120 | MISS | -7.2 | -10 | MISS |
| TOR vs DET | TOR 60% | LOW | TOR 119-108 | OK | -3.5 | +11 | OK |
| MIL vs IND | MIL 72% | MED | MIL 134-123 | OK | -4.4 | +11 | OK |
| PHI vs POR | PHI 74% | V.HIGH | PHI 109-103 | OK | -12.8 | +6 | MISS |
| NYK vs GSW | NYK 85% | HIGH | NYK 110-107 | OK | -8.6 | +3 | MISS |
| SAC vs UTA | SAC 82% | V.HIGH | SAC 116-111 | OK | -10.5 | +5 | MISS |

**Day result: ML 6-1 (85.7%) | ATS 3-4 (42.9%)**

## Key Observations

- **ML model is strong** (85.7%) — picks winners well
- **Spread model overestimates margins** — predicted large spreads but games were closer
- **Best strategy**: Flat ML on ALL picks, not just HIGH confidence
- **Biggest miss**: CLE vs DAL — model had CLE at 85% but DAL won by 10
- **No bets were placed** — Kelly filters rejected everything (odds parsing bug + overaggressive filters)
