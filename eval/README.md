# Eval — nomos-nba-agent

## What it tests
Quantitative sports betting models across 5 categories:
- **odds** — Decimal/American conversion, implied probability, EV, CLV, vig
- **kelly** — Full/fractional Kelly sizing, edge thresholds, portfolio allocation
- **power_rating** — Team matchup predictions, rest adjustments, home court
- **backtesting** — Strategy simulation, ROI, drawdown, calibration checks
- **risk_management** — Stop-loss, exposure caps, correlation, streak analysis

## How to run
```bash
source .env.local
python3 eval/run-eval.py                     # All categories
python3 eval/run-eval.py --category kelly    # Single category
python3 eval/run-eval.py --tolerance strict  # Tighter tolerance
```

## Target: 100K questions per pipeline
Current: 50 seed questions. Scale by generating parametric variants:
every team pair (30x29=870), every odds value (100 steps), every bankroll
level (20 steps) = millions of valid test combinations from 50 seeds.

## Adding new questions
Append to `eval-questions.json`. Required fields:
`id`, `category`, `input`, `expected_output`, `tolerance`
