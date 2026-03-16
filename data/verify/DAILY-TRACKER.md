# NBA Prediction Tracker — Daily Results

> Auto-updated by `ops/nba-verify-results.py`
> Last update: 2026-03-16

## Cumulative Record

| Metric | Record | Rate |
|--------|--------|------|
| **Moneyline** | 6-1 | **85.7%** |
| **Spread ATS** | 5-2 | **71.4%** |

## Bankroll Simulation ($100 start, flat $5 bets)

| Strategy | P/L | Balance | ROI |
|----------|-----|---------|-----|
| **Flat $5 ML (real market odds)** | **+$22.40** | **$122.40** | **+22.4%** |
| Flat $5 Spread ATS (real market lines, -110) | +$12.75 | $112.75 | +12.8% |

## Daily Breakdown

### 2026-03-15 — 7 games, ML 6-1, ATS 5-2

| Game | Agent Pick | Prob | Conf | ML Odds | Mkt Spread | Score | Margin | ML | ATS | ML P/L | ATS P/L |
|------|-----------|------|------|---------|------------|-------|--------|-----|-----|--------|---------|
| OKC vs MIN | OKC | 85% | HIGH | 1.28 | -8.5 | 116-103 | +13 | OK | OK | +$1.40 | +$4.55 |
| CLE vs DAL | CLE | 85% | HIGH | 1.10 | -15.5 | 120-130 | -10 | MISS | MISS | -$5.00 | -$5.00 |
| TOR vs DET | **TOR** | 60% | LOW | **2.38** | **+3.5** | 119-108 | +11 | OK | OK | **+$6.90** | +$4.55 |
| MIL vs IND | MIL | 72% | MED | 1.34 | -7.0 | 134-123 | +11 | OK | OK | +$1.70 | +$4.55 |
| PHI vs POR | **PHI** | 74% | V.HIGH | **3.70** | **+8.5** | 109-103 | +6 | OK | OK | **+$13.50** | +$4.55 |
| NYK vs GSW | NYK | 85% | HIGH | 1.12 | -14.0 | 110-107 | +3 | OK | MISS | +$0.60 | -$5.00 |
| SAC vs UTA | SAC | 82% | V.HIGH | 1.66 | -3.0 | 116-111 | +5 | OK | OK | +$3.30 | +$4.55 |
| | | | | | | | **TOTAL** | **6-1** | **5-2** | **+$22.40** | **+$12.75** |

### Spread note: GSW +14.0

The market had NYK -14.0. NYK only won by 3. Taking **GSW +14.0** at -110 was a massive cover (+$4.55).

## Value Picks Identified (but NOT bet)

The agent identified 2 **underdog winners** that the market had wrong:

| Pick | Agent Prob | Market Prob | ML Odds | Result | Profit on $5 | Why Not Bet |
|------|-----------|-------------|---------|--------|-------------|-------------|
| **PHI ML** | 74% | 27% | **3.70** | PHI 109-103 WIN | **+$13.50** | Kelly bug: "Probabilité estimée aberrante: 95.0%" |
| **TOR ML** | 60% | 42% | **2.38** | TOR 119-108 WIN | **+$6.90** | Not evaluated by Kelly at all |

These 2 bets alone = **+$20.40 profit** on $10 wagered (204% ROI).

## Agent Bugs to Fix

1. **Probability calibration**: Model outputs 95% instead of 74% for PHI → Kelly rejects as "aberrante"
2. **Odds parsing**: Some bookmakers return American odds (126, 46) treated as decimal → all rejected
3. **No post-game verification**: Scores file generated BEFORE games → no tracking
4. **Kelly too restrictive**: MIN_ODDS 1.20 filters out heavy favorites, MAX_ODDS 10.0 filters out valuable underdogs
5. **Spread bets not generated**: odds_analyzer.py has spread logic but Kelly only processes h2h
