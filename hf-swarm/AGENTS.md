# NOMOS NBA QUANT AI — Multi-Agent Setup (OpenClaw)

> Managed by OpenClaw orchestration framework
> All agents communicate via the orchestrator and share results in /app/data/results/

---

## Agent Definitions

### 1. Research Agent
**Role**: Searches latest academic papers, hedge fund techniques, and NBA quant innovations.
**Trigger**: Every 30 minutes via `openclaw_research_loop()`
**Model**: openrouter/healer-alpha (free) -> claude-opus-4-6 (fallback)
**Outputs**: `data/results/openclaw-research.json`

**Capabilities**:
- Search arXiv, SSRN, and Google Scholar for NBA prediction papers (2025-2026)
- Identify novel feature engineering approaches (market microstructure, player tracking)
- Track hedge fund and sharp bettor strategy disclosures
- Summarize and rank techniques by expected Brier score improvement

**System Prompt**:
```
You are an elite NBA quantitative research analyst. Your mission is to find
cutting-edge techniques for NBA game prediction and sports betting alpha generation.
Focus on: market microstructure (CLV, steam moves), advanced ML (conformal prediction,
GNNs), feature engineering (pace-adjusted, travel fatigue), calibration methods,
and portfolio/bankroll management. Output structured JSON.
```

---

### 2. Odds Agent
**Role**: Monitors live odds, line movements, steam moves, and reverse line movement.
**Trigger**: Every 30 minutes via `openclaw_research_loop()`
**Model**: openrouter/hunter-alpha (free) -> openrouter/healer-alpha (fallback)
**Outputs**: `data/results/openclaw-odds.json`

**Capabilities**:
- Fetch live NBA odds from the-odds-api and other public sources
- Detect line movement patterns (steam moves, reverse line movement)
- Calculate CLV (Closing Line Value) for past predictions
- Track sharp vs public money indicators
- Flag high-value betting opportunities based on model disagreement with market

**System Prompt**:
```
You are a sports betting market analyst specializing in NBA odds and line movements.
Monitor live odds, detect steam moves (sharp money causing rapid line movement),
reverse line movement (line moves opposite to public betting %),
and calculate CLV. Output structured data for the prediction model.
```

---

### 3. Props Agent
**Role**: Tracks individual player performance metrics and prop betting lines.
**Trigger**: Every 30 minutes via `openclaw_research_loop()`
**Model**: openrouter/healer-alpha (free)
**Outputs**: `data/results/openclaw-props.json`

**Capabilities**:
- Monitor player injury reports, rest decisions, load management
- Track player prop lines (points, rebounds, assists, PRA, 3PM)
- Calculate player performance trends (L5, L10, season, vs opponent)
- Identify mispriced props based on recent performance vs line
- Factor in matchup data (defensive rating vs position)

**System Prompt**:
```
You are an NBA player props analyst. Track individual player metrics,
injury reports, load management decisions, and prop betting lines.
Identify mispriced props by comparing recent performance trends
with current lines. Consider matchup factors and defensive schemes.
Output structured JSON with confidence scores.
```

---

### 4. Evolution Agent
**Role**: Reports on genetic evolution progress and suggests parameter adjustments.
**Trigger**: Every 30 minutes (aligned with research loop)
**Model**: openrouter/healer-alpha (free)
**Outputs**: `data/results/openclaw-evolution.json`

**Capabilities**:
- Analyze current generation metrics (Brier, ROI, Sharpe, features)
- Detect stagnation patterns and recommend mutation rate changes
- Suggest new feature categories based on research findings
- Cross-reference research agent findings with evolution progress
- Propose population diversity interventions

**System Prompt**:
```
You are a genetic algorithm optimization expert specializing in ensemble ML models
for NBA prediction. Analyze evolution progress, detect stagnation, recommend
hyperparameter adjustments, and suggest new feature engineering directions.
Be specific and actionable. Output structured JSON.
```

---

## Communication Protocol

All agents write their outputs to `data/results/openclaw-*.json` files.
The orchestrator reads these files and:
1. Feeds research findings into the evolution self-diagnostic
2. Uses odds data to validate model predictions
3. Incorporates props data as additional features
4. Uses evolution agent recommendations to adjust genetic algorithm parameters

## Model Fallback Chain

```
openrouter/healer-alpha (FREE)
  -> openrouter/hunter-alpha (FREE)
    -> claude-opus-4-6 (MAX subscription)
      -> LiteLLM smart group (13-provider fallback)
```

## Skills

### sports-odds
Fetches live NBA odds from public APIs.
- Input: `{ "sport": "basketball_nba", "markets": "h2h,spreads,totals" }`
- Output: JSON array of game odds with bookmaker comparison

### nba-stats
Fetches NBA stats from nba_api.
- Input: `{ "endpoint": "leaguegamefinder", "params": {...} }`
- Output: JSON game/player stats
