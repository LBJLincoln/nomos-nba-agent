# Nomos NBA Agent — Expert IA Basketball

> Repo autonome. Objectif : meilleur agent IA NBA au monde.

## MISSION

Agent IA expert NBA capable de repondre a TOUTE question sur le basketball NBA :
- Stats joueurs (RAPTOR, EPM, PER, WS, BPM, VORP)
- Historique equipes, saisons, playoffs, records
- Analyse tactique (lineups, pace, off/def rating)
- Paris sportifs (cotes, value bets, Kelly Criterion, CLV)
- Predictions (injuries, trades, draft)
- Comparaisons historiques (GOAT debates, era-adjusted stats)

## ARCHITECTURE

```
NBA Data Sources → Ingestion → Embeddings → Pinecone → RAG Pipeline → API → Factory
                                                    ↓
                                              Eval/Test Loop
```

### Data Sources
- **balldontlie.io** — Free NBA stats API (players, games, stats, averages)
- **nba_api** (Python) — Official NBA.com data (comprehensive)
- **Basketball Reference** — Historical stats (web scraping)
- **ESPN** — News, injury reports, analysis
- **Pinnacle/odds** — Betting lines, historical odds

### Stack
- **LLM**: LiteLLM S7 (`smart` model group, 13 providers)
- **Embeddings**: Jina v3 1024d (self-hosted HF Space)
- **Vector DB**: Pinecone `nomos-nba-jina-1024` (dedicated index)
- **Storage**: Supabase `nba_documents` table
- **Eval**: Autonomous eval loop with golden answers

## COMMANDS

```bash
source .env.local

# Agent
python3 agents/nba-agent.py                    # One-shot Q&A
python3 agents/nba-agent.py --daemon 300       # Continuous self-test

# Ingestion
python3 ops/ingest-nba.py --source all         # Ingest from all sources
python3 ops/ingest-nba.py --source balldontlie  # Single source

# Tests
python3 tests/test-nba.py --all                # Run all eval questions
python3 tests/test-nba.py --quick              # 10 smoke questions
python3 tests/test-nba.py --category stats     # Category-specific

# Ops
python3 ops/pilot.py                           # Receive commands from mon-ipad
python3 ops/sync.py                            # Push metrics to mon-ipad
```

## EVAL TARGETS

| Category | Questions | Target Accuracy | Current |
|----------|-----------|----------------|---------|
| Player Stats | 50 | >= 90% | - |
| Team History | 50 | >= 85% | - |
| Game Analysis | 30 | >= 80% | - |
| Betting/Odds | 30 | >= 75% | - |
| Predictions | 20 | >= 70% | - |
| GOAT Debates | 20 | >= 80% | - |

**Overall Target: >= 85% accuracy on 200 questions**

## RULES
1. source .env.local before any script
2. 1 fix per iteration
3. Measure before/after every change
4. Auto-stop on 3 consecutive failures
5. Push results to mon-ipad via ops/sync.py

## ⚠️ CRITICAL: COMPUTE RULES

### ZERO ML ON VM
La VM (1 vCPU / 969 MB RAM) ne peut PAS faire de ML.
**TOUT training, Optuna, backtest, karpathy-loop → HF Spaces (16GB RAM)**

| VM (autorise) | HF Spaces (ML ici) | Lightning/Colab (GPU) |
|---------------|--------------------|-----------------------|
| data-server | karpathy-loop | LSTM/Neural |
| quant-daemon (leger) | improve-loop | Large Optuna (100+ trials) |
| monitoring | backtest | MC Dropout |
| git, Claude Code | OddsHarvester | Deep learning |

### HF Spaces NBA
| Space | URL | Secrets |
|-------|-----|---------|
| nomos-nba-quant | lbjlincoln-nomos-nba-quant.hf.space | 101/101 |
| nomos-nba-quant-2 | lbjlincoln-nomos-nba-quant-2.hf.space | 101/101 |

**CHAQUE Space doit avoir les 101 secrets de .env.local**

### Models (9 + calibrated)
LR, RF, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost, Stacking, Meta-learner
All with isotonic calibration. Best: Brier 0.2034 | Acc 68.9%

### Data
9,551+ games, 8 seasons (2018-2026), 75 features (58 base + 17 advanced)
