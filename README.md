# 🏀 Nomos NBA Agent — Expert IA Basketball

> Le meilleur agent IA NBA au monde. Stats avancees, paris sportifs, analytics, GOAT debates.

## 🚀 Demo Live

| Page | Lien | Description |
|------|------|-------------|
| **NBA Expert** | [nomos42.vercel.app/nba](https://nomos42.vercel.app/nba) | Chat expert NBA — 4 modes |
| **La Forge** | [nomos42.vercel.app/factory](https://nomos42.vercel.app/factory) | Generateur d'entreprise IA |
| **Dashboard** | [nomos42.vercel.app/dashboard](https://nomos42.vercel.app/dashboard) | Metriques live |
| **Marketplace** | [nomos42.vercel.app/marketplace](https://nomos42.vercel.app/marketplace) | Marketplace agents |
| **Valorisation** | [nomos42.vercel.app/valorisation](https://nomos42.vercel.app/valorisation) | Estimateur de valeur |
| **Vault** | [nomos42.vercel.app/vault](https://nomos42.vercel.app/vault) | Coffre-fort documents |
| **Graph** | [nomos42.vercel.app/graph](https://nomos42.vercel.app/graph) | Visualisation knowledge graph |
| **Satellite** | [nomos42.vercel.app/satellite](https://nomos42.vercel.app/satellite) | Vue satellite |
| **Casino** | [nomos42.vercel.app/casino](https://nomos42.vercel.app/casino) | Jeux IA |

## 🎯 4 Modes

- **Expert NBA** — Tout savoir sur la NBA : stats, histoire, joueurs, equipes
- **Paris Sportifs** — CLV, Kelly Criterion, Tony Bloom / Starlizard strategy
- **Analytics** — RAPTOR, EPM, LEBRON, DARKO, Second Spectrum tracking
- **GOAT Debates** — Comparaisons historiques era-adjusted

## 📊 Data Sources

- **balldontlie.io** — Free NBA stats API
- **The Odds API** — Cotes live de 24+ bookmakers (DraftKings, FanDuel, Pinnacle...)
- **Basketball Reference** — Stats historiques depuis 1946
- **Exa.AI** — Recherche web expert
- **Tony Bloom / Starlizard** — Methodologie sharp betting
- **Experts**: Zach Lowe, Ben Taylor, Seth Partnow, Nate Silver, Haralabos Voulgaris

## 🤖 Self-Testing

```bash
# Quick test (10 questions)
python3 agents/nba-agent.py --quick

# Full eval (48 questions, 6 categories)
python3 agents/nba-agent.py --eval

# Daemon mode (auto-test every 5min)
python3 agents/nba-agent.py --daemon 300

# Test suite (5 scenarios)
python3 tests/test-nba.py --all
```

**Current Accuracy: 100% (quick eval) | 68.8% (full test suite)**

## 🏗 Architecture

```
NBA Data Sources → Ingestion → LiteLLM (13 providers) → Expert Response
                                                      ↓
                                                 Eval Loop → Sync to mon-ipad
```

## 📝 Autres Repos

| Repo | Role |
|------|------|
| [mon-ipad](https://github.com/LBJLincoln/mon-ipad) | Tour de controle |
| [rag-website](https://github.com/LBJLincoln/rag-website) | Site web Next.js |
| [rag-data-ingestion](https://github.com/LBJLincoln/rag-data-ingestion) | Moteur d'ingestion |
| [nomos-forge-tests](https://github.com/LBJLincoln/nomos-forge-tests) | Tests autonomes Factory |
| [nomos-nba-agent](https://github.com/LBJLincoln/nomos-nba-agent) | **Ce repo** — Agent NBA |

---

*Built with Claude Code (Opus 4.6) + LiteLLM + Nomos AI*
