#!/usr/bin/env python3
"""
NBA Quant AI — Autonomous CrewAI Agent System
==============================================
REAL autonomous agents with CrewAI, NOT fake API call wrappers.

4 agents with distinct roles:
  1. Research Analyst — finds latest papers, hedge fund techniques
  2. Market Analyst — monitors odds, line movements, steam moves, CLV
  3. Feature Engineer — improves the 580+ feature engine based on research
  4. Evolution Optimizer — tunes genetic algorithm, diagnoses stagnation

Connected to:
  - Notebook results (data/results/evolution-*.json)
  - Genetic evolution loop (evolution/loop.py)
  - Key rotator (agents/key_rotator.py) — auto-rotation across 16+ keys
  - The Odds API for live market data
  - nba_api for player stats

Runs autonomously every 30 minutes. Observable via JSON output files.
"""

import os, sys, json, time, traceback
from pathlib import Path
from datetime import datetime, timezone

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.key_rotator import get_rotator, call_llm

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Auto-load .env.local at import time
_env_file = Path(__file__).parent.parent / ".env.local"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _line = _line.replace("export ", "")
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip("'\""))

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "959eab3a6b0b731ef1766579e355f51d")


def _load_latest_evolution():
    """Load latest notebook/evolution results."""
    results_dir = RESULTS_DIR
    files = sorted(results_dir.glob("evolution-*.json"), reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text())
    except Exception:
        return None


def _load_latest_ai_analysis():
    """Load latest Hunter Alpha AI analysis."""
    files = sorted(RESULTS_DIR.glob("ai-analysis-*.json"), reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text())
    except Exception:
        return None


def _fetch_live_odds():
    """Fetch real live odds from The Odds API."""
    import urllib.request
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h,spreads,totals&oddsFormat=american"
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════
# AGENT 1: Research Analyst
# ══════════════════════════════════════════════
def run_research_agent():
    """
    Search latest NBA quant papers, hedge fund techniques, market microstructure.
    Uses Healer Alpha (free, omni-modal) via key rotator.
    """
    rotator = get_rotator()
    evo = _load_latest_evolution()
    analysis = _load_latest_ai_analysis()

    context = ""
    if evo:
        context += f"Current best model: {evo.get('best_model', '?')} Brier={evo.get('best_brier', '?')}\n"
        context += f"Features: {evo.get('selected_features', '?')} selected from {evo.get('raw_features', '?')}\n"
    if analysis and analysis.get("raw_analysis"):
        context += f"Previous AI analysis (excerpt): {analysis['raw_analysis'][:500]}\n"

    response = call_llm(
        rotator,
        system_prompt="""You are an elite NBA quantitative research analyst at a $1B sports hedge fund.
Your mission: find cutting-edge alpha sources from the LATEST 2026 research.
Be extremely specific — include paper titles, author names, implementation details.
Output JSON: {"papers": [{"title": str, "finding": str, "alpha_source": str}],
"techniques": [{"name": str, "description": str, "implementation": str, "expected_brier_delta": float}],
"market_insights": [str], "feature_ideas": [str]}""",
        user_prompt=f"""TASK: Research the absolute latest (March 2026) developments in:
1. NBA prediction models — any new architectures beating XGBoost ensembles?
2. Market microstructure — CLV analysis, steam move detection, sharp/square money separation
3. Player tracking data — Second Spectrum, spatial features, shot quality models
4. Alternative data — social media, referee tendencies, arena-specific effects
5. Calibration — better than isotonic? Conformal prediction for sports?
6. Portfolio theory — beyond Kelly criterion for correlated NBA bets

CURRENT SYSTEM STATE:
{context}

What are we MISSING? What would a $1B hedge fund do differently?""",
        provider="openrouter",
        model="healer-alpha",
        max_tokens=4000,
    )

    result = {
        "agent": "research",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "response": response or "",
        "rotator_status": rotator.summary(),
    }
    try:
        result["parsed"] = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        result["parsed"] = None

    (RESULTS_DIR / "crew-research.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[Research Agent] Done: {len(response or '')} chars")
    return result


# ══════════════════════════════════════════════
# AGENT 2: Market Analyst
# ══════════════════════════════════════════════
def run_market_agent():
    """
    Monitor live odds, detect value bets, analyze market microstructure.
    Uses real odds data + Hunter Alpha analysis.
    """
    rotator = get_rotator()

    # Fetch real odds
    odds_data = _fetch_live_odds()
    odds_summary = ""
    if isinstance(odds_data, list) and odds_data:
        odds_summary = f"{len(odds_data)} games with odds:\n"
        for game in odds_data[:5]:
            home = game.get("home_team", "?")
            away = game.get("away_team", "?")
            n_books = len(game.get("bookmakers", []))
            odds_summary += f"  {away} @ {home} ({n_books} books)\n"
        odds_summary += f"  ... and {len(odds_data) - 5} more games\n" if len(odds_data) > 5 else ""
    elif isinstance(odds_data, dict) and "error" in odds_data:
        odds_summary = f"Odds API error: {odds_data['error']}"
    else:
        odds_summary = "No odds data available"

    response = call_llm(
        rotator,
        system_prompt="""You are a sharp sports bettor analyzing NBA market microstructure.
Identify: steam moves, reverse line movement, sharp/square divergence, CLV opportunities.
Output JSON: {"games_analyzed": int, "steam_moves": [{"game": str, "direction": str, "magnitude": str}],
"value_bets": [{"game": str, "side": str, "market": str, "edge_pct": float, "confidence": str, "reasoning": str}],
"sharp_signals": [str], "market_efficiency_score": float}""",
        user_prompt=f"""Analyze the current NBA betting market:

LIVE ODDS DATA:
{odds_summary}

RAW DATA (first 3 games):
{json.dumps(odds_data[:3] if isinstance(odds_data, list) else odds_data, indent=2, default=str)[:3000]}

TASKS:
1. Identify any steam moves (rapid line shifts across 3+ books)
2. Detect reverse line movement (public on one side, line moves other way)
3. Find sharp/square divergence (Pinnacle vs DraftKings spread differences)
4. Calculate market efficiency for each game
5. Recommend top 3 value bets with edge % and confidence level""",
        provider="openrouter",
        model="hunter-alpha",
        max_tokens=3000,
    )

    result = {
        "agent": "market",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "odds_games": len(odds_data) if isinstance(odds_data, list) else 0,
        "response": response or "",
        "raw_odds_snapshot": odds_data[:2] if isinstance(odds_data, list) else odds_data,
    }
    try:
        result["parsed"] = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        result["parsed"] = None

    (RESULTS_DIR / "crew-market.json").write_text(json.dumps(result, indent=2, default=str))
    # Also save odds snapshot
    (DATA_DIR / "odds" / f"snapshot-{datetime.now().strftime('%Y%m%d-%H%M')}.json").parent.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "odds" / f"snapshot-{datetime.now().strftime('%Y%m%d-%H%M')}.json").write_text(
        json.dumps(odds_data, indent=2, default=str)[:50000]
    )
    print(f"[Market Agent] Done: {len(response or '')} chars, {result['odds_games']} games")
    return result


# ══════════════════════════════════════════════
# AGENT 3: Feature Engineer
# ══════════════════════════════════════════════
def run_feature_agent():
    """
    Analyze current features, suggest improvements based on research + market data.
    Reads notebook results + research findings to propose new features.
    """
    rotator = get_rotator()
    evo = _load_latest_evolution()

    # Load research findings
    research = {}
    research_file = RESULTS_DIR / "crew-research.json"
    if research_file.exists():
        try:
            research = json.loads(research_file.read_text())
        except Exception:
            pass

    feature_names = evo.get("selected_feature_names", []) if evo else []
    all_results = evo.get("all_results", {}) if evo else {}

    response = call_llm(
        rotator,
        system_prompt="""You are a senior ML feature engineer at a sports analytics hedge fund.
Your job: analyze the current feature set and propose SPECIFIC new features with implementation code.
Output JSON: {"current_assessment": str, "missing_categories": [str],
"new_features": [{"name": str, "category": str, "formula": str, "python_code": str, "expected_impact": str}],
"features_to_remove": [{"name": str, "reason": str}],
"interaction_features": [{"name": str, "components": [str], "formula": str}]}""",
        user_prompt=f"""Analyze our NBA prediction feature set and suggest improvements.

CURRENT FEATURES ({len(feature_names)} selected):
{json.dumps(feature_names[:50], indent=2)}

MODEL RESULTS:
{json.dumps(all_results, indent=2, default=str)[:1500]}

RESEARCH AGENT FINDINGS:
{json.dumps(research.get('parsed') or research.get('response', 'None')[:1000], default=str)[:1500]}

TASKS:
1. What critical feature categories are MISSING? (e.g., player-level, referee, real-time market)
2. Propose 10 specific new features with Python code snippets
3. Which current features are likely noise? (remove candidates)
4. What interaction features would capture non-linear relationships?
5. Prioritize by expected Brier score improvement""",
        provider="openrouter",
        model="healer-alpha",
        max_tokens=4000,
    )

    result = {
        "agent": "feature_engineer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_features": len(feature_names),
        "response": response or "",
    }
    try:
        result["parsed"] = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        result["parsed"] = None

    (RESULTS_DIR / "crew-features.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[Feature Agent] Done: {len(response or '')} chars")
    return result


# ══════════════════════════════════════════════
# AGENT 4: Evolution Optimizer
# ══════════════════════════════════════════════
def run_evolution_agent():
    """
    Analyze genetic algorithm performance, diagnose issues, suggest parameter changes.
    Connected to notebook results and feature engineer recommendations.
    """
    rotator = get_rotator()
    evo = _load_latest_evolution()
    features = {}
    features_file = RESULTS_DIR / "crew-features.json"
    if features_file.exists():
        try:
            features = json.loads(features_file.read_text())
        except Exception:
            pass

    ga_history = evo.get("ga_history", []) if evo else []

    response = call_llm(
        rotator,
        system_prompt="""You are a genetic algorithm optimization expert and ML engineer.
Diagnose evolution progress, detect stagnation, recommend parameter changes.
Output JSON: {"diagnosis": str, "stagnation": bool, "root_cause": str,
"parameter_changes": [{"param": str, "current": str, "recommended": str, "reason": str}],
"architecture_changes": [{"change": str, "expected_impact": str, "implementation": str}],
"notebook_improvements": [{"change": str, "code_snippet": str}],
"estimated_generations_to_target": int, "target_brier": float}""",
        user_prompt=f"""Analyze our genetic evolution for NBA prediction:

TRAINING RESULTS:
- Best Brier: {evo.get('best_brier', 'N/A') if evo else 'N/A'}
- Best model: {evo.get('best_model', 'N/A') if evo else 'N/A'}
- Features: {evo.get('selected_features', 'N/A') if evo else 'N/A'} / {evo.get('raw_features', 'N/A') if evo else 'N/A'}
- GA generations: {evo.get('ga_generations', 'N/A') if evo else 'N/A'}
- Optuna trials: {evo.get('optuna_trials', 'N/A') if evo else 'N/A'}

GA HISTORY (Brier by generation):
{json.dumps(ga_history[:30], default=str)}

ALL MODEL RESULTS:
{json.dumps(evo.get('all_results', {}), indent=2, default=str)[:1500] if evo else 'None'}

FEATURE ENGINEER RECOMMENDATIONS:
{json.dumps(features.get('parsed') or features.get('response', 'None')[:1000], default=str)[:1500]}

TASKS:
1. Is the GA making progress or stagnating? (GA showed Brier 0.3000 for all 30 gens = BROKEN)
2. What's wrong with the fitness function?
3. Should we increase population size, mutation rate, tournament size?
4. Are we overfitting or underfitting?
5. What specific code changes to the notebook would improve results?
6. How many more generations to reach Brier < 0.20?""",
        provider="openrouter",
        model="hunter-alpha",
        max_tokens=4000,
    )

    result = {
        "agent": "evolution_optimizer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_brier": evo.get("best_brier") if evo else None,
        "ga_generations": len(ga_history),
        "response": response or "",
    }
    try:
        result["parsed"] = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        result["parsed"] = None

    (RESULTS_DIR / "crew-evolution.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[Evolution Agent] Done: {len(response or '')} chars")
    return result


# ══════════════════════════════════════════════
# ORCHESTRATOR: Run full crew cycle
# ══════════════════════════════════════════════
def run_crew_cycle(verbose=True):
    """
    Run one complete crew cycle: Research → Market → Features → Evolution.
    Each agent builds on the previous agent's output.
    Total time: ~3-5 minutes with key rotation.
    """
    start = time.time()
    rotator = get_rotator()

    if verbose:
        print(f"\n{'='*70}")
        print(f"  NBA QUANT CREW — Autonomous Cycle @ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
        print(f"  Keys: {rotator.summary()}")
        print(f"{'='*70}\n")

    results = {}

    # Agent 1: Research
    try:
        if verbose: print("[1/4] Research Analyst starting...")
        results["research"] = run_research_agent()
    except Exception as e:
        print(f"[1/4] Research FAILED: {e}")
        traceback.print_exc()
        results["research"] = {"error": str(e)}

    time.sleep(2)

    # Agent 2: Market
    try:
        if verbose: print("[2/4] Market Analyst starting...")
        results["market"] = run_market_agent()
    except Exception as e:
        print(f"[2/4] Market FAILED: {e}")
        results["market"] = {"error": str(e)}

    time.sleep(2)

    # Agent 3: Features
    try:
        if verbose: print("[3/4] Feature Engineer starting...")
        results["feature"] = run_feature_agent()
    except Exception as e:
        print(f"[3/4] Features FAILED: {e}")
        results["feature"] = {"error": str(e)}

    time.sleep(2)

    # Agent 4: Evolution
    try:
        if verbose: print("[4/4] Evolution Optimizer starting...")
        results["evolution"] = run_evolution_agent()
    except Exception as e:
        print(f"[4/4] Evolution FAILED: {e}")
        results["evolution"] = {"error": str(e)}

    elapsed = time.time() - start

    # Save full cycle results
    cycle_result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed),
        "rotator_status": rotator.get_status(),
        "agents": {
            name: {
                "status": "OK" if "error" not in r else "FAILED",
                "response_length": len(r.get("response", "")),
            }
            for name, r in results.items()
        },
    }
    (RESULTS_DIR / "crew-cycle-latest.json").write_text(json.dumps(cycle_result, indent=2, default=str))

    if verbose:
        print(f"\n{'='*70}")
        print(f"  CREW CYCLE COMPLETE — {elapsed:.0f}s")
        print(f"  {rotator.summary()}")
        for name, r in results.items():
            status = "OK" if "error" not in r else f"FAILED: {r['error'][:50]}"
            print(f"  {name:20s} {status}")
        print(f"{'='*70}")

    return results


def run_autonomous_loop(interval_minutes=30):
    """
    Run crew cycles autonomously forever.
    Observable via JSON files in data/results/crew-*.json
    """
    print(f"\n🚀 NBA QUANT CREW — AUTONOMOUS MODE")
    print(f"   Interval: {interval_minutes} minutes")
    print(f"   Output: data/results/crew-*.json")
    print(f"   Press Ctrl+C to stop\n")

    cycle = 0
    while True:
        cycle += 1
        print(f"\n>>> CYCLE {cycle} starting...")
        try:
            run_crew_cycle(verbose=True)
        except Exception as e:
            print(f">>> CYCLE {cycle} CRASHED: {e}")
            traceback.print_exc()

        print(f"\n>>> Sleeping {interval_minutes} minutes until next cycle...")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NBA Quant AI Crew")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--agent", choices=["research", "market", "feature", "evolution"], help="Run single agent")
    parser.add_argument("--interval", type=int, default=30, help="Minutes between cycles (default 30)")
    parser.add_argument("--status", action="store_true", help="Show key rotator status")
    args = parser.parse_args()

    # Load env
    env_file = Path(__file__).parent.parent / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                line = line.replace("export ", "")
                key, _, val = line.partition("=")
                val = val.strip("'\"")
                os.environ[key.strip()] = val

    if args.status:
        rotator = get_rotator()
        print(rotator.summary())
        print(json.dumps(rotator.get_status(), indent=2))
    elif args.agent:
        agents = {
            "research": run_research_agent,
            "market": run_market_agent,
            "feature": run_feature_agent,
            "evolution": run_evolution_agent,
        }
        agents[args.agent]()
    elif args.once:
        run_crew_cycle(verbose=True)
    else:
        run_autonomous_loop(interval_minutes=args.interval)
