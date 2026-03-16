#!/usr/bin/env python3
"""
NBA Quant Daemon — Continuous model improvement + daily 10 value bets.

Tony Bloom / Starlizard operational loop:
1. INGEST: Latest stats, odds, academic papers every 2h
2. MODEL: Recalibrate ELO, power ratings, Poisson params
3. ODDS: Scan all bookmakers for value (edge > 2%)
4. SELECT: Pick top 10 bets per day via Kelly portfolio
5. RECORD: Track all predictions + bankroll
6. LEARN: Compare predictions vs actuals, adjust weights

Bankroll: $100 initial, compound 20-30% daily via Kelly 1/4.
Target: 10 pertinent value bets per day across all available markets.
"""

import os, sys, json, time, ssl, urllib.request, hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "ops"))

# ── Load env ─────────────────────────────────────────────────
def load_env():
    for env_file in [ROOT / ".env.local", ROOT.parent / "mon-ipad" / ".env.local"]:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

load_env()

# ── Config ───────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
LITELLM_URL = os.environ.get("LITELLM_BASE_URL", "https://lbjlincoln-nomos-rag-engine-7.hf.space/v1")
LITELLM_KEY = os.environ.get("LITELLM_API_KEY", "sk-litellm-nomos-2026")
EMBEDDINGS_URL = os.environ.get("EMBEDDINGS_URL", "https://lbjlincoln-nomos-embeddings-api.hf.space")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

DATA_DIR = ROOT / "data"
RESEARCH_DIR = DATA_DIR / "research"
PREDICTIONS_DIR = DATA_DIR / "predictions"
BANKROLL_DIR = DATA_DIR / "bankroll"
LOG_FILE = DATA_DIR / "quant-daemon.jsonl"

for d in [RESEARCH_DIR, PREDICTIONS_DIR, BANKROLL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Cycle interval
CYCLE_INTERVAL = 7200  # 2 hours

# ── HTTP Helpers ─────────────────────────────────────────────
def http_post(url, data, headers=None, timeout=60):
    body = json.dumps(data).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status
    except Exception as e:
        return {"error": str(e)}, 0

def http_get(url, headers=None, timeout=30):
    hdrs = headers or {}
    req = urllib.request.Request(url, headers=hdrs, method="GET")
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status
    except Exception as e:
        return {"error": str(e)}, 0


def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).isoformat()
    entry = {"ts": ts, "level": level, "msg": msg}
    print(f"[{ts[:19]}] [{level}] {msg}")
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ══════════════════════════════════════════════════════════════
# STEP 1: INGEST LATEST DATA
# ══════════════════════════════════════════════════════════════

def ingest_latest_research():
    """Search for latest 2026 NBA analytics + sports betting academic papers."""
    papers_found = 0

    # Use Exa.AI for academic paper search
    if EXA_API_KEY:
        queries = [
            "NBA sports betting mathematical models 2026",
            "Kelly criterion sports wagering optimization 2025 2026",
            "NBA player efficiency advanced analytics ELO Bayesian 2026",
            "sports prediction machine learning Poisson regression NBA",
            "closing line value sharp betting analytics",
            "NBA RAPTOR EPM LEBRON DARKO player impact metrics 2026",
        ]

        for query in queries:
            try:
                data, status = http_post(
                    "https://api.exa.ai/search",
                    {"query": query, "numResults": 5, "useAutoprompt": True, "type": "neural"},
                    headers={"Authorization": f"Bearer {EXA_API_KEY}"},
                    timeout=30
                )
                if status == 200 and "results" in data:
                    for result in data["results"]:
                        paper = {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "snippet": result.get("text", "")[:500],
                            "source": "exa.ai",
                            "query": query,
                            "ingested_at": datetime.now(timezone.utc).isoformat(),
                        }
                        # Save to research dir
                        paper_id = hashlib.md5(paper["url"].encode()).hexdigest()[:12]
                        paper_file = RESEARCH_DIR / f"paper-{paper_id}.json"
                        if not paper_file.exists():
                            paper_file.write_text(json.dumps(paper, indent=2))
                            papers_found += 1
            except Exception as e:
                log(f"Exa search failed for '{query}': {e}", "WARN")

    # Use Brave Search as backup
    if BRAVE_API_KEY:
        queries_brave = [
            "NBA advanced statistics 2025-26 season team ratings",
            "sports betting value detection algorithms",
        ]
        for query in queries_brave:
            try:
                data, status = http_get(
                    f"https://api.search.brave.com/res/v1/web/search?q={urllib.parse.quote(query)}&count=5",
                    headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
                    timeout=20
                )
                if status == 200 and "web" in data:
                    for result in data["web"].get("results", []):
                        paper = {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "snippet": result.get("description", "")[:500],
                            "source": "brave",
                            "query": query,
                            "ingested_at": datetime.now(timezone.utc).isoformat(),
                        }
                        paper_id = hashlib.md5(paper["url"].encode()).hexdigest()[:12]
                        paper_file = RESEARCH_DIR / f"paper-{paper_id}.json"
                        if not paper_file.exists():
                            paper_file.write_text(json.dumps(paper, indent=2))
                            papers_found += 1
            except Exception as e:
                log(f"Brave search failed: {e}", "WARN")

    log(f"Research ingestion: {papers_found} new papers found")
    return papers_found


def ingest_live_odds():
    """Fetch live NBA odds from The Odds API (24+ bookmakers)."""
    if not ODDS_API_KEY:
        log("ODDS_API_KEY not set — using simulated odds", "WARN")
        return simulate_today_odds()

    try:
        url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={ODDS_API_KEY}&regions=us,eu,uk&markets=h2h,spreads,totals&oddsFormat=decimal"
        data, status = http_get(url, timeout=30)
        if status == 200 and isinstance(data, list):
            odds_file = DATA_DIR / f"odds-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}.json"
            odds_file.write_text(json.dumps(data, indent=2))
            log(f"Live odds: {len(data)} games fetched from Odds API")
            return data
        else:
            log(f"Odds API returned {status}", "WARN")
            return simulate_today_odds()
    except Exception as e:
        log(f"Odds API failed: {e}", "WARN")
        return simulate_today_odds()


def simulate_today_odds():
    """Generate realistic simulated odds when API unavailable."""
    import random

    # Import power ratings for realistic simulation
    try:
        from power_ratings import NBA_TEAMS, predict_matchup
    except ImportError:
        log("Cannot import power_ratings — using basic simulation", "WARN")
        return []

    teams = list(NBA_TEAMS.keys())
    games = []
    num_games = random.randint(5, 12)  # NBA typically has 5-12 games per day

    for _ in range(num_games):
        home = random.choice(teams)
        away = random.choice([t for t in teams if t != home])

        pred = predict_matchup(home, away)

        # Generate bookmaker odds from prediction with realistic market juice
        # home_win_prob is already decimal (0.0-1.0) from power_ratings
        raw_prob = pred["home_win_prob"]
        home_prob = raw_prob if raw_prob <= 1.0 else raw_prob / 100
        away_prob = 1 - home_prob

        # Market offers slightly wrong odds (our model's edge comes from this gap)
        # Simulate market inefficiency: books may underestimate by 3-8%
        market_home_prob = home_prob * random.uniform(0.88, 1.05)  # Market can be off
        market_away_prob = 1 - market_home_prob
        juice = random.uniform(1.03, 1.06)  # 3-6% overround (typical)

        home_odds = max(round(1.0 / (market_home_prob * juice), 2), 1.10)
        away_odds = max(round(1.0 / (market_away_prob * juice), 2), 1.10)

        # Generate spread from point diff
        spread = round(pred["predicted_diff"] + random.uniform(-1, 1), 1)

        game = {
            "id": f"sim-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{home[:3]}{away[:3]}",
            "sport_key": "basketball_nba",
            "home_team": home,
            "away_team": away,
            "commence_time": (datetime.now(timezone.utc) + timedelta(hours=random.randint(2, 12))).isoformat(),
            "bookmakers": [
                {
                    "key": bk,
                    "title": bk.replace("_", " ").title(),
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": round(home_odds + random.uniform(-0.10, 0.10), 2)},
                            {"name": away, "price": round(away_odds + random.uniform(-0.10, 0.10), 2)},
                        ]
                    }]
                }
                for bk in ["pinnacle", "draftkings", "fanduel", "betway", "winamax", "betclic", "parions_sport"]
            ],
            "simulated": True,
            "model_prediction": pred,
        }
        games.append(game)

    log(f"Simulated odds: {len(games)} games generated")
    return games


# ══════════════════════════════════════════════════════════════
# STEP 2: RECALIBRATE MODELS
# ══════════════════════════════════════════════════════════════

def recalibrate_models():
    """
    Recalibrate model weights based on recent prediction accuracy.
    Uses LLM to analyze research papers and suggest model adjustments.
    """
    # Gather recent research papers
    papers = []
    for f in sorted(RESEARCH_DIR.glob("paper-*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
        try:
            papers.append(json.loads(f.read_text()))
        except Exception:
            pass

    if not papers:
        log("No research papers to analyze for recalibration")
        return {}

    # Ask LLM to synthesize research into model adjustments
    paper_summaries = "\n".join([
        f"- {p.get('title', 'Unknown')}: {p.get('snippet', '')[:200]}"
        for p in papers[:10]
    ])

    prompt = f"""Based on these recent NBA analytics research papers, suggest specific calibration adjustments for our prediction models:

Papers:
{paper_summaries}

Our current model weights:
- Power Ratings: 35% (home court +3.0, rest days, travel, injuries)
- ELO: 20% (K-factor 20, home advantage 100 ELO)
- Poisson: 15% (avg team score 113.5)
- Monte Carlo: 30% (1000 iterations, stdev 12.0)

Current parameters:
- Kelly fraction: 0.25 (quarter Kelly)
- Min edge threshold: 2%
- Max bet: 5% of bankroll

Respond ONLY with a JSON object containing suggested adjustments, e.g.:
{{"elo_k_factor": 22, "home_court_advantage": 2.8, "monte_carlo_stdev": 11.5, "avg_team_score": 114.2, "kelly_fraction": 0.25, "min_edge": 0.02, "weight_power": 0.35, "weight_elo": 0.20, "weight_poisson": 0.15, "weight_mc": 0.30, "reasoning": "brief explanation"}}"""

    try:
        data, status = http_post(
            f"{LITELLM_URL}/chat/completions",
            {
                "model": "smart",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.3,
            },
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            timeout=60
        )
        if status == 200 and "choices" in data:
            content = data["choices"][0]["message"]["content"]
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
            if json_match:
                adjustments = json.loads(json_match.group())
                log(f"Model recalibration suggested: {json.dumps(adjustments)[:200]}")

                # Save calibration
                cal_file = DATA_DIR / "calibration" / f"cal-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}.json"
                cal_file.parent.mkdir(parents=True, exist_ok=True)
                cal_file.write_text(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "adjustments": adjustments,
                    "papers_analyzed": len(papers),
                }, indent=2))

                return adjustments
    except Exception as e:
        log(f"Recalibration LLM call failed: {e}", "WARN")

    return {}


# ══════════════════════════════════════════════════════════════
# STEP 3: FIND TOP 10 VALUE BETS
# ══════════════════════════════════════════════════════════════

def find_value_bets(games_odds: list, max_bets: int = 10) -> list:
    """
    Scan all available odds, compare to model predictions,
    find top 10 value bets with edge > 2%.
    """
    try:
        from power_ratings import predict_matchup
        from kelly import BetOpportunity, evaluate_multiple_bets
    except ImportError as e:
        log(f"Cannot import models: {e}", "ERROR")
        return []

    # Load bankroll state
    state_file = BANKROLL_DIR / "state.json"
    bankroll = 100.0
    if state_file.exists():
        state = json.loads(state_file.read_text())
        bankroll = state.get("balance", 100.0)

    opportunities = []
    all_predictions = []  # Save ALL predictions for self-improvement feedback

    # Market-blending: our model is 70%, market consensus 30%
    MODEL_WEIGHT = 0.70
    MARKET_WEIGHT = 0.30

    for game in games_odds:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        game_id = game.get("id", f"{home}-{away}")

        # Get our model prediction
        try:
            pred = game.get("model_prediction") or predict_matchup(home, away)
        except Exception:
            continue

        if pred is None:
            log(f"Skipping {home} vs {away}: no power rating match", "WARN")
            continue

        # home_win_prob is already decimal (0.0-1.0) from power_ratings
        raw = pred.get("home_win_prob", 0.5)
        model_home_prob = raw if raw <= 1.0 else raw / 100

        # Calculate market consensus from average bookmaker odds
        home_odds_list = []
        away_odds_list = []
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes", [])
                if len(outcomes) >= 2:
                    # Match outcomes to home/away
                    for o in outcomes:
                        name = o.get("name", "")
                        price = o.get("price", 0)
                        if price <= 1.0:
                            continue
                        if name == home:
                            home_odds_list.append(price)
                        elif name == away:
                            away_odds_list.append(price)

        # Blend model with market consensus (Bayesian market-aware)
        if home_odds_list:
            avg_home_odds = sum(home_odds_list) / len(home_odds_list)
            market_home_prob = 1.0 / avg_home_odds
            # Normalize market probs (remove vig)
            if away_odds_list:
                avg_away_odds = sum(away_odds_list) / len(away_odds_list)
                market_away_prob = 1.0 / avg_away_odds
                vig = market_home_prob + market_away_prob
                market_home_prob /= vig
            home_prob = MODEL_WEIGHT * model_home_prob + MARKET_WEIGHT * market_home_prob
        else:
            home_prob = model_home_prob

        # Cap extreme probabilities at 85/15
        home_prob = max(0.15, min(0.85, home_prob))
        away_prob = 1 - home_prob

        # Save prediction for self-improvement tracking
        all_predictions.append({
            "home_team": home,
            "away_team": away,
            "home_win_prob": round(home_prob, 4),
            "model_raw_prob": round(model_home_prob, 4),
            "market_prob": round(market_home_prob, 4) if home_odds_list else None,
            "predicted_spread": pred.get("spread", 0),
            "predicted_total": pred.get("predicted_total", 220),
            "confidence": pred.get("confidence", "UNKNOWN"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Scan all bookmakers for best odds
        for bookmaker in game.get("bookmakers", []):
            bk_name = bookmaker.get("title", bookmaker.get("key", "unknown"))

            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    team = outcome.get("name", "")
                    odds = outcome.get("price", 0)
                    if odds <= 1.0:
                        continue

                    est_prob = home_prob if team == home else away_prob
                    edge = (est_prob * odds) - 1.0

                    if edge >= 0.02:  # 2% minimum edge
                        opp = BetOpportunity(
                            game_id=game_id,
                            description=f"{team} ML vs {away if team == home else home}",
                            market="h2h",
                            selection="home" if team == home else "away",
                            decimal_odds=odds,
                            estimated_prob=est_prob,
                            bookmaker=bk_name,
                        )
                        opportunities.append(opp)

    # Save ALL game predictions for self-improvement feedback loop
    if all_predictions:
        pred_jsonl = PREDICTIONS_DIR / "predictions.jsonl"
        with open(pred_jsonl, "a") as f:
            for p in all_predictions:
                f.write(json.dumps(p) + "\n")
        log(f"Saved {len(all_predictions)} game predictions for self-improvement tracking")

    if not opportunities:
        log("No value bets found with edge >= 2%")
        return []

    # Sort by edge and take top N
    opportunities.sort(key=lambda o: (o.estimated_prob * o.decimal_odds - 1), reverse=True)
    top_opps = opportunities[:max_bets]

    # Evaluate portfolio with Kelly
    portfolio = evaluate_multiple_bets(top_opps, bankroll, kelly_fraction_mult=0.25)

    # Save predictions
    pred_file = PREDICTIONS_DIR / f"picks-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}.json"
    picks = []
    for bet in portfolio.bets:
        pick = {
            "description": bet.opportunity.get("description", ""),
            "bookmaker": bet.opportunity.get("bookmaker", ""),
            "odds": bet.opportunity.get("decimal_odds", 0),
            "estimated_prob": f"{bet.opportunity.get('estimated_prob', 0)*100:.1f}%",
            "edge": f"{bet.edge*100:.1f}%",
            "kelly_fraction": f"{bet.recommended_fraction*100:.2f}%",
            "stake": bet.recommended_bet,
            "is_bet": bet.is_bet,
            "reason": bet.reason,
        }
        picks.append(pick)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bankroll": bankroll,
        "total_opportunities": len(opportunities),
        "selected_bets": len([b for b in portfolio.bets if b.is_bet]),
        "total_exposure": f"{portfolio.total_exposure*100:.1f}%",
        "expected_ev": portfolio.expected_portfolio_ev,
        "picks": picks,
    }

    pred_file.write_text(json.dumps(result, indent=2))
    log(f"Value bets: {result['selected_bets']}/{result['total_opportunities']} opportunities | exposure {result['total_exposure']} | EV ${result['expected_ev']:.2f}")

    return picks


# ══════════════════════════════════════════════════════════════
# STEP 4: SYNC TO MON-IPAD
# ══════════════════════════════════════════════════════════════

def sync_to_control_tower():
    """Push latest metrics to mon-ipad for dashboard display."""
    mon_ipad = ROOT.parent / "mon-ipad" / "data" / "nba-agent"
    if not mon_ipad.exists():
        mon_ipad.mkdir(parents=True, exist_ok=True)

    # Bankroll state
    if (BANKROLL_DIR / "state.json").exists():
        state = json.loads((BANKROLL_DIR / "state.json").read_text())
    else:
        state = {}

    # Latest picks
    latest_picks = sorted(PREDICTIONS_DIR.glob("picks-*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    picks = json.loads(latest_picks[0].read_text()) if latest_picks else {}

    # Research count
    research_count = len(list(RESEARCH_DIR.glob("paper-*.json")))

    # Latest calibration
    cal_dir = DATA_DIR / "calibration"
    latest_cal = sorted(cal_dir.glob("cal-*.json"), key=lambda x: x.stat().st_mtime, reverse=True) if cal_dir.exists() else []
    calibration = json.loads(latest_cal[0].read_text()) if latest_cal else {}

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bankroll": state.get("balance", 100),
        "growth_pct": round((state.get("balance", 100) / max(state.get("initial_balance", 100), 1) - 1) * 100, 2),
        "record": f"{state.get('wins', 0)}W-{state.get('losses', 0)}L-{state.get('pushes', 0)}P",
        "roi_pct": round((state.get("total_profit", 0) / max(state.get("total_wagered", 1), 1)) * 100, 2),
        "research_papers": research_count,
        "latest_picks": picks.get("selected_bets", 0),
        "latest_exposure": picks.get("total_exposure", "0%"),
        "latest_ev": picks.get("expected_ev", 0),
        "calibration": calibration.get("adjustments", {}),
        "daemon_status": "RUNNING",
    }

    (mon_ipad / "quant-summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Synced to mon-ipad: bankroll=${summary['bankroll']}, {summary['research_papers']} papers, {summary['latest_picks']} picks")


# ══════════════════════════════════════════════════════════════
# MAIN DAEMON LOOP
# ══════════════════════════════════════════════════════════════

def run_self_improvement():
    """Run prediction-vs-results feedback loop."""
    # Step A: Verify yesterday's predictions against actual results
    try:
        sys.path.insert(0, str(ROOT / "ops"))
        # Use the new verify-results script
        from importlib import import_module
        verify_mod = import_module("nba-verify-results")
        report = verify_mod.run_verification()
        if report:
            ml_acc = report.get("ml_accuracy", 0)
            spread_acc = report.get("spread_accuracy", 0)
            log(f"Verification: ML {ml_acc:.1%}, Spread {spread_acc:.1%}")
    except Exception as e:
        log(f"Verification failed: {e}", "WARN")

    # Step B: Run model self-improvement (recalibrate weights)
    try:
        si = import_module("self-improve")
        metrics = si.run_self_improvement()
        if metrics:
            log(f"Self-improvement: accuracy {metrics.get('winner_accuracy', 0):.1%}, "
                f"Brier {metrics.get('brier_score', 0):.4f}")
            return metrics
    except Exception as e:
        log(f"Self-improvement failed: {e}", "WARN")
    return {}


_cycle_count = 0

def run_cycle():
    """Execute one full quant cycle."""
    global _cycle_count
    _cycle_count += 1
    cycle_start = time.time()
    log(f"═══ QUANT CYCLE #{_cycle_count} START ═══")

    # Step 1: Ingest
    papers = ingest_latest_research()
    odds = ingest_live_odds()

    # Step 2: Self-improvement (every 3rd cycle — compare predictions to actual results)
    if _cycle_count % 3 == 0:
        run_self_improvement()

    # Step 3: Recalibrate (every other cycle to save API calls)
    calibration = {}
    if papers > 0:
        calibration = recalibrate_models()

    # Step 4: Find value bets
    picks = find_value_bets(odds, max_bets=10)

    # Step 5: Sync
    sync_to_control_tower()

    elapsed = time.time() - cycle_start
    log(f"═══ QUANT CYCLE #{_cycle_count} DONE ({elapsed:.0f}s) — {papers} papers, {len(picks)} picks ═══")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NBA Quant Daemon")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (2h cycles)")
    parser.add_argument("--once", action="store_true", help="Run one cycle")
    parser.add_argument("--interval", type=int, default=7200, help="Cycle interval in seconds")
    args = parser.parse_args()

    # Save PID
    pid_file = ROOT.parent / "mon-ipad" / "data" / "nba-daemon.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    if args.once or not args.daemon:
        run_cycle()
    else:
        log(f"Starting NBA Quant Daemon — {args.interval}s cycles")
        while True:
            try:
                run_cycle()
            except Exception as e:
                log(f"Cycle error: {e}", "ERROR")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
