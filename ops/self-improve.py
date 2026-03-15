#!/usr/bin/env python3
"""
Self-Improvement Loop — Track predictions vs actual results, recalibrate models.

Tony Bloom principle: "The model that learns fastest from its mistakes wins."

Flow:
1. Fetch completed NBA game scores (via free API)
2. Match against our predictions
3. Score: was our predicted winner correct? Was spread accurate?
4. Calculate calibration metrics (Brier score, log-loss, ROI)
5. Adjust ensemble weights based on which models performed best
6. Save performance history for long-term tracking
"""

import os, sys, json, ssl, urllib.request, math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

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

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

DATA_DIR = ROOT / "data"
PREDICTIONS_DIR = DATA_DIR / "predictions"
RESULTS_DIR = DATA_DIR / "results"
PERFORMANCE_DIR = DATA_DIR / "performance"
WEIGHTS_FILE = DATA_DIR / "ensemble-weights.json"

for d in [RESULTS_DIR, PERFORMANCE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")


def http_get(url, headers=None, timeout=30):
    hdrs = headers or {}
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status
    except Exception as e:
        return {"error": str(e)}, 0


# ══════════════════════════════════════════════════════════════
# STEP 1: FETCH COMPLETED GAME SCORES
# ══════════════════════════════════════════════════════════════

def fetch_completed_scores(days_back=3):
    """Fetch completed NBA game scores from The Odds API."""
    if not ODDS_API_KEY:
        print("[SELF-IMPROVE] No ODDS_API_KEY — cannot fetch scores")
        return []

    # Fetch scores for completed games
    url = (
        f"https://api.the-odds-api.com/v4/sports/basketball_nba/scores/"
        f"?apiKey={ODDS_API_KEY}"
        f"&daysFrom={days_back}"
        f"&dateFormat=iso"
    )

    data, status = http_get(url, timeout=15)
    if not isinstance(data, list):
        print(f"[SELF-IMPROVE] Scores API error: {data}")
        return []

    completed = [g for g in data if g.get("completed", False)]
    print(f"[SELF-IMPROVE] Fetched {len(completed)} completed games (last {days_back} days)")

    # Save results
    results_file = RESULTS_DIR / f"scores-{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    results_file.write_text(json.dumps(completed, indent=2))

    return completed


# ══════════════════════════════════════════════════════════════
# STEP 2: MATCH PREDICTIONS TO RESULTS
# ══════════════════════════════════════════════════════════════

def match_predictions_to_results(scores):
    """Match our predictions against actual game results."""
    # Load all recent predictions
    predictions = []
    for f in sorted(PREDICTIONS_DIR.glob("pred-*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:200]:
        try:
            predictions.append(json.loads(f.read_text()))
        except Exception:
            pass

    if not predictions:
        # Also check JSONL log
        jsonl_file = PREDICTIONS_DIR / "predictions.jsonl"
        if jsonl_file.exists():
            for line in jsonl_file.read_text().splitlines()[-200:]:
                try:
                    predictions.append(json.loads(line))
                except Exception:
                    pass

    if not predictions:
        print("[SELF-IMPROVE] No predictions found to evaluate")
        return []

    matched = []
    for score in scores:
        home = score.get("home_team", "")
        away = score.get("away_team", "")
        game_scores = score.get("scores", [])

        if not game_scores or len(game_scores) < 2:
            continue

        # Parse scores
        home_score = None
        away_score = None
        for s in game_scores:
            if s.get("name") == home:
                home_score = int(s.get("score", 0))
            elif s.get("name") == away:
                away_score = int(s.get("score", 0))

        if home_score is None or away_score is None:
            continue

        home_won = home_score > away_score
        margin = home_score - away_score
        total = home_score + away_score

        # Find matching prediction
        for pred in predictions:
            pred_home = pred.get("home_team", pred.get("home", ""))
            pred_away = pred.get("away_team", pred.get("away", ""))

            # Match by team abbreviation or full name
            if (_teams_match(pred_home, home) and _teams_match(pred_away, away)):
                home_prob = pred.get("ensemble_home_win_prob", pred.get("home_prob", 0.5))
                pred_spread = pred.get("predicted_spread", 0)
                pred_total = pred.get("predicted_total", 220)

                match = {
                    "game": f"{away} @ {home}",
                    "actual_home_score": home_score,
                    "actual_away_score": away_score,
                    "actual_margin": margin,
                    "actual_total": total,
                    "actual_home_won": home_won,
                    "predicted_home_prob": home_prob,
                    "predicted_spread": pred_spread,
                    "predicted_total": pred_total,
                    "predicted_winner_correct": (home_prob > 0.5) == home_won,
                    "spread_error": abs(margin - (-pred_spread)),
                    "total_error": abs(total - pred_total),
                    "brier_score": (home_prob - (1.0 if home_won else 0.0)) ** 2,
                    "confidence": pred.get("confidence", "UNKNOWN"),
                    "individual_models": pred.get("individual_models", {}),
                    "timestamp": pred.get("timestamp", pred.get("ts", "")),
                }
                matched.append(match)
                break

    print(f"[SELF-IMPROVE] Matched {len(matched)} predictions to results")
    return matched


def _teams_match(pred_name, api_name):
    """Fuzzy match team names."""
    if not pred_name or not api_name:
        return False
    pred_lower = pred_name.lower().strip()
    api_lower = api_name.lower().strip()
    if pred_lower == api_lower:
        return True
    # Check if abbreviation matches team name
    from power_ratings import NBA_TEAMS
    for abbrev, team in NBA_TEAMS.items():
        team_name_lower = team["name"].lower()
        if (pred_lower == abbrev.lower() or pred_lower == team_name_lower):
            if (api_lower == abbrev.lower() or api_lower == team_name_lower or
                team_name_lower.endswith(api_lower.split()[-1]) or
                api_lower.endswith(team_name_lower.split()[-1])):
                return True
    return False


# ══════════════════════════════════════════════════════════════
# STEP 3: CALCULATE PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════

def calculate_metrics(matched_results):
    """Calculate comprehensive performance metrics."""
    if not matched_results:
        return {}

    n = len(matched_results)
    correct = sum(1 for m in matched_results if m["predicted_winner_correct"])
    brier_scores = [m["brier_score"] for m in matched_results]
    spread_errors = [m["spread_error"] for m in matched_results]
    total_errors = [m["total_error"] for m in matched_results]

    # Log-loss
    log_losses = []
    for m in matched_results:
        p = m["predicted_home_prob"]
        y = 1.0 if m["actual_home_won"] else 0.0
        p_clamped = max(0.01, min(0.99, p))
        ll = -(y * math.log(p_clamped) + (1 - y) * math.log(1 - p_clamped))
        log_losses.append(ll)

    # Calibration by confidence bucket
    buckets = defaultdict(lambda: {"total": 0, "correct": 0, "avg_prob": 0})
    for m in matched_results:
        p = m["predicted_home_prob"]
        bucket = round(p * 10) / 10  # Round to nearest 10%
        buckets[bucket]["total"] += 1
        if m["actual_home_won"]:
            buckets[bucket]["correct"] += 1
        buckets[bucket]["avg_prob"] += p

    calibration = {}
    for bucket, data in sorted(buckets.items()):
        if data["total"] > 0:
            calibration[f"{bucket:.1f}"] = {
                "predicted": round(data["avg_prob"] / data["total"], 3),
                "actual": round(data["correct"] / data["total"], 3),
                "n": data["total"],
            }

    # Per-model accuracy (if individual model data available)
    model_accuracy = {}
    for model_name in ["power_ratings", "elo", "poisson", "monte_carlo"]:
        model_correct = 0
        model_total = 0
        for m in matched_results:
            models = m.get("individual_models", {})
            if model_name in models:
                model_prob = models[model_name].get("home_win_prob", 0.5)
                if (model_prob > 0.5) == m["actual_home_won"]:
                    model_correct += 1
                model_total += 1
        if model_total > 0:
            model_accuracy[model_name] = {
                "correct": model_correct,
                "total": model_total,
                "accuracy": round(model_correct / model_total, 4),
            }

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_size": n,
        "winner_accuracy": round(correct / n, 4),
        "brier_score": round(sum(brier_scores) / n, 4),
        "log_loss": round(sum(log_losses) / n, 4),
        "avg_spread_error": round(sum(spread_errors) / n, 1),
        "avg_total_error": round(sum(total_errors) / n, 1),
        "calibration": calibration,
        "model_accuracy": model_accuracy,
        "best_model": max(model_accuracy.items(), key=lambda x: x[1]["accuracy"])[0] if model_accuracy else "unknown",
        "worst_model": min(model_accuracy.items(), key=lambda x: x[1]["accuracy"])[0] if model_accuracy else "unknown",
    }

    return metrics


# ══════════════════════════════════════════════════════════════
# STEP 4: ADJUST ENSEMBLE WEIGHTS
# ══════════════════════════════════════════════════════════════

def adjust_ensemble_weights(metrics):
    """
    Adjust ensemble weights based on model performance.
    Models that perform better get more weight.
    Uses softmax of accuracy scores for smooth rebalancing.
    """
    model_acc = metrics.get("model_accuracy", {})
    if len(model_acc) < 2:
        print("[SELF-IMPROVE] Not enough model data to adjust weights")
        return None

    # Current weights
    current_weights = {
        "power_ratings": 0.35,
        "elo": 0.20,
        "poisson": 0.15,
        "monte_carlo": 0.30,
    }

    # Load saved weights if they exist
    if WEIGHTS_FILE.exists():
        try:
            saved = json.loads(WEIGHTS_FILE.read_text())
            current_weights = saved.get("weights", current_weights)
        except Exception:
            pass

    # Calculate new weights using softmax of accuracy
    accuracies = {}
    for model in current_weights:
        if model in model_acc:
            accuracies[model] = model_acc[model]["accuracy"]
        else:
            accuracies[model] = 0.5  # Default to coin flip if no data

    # Softmax with temperature (higher = smoother, less reactive)
    temperature = 2.0
    exp_vals = {k: math.exp(v / temperature) for k, v in accuracies.items()}
    total_exp = sum(exp_vals.values())
    new_weights = {k: round(v / total_exp, 4) for k, v in exp_vals.items()}

    # Blend: 70% current + 30% performance-adjusted (gradual adaptation)
    blended = {}
    for model in current_weights:
        blended[model] = round(0.70 * current_weights[model] + 0.30 * new_weights.get(model, 0.25), 4)

    # Normalize to sum to 1.0
    total = sum(blended.values())
    blended = {k: round(v / total, 4) for k, v in blended.items()}

    # Save
    weight_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": blended,
        "previous_weights": current_weights,
        "raw_performance": new_weights,
        "model_accuracies": accuracies,
        "adjustment_reason": f"Based on {metrics['sample_size']} game evaluations",
    }
    WEIGHTS_FILE.write_text(json.dumps(weight_data, indent=2))

    print(f"[SELF-IMPROVE] Weights adjusted:")
    for model, w in blended.items():
        old_w = current_weights.get(model, 0)
        delta = w - old_w
        print(f"  {model}: {old_w:.2%} → {w:.2%} ({delta:+.2%})")

    return blended


# ══════════════════════════════════════════════════════════════
# STEP 5: SAVE PERFORMANCE HISTORY
# ══════════════════════════════════════════════════════════════

def save_performance(metrics, matched_results):
    """Save performance snapshot for long-term tracking."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")

    # Save detailed results
    detail_file = PERFORMANCE_DIR / f"eval-{ts}.json"
    detail_file.write_text(json.dumps({
        "metrics": metrics,
        "results": matched_results,
    }, indent=2))

    # Append to history JSONL
    history_file = PERFORMANCE_DIR / "history.jsonl"
    with open(history_file, "a") as f:
        compact = {
            "ts": metrics["timestamp"],
            "n": metrics["sample_size"],
            "accuracy": metrics["winner_accuracy"],
            "brier": metrics["brier_score"],
            "log_loss": metrics["log_loss"],
            "spread_err": metrics["avg_spread_error"],
            "total_err": metrics["avg_total_error"],
            "best_model": metrics["best_model"],
        }
        f.write(json.dumps(compact) + "\n")

    # Sync to mon-ipad
    mon_ipad = ROOT.parent / "mon-ipad" / "data" / "nba-agent"
    if mon_ipad.exists():
        (mon_ipad / "eval-history.jsonl").write_text(
            history_file.read_text() if history_file.exists() else ""
        )
        (mon_ipad / "latest-eval.json").write_text(json.dumps(metrics, indent=2))

    print(f"[SELF-IMPROVE] Performance saved: accuracy {metrics['winner_accuracy']:.1%}, Brier {metrics['brier_score']:.4f}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run_self_improvement():
    """Full self-improvement cycle."""
    print("═══ SELF-IMPROVEMENT CYCLE START ═══")

    # 1. Fetch completed scores
    scores = fetch_completed_scores(days_back=3)
    if not scores:
        print("[SELF-IMPROVE] No completed scores available")
        return

    # 2. Match predictions to results
    matched = match_predictions_to_results(scores)
    if not matched:
        print("[SELF-IMPROVE] No predictions matched to results")
        return

    # 3. Calculate metrics
    metrics = calculate_metrics(matched)
    print(f"\n[PERFORMANCE] Winner accuracy: {metrics['winner_accuracy']:.1%}")
    print(f"[PERFORMANCE] Brier score: {metrics['brier_score']:.4f} (perfect=0, coin-flip=0.25)")
    print(f"[PERFORMANCE] Avg spread error: {metrics['avg_spread_error']:.1f} pts")
    print(f"[PERFORMANCE] Best model: {metrics['best_model']}")

    # 4. Adjust weights
    new_weights = adjust_ensemble_weights(metrics)

    # 5. Save performance
    save_performance(metrics, matched)

    print("═══ SELF-IMPROVEMENT CYCLE DONE ═══")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NBA Self-Improvement Loop")
    parser.add_argument("--once", action="store_true", help="Run one cycle")
    parser.add_argument("--daemon", action="store_true", help="Run every 6h")
    parser.add_argument("--days", type=int, default=3, help="Days back to check")
    parser.add_argument("--history", action="store_true", help="Show performance history")
    args = parser.parse_args()

    if args.history:
        history_file = PERFORMANCE_DIR / "history.jsonl"
        if history_file.exists():
            print("\nPerformance History:")
            print("=" * 70)
            for line in history_file.read_text().splitlines()[-20:]:
                entry = json.loads(line)
                print(f"  {entry['ts'][:16]} | Acc {entry['accuracy']:.1%} | Brier {entry['brier']:.4f} | "
                      f"Spread err {entry['spread_err']:.1f} | Best: {entry['best_model']}")
        else:
            print("No performance history yet")
    elif args.daemon:
        import time
        print("[SELF-IMPROVE] Starting daemon — 6h cycles")
        while True:
            try:
                run_self_improvement()
            except Exception as e:
                print(f"[ERROR] {e}")
            time.sleep(21600)  # 6 hours
    else:
        run_self_improvement()
