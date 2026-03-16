#!/usr/bin/env python3
"""
NBA Verify Results — Post-game prediction accuracy tracker.

Fetches actual NBA scores, compares to predictions, calculates:
- Moneyline accuracy
- Spread ATS accuracy
- Over/Under accuracy
- Bankroll simulation (flat bet + Kelly)
- Cumulative performance tracking

Usage:
    python3 ops/nba-verify-results.py                    # Verify latest day
    python3 ops/nba-verify-results.py --date 2026-03-15  # Specific date
    python3 ops/nba-verify-results.py --all              # All historical
    python3 ops/nba-verify-results.py --daemon 3600      # Every hour
"""

import os, sys, json, ssl, time, argparse, hashlib
import urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PREDICTIONS_DIR = DATA / "predictions"
RESULTS_DIR = DATA / "results"
VERIFY_DIR = DATA / "verify"
VERIFY_DIR.mkdir(parents=True, exist_ok=True)

# ── Load env ──────────────────────────────────────────────────────────────────
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
            break

load_env()

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


# ══════════════════════════════════════════════════════════════════════════════
# FETCH SCORES
# ══════════════════════════════════════════════════════════════════════════════

def fetch_scores(days_from=3):
    """Fetch completed NBA scores from The Odds API."""
    if not ODDS_API_KEY:
        print("[ERROR] ODDS_API_KEY not set")
        return []

    url = (
        f"https://api.the-odds-api.com/v4/sports/basketball_nba/scores/"
        f"?apiKey={ODDS_API_KEY}&daysFrom={days_from}&dateFormat=iso"
    )

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
            completed = [g for g in data if g.get("completed")]
            print(f"[OK] Fetched {len(completed)} completed games ({len(data)} total)")
            return completed
    except Exception as e:
        print(f"[ERROR] Fetch scores failed: {e}")
        return []


def parse_game_scores(games):
    """Parse API scores into a clean dict keyed by (home, away) tuple."""
    parsed = {}
    for g in games:
        home = g.get("home_team", "")
        scores = g.get("scores") or []
        if len(scores) < 2:
            continue

        score_map = {}
        teams = []
        for s in scores:
            name = s.get("name", "")
            score = s.get("score")
            if score is not None:
                score_map[name] = int(score)
                teams.append(name)

        away = [t for t in teams if t != home]
        if not away or home not in score_map:
            continue
        away = away[0]

        parsed[(home, away)] = {
            "home": home,
            "away": away,
            "home_score": score_map.get(home, 0),
            "away_score": score_map.get(away, 0),
            "commence_time": g.get("commence_time", ""),
            "id": g.get("id", ""),
        }
        # Also index by (away, home) for flexible matching
        parsed[(away, home)] = parsed[(home, away)]

    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# LOAD PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_predictions(target_date=None):
    """Load predictions from predictions.jsonl, optionally filtered by date."""
    pred_file = PREDICTIONS_DIR / "predictions.jsonl"
    if not pred_file.exists():
        print(f"[WARN] No predictions file: {pred_file}")
        return []

    preds = []
    with open(pred_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
                ts = p.get("timestamp", "")[:10]
                if target_date and ts != target_date:
                    continue
                preds.append(p)
            except json.JSONDecodeError:
                continue

    print(f"[OK] Loaded {len(preds)} predictions" + (f" for {target_date}" if target_date else ""))
    return preds


def load_picks_with_bets(target_date=None):
    """Load picks files to get bet recommendations with odds."""
    picks = []
    for f in sorted(PREDICTIONS_DIR.glob("picks-*.json")):
        if target_date:
            # picks-20260315-1125.json
            file_date = f.stem.split("-")[1][:8]  # 20260315
            fmt_date = f"{file_date[:4]}-{file_date[4:6]}-{file_date[6:8]}"
            if fmt_date != target_date:
                continue
        try:
            data = json.loads(f.read_text())
            picks.append(data)
        except Exception:
            continue

    if picks:
        print(f"[OK] Loaded {len(picks)} picks files")
    return picks


# ══════════════════════════════════════════════════════════════════════════════
# VERIFY & SCORE
# ══════════════════════════════════════════════════════════════════════════════

def verify_predictions(predictions, score_map):
    """Compare predictions against actual results."""
    results = []

    for pred in predictions:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        home_prob = pred.get("home_win_prob", 0.5)
        pred_spread = pred.get("predicted_spread", 0)
        pred_total = pred.get("predicted_total", 0)
        confidence = pred.get("confidence", "UNKNOWN")

        # Try to match
        actual = score_map.get((home, away))
        if not actual:
            results.append({
                "home": home, "away": away, "status": "NO_RESULT",
                "home_prob": home_prob, "confidence": confidence,
                "pred_spread": pred_spread, "pred_total": pred_total,
            })
            continue

        home_score = actual["home_score"]
        away_score = actual["away_score"]
        actual_margin = home_score - away_score
        actual_total = home_score + away_score
        predicted_winner = home if home_prob > 0.5 else away
        actual_winner = home if home_score > away_score else away

        # ML
        ml_correct = predicted_winner == actual_winner

        # Spread ATS (negative spread = home favored)
        # Covers if actual margin beats the spread
        if pred_spread < 0:
            spread_covers = actual_margin > abs(pred_spread)
        else:
            spread_covers = actual_margin < -abs(pred_spread)

        # Over/Under
        over = actual_total > pred_total

        results.append({
            "home": home,
            "away": away,
            "status": "VERIFIED",
            "home_prob": home_prob,
            "confidence": confidence,
            "predicted_winner": predicted_winner,
            "actual_winner": actual_winner,
            "home_score": home_score,
            "away_score": away_score,
            "actual_margin": actual_margin,
            "actual_total": actual_total,
            "pred_spread": pred_spread,
            "pred_total": pred_total,
            "ml_correct": ml_correct,
            "spread_covers": spread_covers,
            "total_over": over,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# BANKROLL SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate_bankroll(verified, initial=100.0):
    """Simulate bankroll with multiple strategies."""
    strategies = {}

    # --- Strategy 1: Flat $5 ML on HIGH+ confidence ---
    b1 = initial
    s1_bets = []
    for v in verified:
        if v["status"] != "VERIFIED":
            continue
        if v["confidence"] not in ("HIGH", "VERY HIGH"):
            continue

        stake = 5.0
        # Estimate decimal odds from our probability (with 7% vig)
        odds = max(1.10, (1.0 / v["home_prob"]) * 0.93) if v["predicted_winner"] == v["home"] else max(1.10, (1.0 / (1 - v["home_prob"])) * 0.93)

        if v["ml_correct"]:
            profit = stake * (odds - 1)
            b1 += profit
            s1_bets.append({"game": f"{v['home']} vs {v['away']}", "result": "WIN", "stake": stake, "odds": odds, "pnl": profit, "balance": b1})
        else:
            b1 -= stake
            s1_bets.append({"game": f"{v['home']} vs {v['away']}", "result": "LOSS", "stake": stake, "odds": odds, "pnl": -stake, "balance": b1})

    strategies["flat_ml_high"] = {"final": b1, "pnl": b1 - initial, "bets": s1_bets, "name": "Flat $5 ML (HIGH+ confidence)"}

    # --- Strategy 2: Flat $5 ML on ALL picks ---
    b2 = initial
    s2_bets = []
    for v in verified:
        if v["status"] != "VERIFIED":
            continue
        stake = 5.0
        odds = max(1.10, (1.0 / v["home_prob"]) * 0.93) if v["predicted_winner"] == v["home"] else max(1.10, (1.0 / (1 - v["home_prob"])) * 0.93)

        if v["ml_correct"]:
            profit = stake * (odds - 1)
            b2 += profit
            s2_bets.append({"game": f"{v['home']} vs {v['away']}", "result": "WIN", "stake": stake, "odds": odds, "pnl": profit, "balance": b2})
        else:
            b2 -= stake
            s2_bets.append({"game": f"{v['home']} vs {v['away']}", "result": "LOSS", "stake": stake, "odds": odds, "pnl": -stake, "balance": b2})

    strategies["flat_ml_all"] = {"final": b2, "pnl": b2 - initial, "bets": s2_bets, "name": "Flat $5 ML (ALL picks)"}

    # --- Strategy 3: Flat $5 Spread ATS (-110 / 1.91) ---
    b3 = initial
    s3_bets = []
    for v in verified:
        if v["status"] != "VERIFIED":
            continue
        stake = 5.0
        odds = 1.91  # standard -110

        if v["spread_covers"]:
            profit = stake * (odds - 1)
            b3 += profit
            s3_bets.append({"game": f"{v['home']} vs {v['away']}", "result": "COVER", "stake": stake, "spread": v["pred_spread"], "actual_margin": v["actual_margin"], "pnl": profit, "balance": b3})
        else:
            b3 -= stake
            s3_bets.append({"game": f"{v['home']} vs {v['away']}", "result": "MISS", "stake": stake, "spread": v["pred_spread"], "actual_margin": v["actual_margin"], "pnl": -stake, "balance": b3})

    strategies["flat_spread"] = {"final": b3, "pnl": b3 - initial, "bets": s3_bets, "name": "Flat $5 Spread ATS (-110)"}

    return strategies


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def print_report(verified, strategies, target_date):
    """Print a full verification report."""
    verified_games = [v for v in verified if v["status"] == "VERIFIED"]
    no_result = [v for v in verified if v["status"] == "NO_RESULT"]

    ml_correct = sum(1 for v in verified_games if v["ml_correct"])
    spread_correct = sum(1 for v in verified_games if v["spread_covers"])
    total = len(verified_games)

    print()
    print("=" * 90)
    print(f"  NBA PREDICTION VERIFICATION — {target_date or 'ALL'}")
    print("=" * 90)
    print()

    # Per-game results
    for v in verified:
        if v["status"] == "NO_RESULT":
            print(f"  [???] {v['home']} vs {v['away']} — NO RESULT AVAILABLE")
            continue

        ml_tag = " OK " if v["ml_correct"] else "MISS"
        sp_tag = " OK " if v["spread_covers"] else "MISS"
        ou_tag = "OVER" if v["total_over"] else "UNDR"

        print(f"  [{ml_tag}] {v['home']} vs {v['away']}")
        print(f"         Pred: {v['predicted_winner']} ({v['home_prob']:.0%}) [{v['confidence']}] | Spread {v['pred_spread']:+.1f} | O/U {v['pred_total']:.1f}")
        print(f"         Real: {v['actual_winner']} ({v['home_score']}-{v['away_score']}) | Margin {v['actual_margin']:+d} | Total {v['actual_total']}")
        print(f"         ML [{ml_tag}]  Spread [{sp_tag}]  O/U [{ou_tag}]")
        print()

    # Summary
    print("-" * 90)
    print(f"  MONEYLINE:    {ml_correct}/{total} ({ml_correct/total*100:.1f}%)" if total else "  No verified games")
    print(f"  SPREAD ATS:   {spread_correct}/{total} ({spread_correct/total*100:.1f}%)" if total else "")
    if no_result:
        print(f"  NO RESULT:    {len(no_result)} games (scores not yet available)")
    print()

    # Bankroll strategies
    print("=" * 90)
    print("  BANKROLL SIMULATION ($100 starting)")
    print("=" * 90)
    for key, strat in strategies.items():
        print()
        print(f"  --- {strat['name']} ---")
        for b in strat["bets"]:
            tag = b["result"]
            game = b["game"]
            pnl = b["pnl"]
            bal = b["balance"]
            extra = ""
            if "spread" in b:
                extra = f" | spread {b['spread']:+.1f} actual {b['actual_margin']:+d}"
            print(f"    {tag:5s} {game:50s} | ${b['stake']:.0f} | P/L {pnl:+.2f} | ${bal:.2f}{extra}")

        pnl = strat["pnl"]
        sign = "+" if pnl >= 0 else ""
        print(f"    RESULT: ${strat['final']:.2f} ({sign}${pnl:.2f})")

    print()
    print("=" * 90)

    return {
        "date": target_date,
        "predictions": len(verified),
        "verified": total,
        "no_result": len(no_result),
        "ml_accuracy": ml_correct / total if total else 0,
        "spread_accuracy": spread_correct / total if total else 0,
        "ml_record": f"{ml_correct}-{total - ml_correct}",
        "spread_record": f"{spread_correct}-{total - spread_correct}",
        "strategies": {k: {"pnl": v["pnl"], "final": v["final"]} for k, v in strategies.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# SAVE & CUMULATIVE TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def save_verification(report, verified):
    """Save verification results for cumulative tracking."""
    date = report.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    # Save daily report
    daily_file = VERIFY_DIR / f"verify-{date}.json"
    daily_file.write_text(json.dumps(report, indent=2, default=str))

    # Append to cumulative JSONL
    cumulative_file = VERIFY_DIR / "cumulative.jsonl"
    with open(cumulative_file, "a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            **report,
        }) + "\n")

    # Save detailed game-by-game results
    games_file = VERIFY_DIR / f"games-{date}.json"
    games_file.write_text(json.dumps(verified, indent=2, default=str))

    print(f"[OK] Saved: {daily_file.name}, {games_file.name}, cumulative.jsonl")


def print_cumulative_summary():
    """Print cumulative performance across all verified dates."""
    cumulative_file = VERIFY_DIR / "cumulative.jsonl"
    if not cumulative_file.exists():
        return

    records = []
    with open(cumulative_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        return

    total_verified = sum(r.get("verified", 0) for r in records)
    total_ml = sum(int(r.get("ml_record", "0-0").split("-")[0]) for r in records)
    total_spread = sum(int(r.get("spread_record", "0-0").split("-")[0]) for r in records)

    print()
    print("=" * 90)
    print(f"  CUMULATIVE PERFORMANCE ({len(records)} sessions, {total_verified} games)")
    print("=" * 90)
    if total_verified > 0:
        print(f"  ML accuracy:     {total_ml}/{total_verified} ({total_ml/total_verified*100:.1f}%)")
        print(f"  Spread accuracy: {total_spread}/{total_verified} ({total_spread/total_verified*100:.1f}%)")

    for key in ("flat_ml_high", "flat_ml_all", "flat_spread"):
        total_pnl = sum(r.get("strategies", {}).get(key, {}).get("pnl", 0) for r in records)
        print(f"  {key}: cumulative P/L ${total_pnl:+.2f}")

    print("=" * 90)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_verification(target_date=None):
    """Run full verification cycle."""
    if not target_date:
        target_date = (datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y-%m-%d")

    print(f"\n[+] Verifying predictions for {target_date}...")

    # 1. Fetch scores
    scores = fetch_scores(days_from=3)
    if not scores:
        print("[WARN] No scores fetched — skipping")
        return None

    score_map = parse_game_scores(scores)
    print(f"[OK] Parsed {len(score_map) // 2} unique games")

    # 2. Load predictions
    predictions = load_predictions(target_date)
    if not predictions:
        print(f"[WARN] No predictions found for {target_date}")
        return None

    # 3. Verify
    verified = verify_predictions(predictions, score_map)

    # 4. Simulate bankroll
    strategies = simulate_bankroll(verified)

    # 5. Display report
    report = print_report(verified, strategies, target_date)

    # 6. Save
    save_verification(report, verified)

    # 7. Cumulative
    print_cumulative_summary()

    return report


def main():
    parser = argparse.ArgumentParser(description="NBA Prediction Verification")
    parser.add_argument("--date", help="Date to verify (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="Verify all historical predictions")
    parser.add_argument("--daemon", type=int, help="Run every N seconds")
    parser.add_argument("--summary", action="store_true", help="Show cumulative summary only")
    args = parser.parse_args()

    if args.summary:
        print_cumulative_summary()
        return

    if args.daemon:
        print(f"[+] Daemon mode: verifying every {args.daemon}s")
        while True:
            try:
                run_verification(args.date)
            except Exception as e:
                print(f"[ERROR] {e}")
            print(f"\n[+] Next verification in {args.daemon}s...")
            time.sleep(args.daemon)
    elif args.all:
        # Find all unique dates in predictions
        pred_file = PREDICTIONS_DIR / "predictions.jsonl"
        if pred_file.exists():
            dates = set()
            with open(pred_file) as f:
                for line in f:
                    if line.strip():
                        p = json.loads(line)
                        dates.add(p.get("timestamp", "")[:10])
            for date in sorted(dates):
                run_verification(date)
    else:
        run_verification(args.date)


if __name__ == "__main__":
    main()
