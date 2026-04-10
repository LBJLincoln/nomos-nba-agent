#!/usr/bin/env python3
"""
NBA Prediction Evaluator — Compare predictions to actual results.

Runs daily to:
  1. Fetch actual NBA game results (from ESPN/NBA API)
  2. Match against stored predictions in nba_predictions
  3. Compute Brier score, accuracy, ROI (using real market odds)
  4. Store daily metrics in nba_daily_eval
  5. Update nba_predictions rows with actual_home_win

Usage:
  python3 evaluate_predictions.py                  # Evaluate yesterday
  python3 evaluate_predictions.py --date 2026-03-18  # Evaluate specific date
"""

import os, sys, json, ssl, math, time
from pathlib import Path
from datetime import datetime, timezone, timedelta, date
from typing import Dict, List, Optional

# ── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Load .env.local ──────────────────────────────────────────────────────────
def _load_env():
    for env_path in [ROOT / ".env.local", Path("/home/termius/mon-ipad/.env.local")]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

_load_env()

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

def _get_conn():
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("[EVAL] ERROR: DATABASE_URL not set")
        sys.exit(1)
    import psycopg2
    return psycopg2.connect(db_url, options="-c search_path=public")


# ══════════════════════════════════════════════════════════════════════════════
# FETCH ACTUAL RESULTS — ESPN Scoreboard API (free, no key needed)
# ══════════════════════════════════════════════════════════════════════════════

# NBA team name normalization (ESPN name → our abbreviation)
ESPN_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def fetch_espn_scores(eval_date: str) -> List[Dict]:
    """Fetch completed game scores from ESPN for a given date.

    Returns list of {home_team, away_team, home_score, away_score, home_win}
    """
    import urllib.request

    # ESPN scoreboard API — free, no key
    date_compact = eval_date.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_compact}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Nomos42-NBA/1.0"})
        with urllib.request.urlopen(req, timeout=15, context=SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"[EVAL] ESPN fetch failed: {e}")
        return []

    results = []
    events = data.get("events", [])

    for event in events:
        status = event.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue  # Skip in-progress or postponed games

        competitors = event.get("competitions", [{}])[0].get("competitors", [])
        if len(competitors) != 2:
            continue

        home = away = None
        home_score = away_score = 0

        for c in competitors:
            team_name = c.get("team", {}).get("displayName", "")
            abbr = ESPN_TO_ABBR.get(team_name, c.get("team", {}).get("abbreviation", ""))
            score = int(c.get("score", "0"))

            if c.get("homeAway") == "home":
                home = abbr
                home_score = score
            else:
                away = abbr
                away_score = score

        if home and away:
            results.append({
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "home_win": home_score > away_score,
            })

    print(f"[EVAL] ESPN: {len(results)} completed games on {eval_date}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE — Match predictions to results, compute metrics
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_date(eval_date: str):
    """Evaluate predictions for a specific date against actual results."""

    # 1. Fetch actual results
    results = fetch_espn_scores(eval_date)
    if not results:
        print(f"[EVAL] No completed games found for {eval_date}")
        return None

    # Build lookup: (home, away) → result
    result_map = {}
    for r in results:
        result_map[(r["home_team"], r["away_team"])] = r

    # 2. Fetch our predictions from Supabase
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, home_team, away_team, predicted_home_prob,
                       market_home_prob, market_odds_home, market_odds_away,
                       edge, confidence, model_version
                FROM nba_predictions
                WHERE game_date = %s AND actual_home_win IS NULL
            """, (eval_date,))
            predictions = cur.fetchall()
    except Exception as e:
        print(f"[EVAL] Error fetching predictions: {e}")
        conn.close()
        return None

    if not predictions:
        print(f"[EVAL] No unevaluated predictions found for {eval_date}")
        conn.close()
        return None

    print(f"[EVAL] Matching {len(predictions)} predictions against {len(results)} results")

    # 3. Match and compute metrics
    matched = 0
    correct = 0
    brier_sum = 0.0
    profit = 0.0
    n_bets = 0
    details = []

    try:
        with conn.cursor() as cur:
            for pred in predictions:
                pid, home, away, home_prob, market_prob, odds_h, odds_a, edge, conf, model = pred

                # Try to match with actual result
                result = result_map.get((home, away))
                if not result:
                    # Try fuzzy match — sometimes abbreviations differ
                    for key, r in result_map.items():
                        if key[0] == home or key[1] == away:
                            result = r
                            break

                if not result:
                    continue

                matched += 1
                actual_home_win = result["home_win"]
                actual = 1.0 if actual_home_win else 0.0

                # Accuracy
                predicted_home = home_prob > 0.5
                if predicted_home == actual_home_win:
                    correct += 1

                # Brier score
                brier = (home_prob - actual) ** 2
                brier_sum += brier

                # ROI using real market odds (if available)
                if market_prob and market_prob > 0:
                    # Bet on home if model edge > 5%
                    model_edge = home_prob - market_prob
                    if abs(model_edge) > 0.05:
                        n_bets += 1
                        bet_on_home = model_edge > 0
                        if bet_on_home:
                            decimal_odds = 1.0 / market_prob if market_prob > 0 else 2.0
                            if actual_home_win:
                                profit += (decimal_odds - 1)  # Win
                            else:
                                profit -= 1  # Lose
                        else:
                            decimal_odds = 1.0 / (1 - market_prob) if market_prob < 1 else 2.0
                            if not actual_home_win:
                                profit += (decimal_odds - 1)  # Win
                            else:
                                profit -= 1  # Lose

                # Update prediction row with actual result
                cur.execute("""
                    UPDATE nba_predictions
                    SET actual_home_win = %s, evaluated_at = NOW()
                    WHERE id = %s
                """, (actual_home_win, pid))

                details.append({
                    "home": home, "away": away,
                    "predicted": round(home_prob, 3),
                    "market": round(market_prob, 3) if market_prob else None,
                    "actual_home_win": actual_home_win,
                    "correct": predicted_home == actual_home_win,
                    "brier": round(brier, 4),
                    "score": f"{result['home_score']}-{result['away_score']}",
                })

            conn.commit()
    except Exception as e:
        print(f"[EVAL] Error during evaluation: {e}")
        conn.rollback()
        conn.close()
        return None

    if matched == 0:
        print(f"[EVAL] No predictions matched actual games")
        conn.close()
        return None

    # 4. Compute aggregates
    accuracy = correct / matched
    brier_score = brier_sum / matched
    roi = (profit / n_bets * 100) if n_bets > 0 else 0.0

    eval_result = {
        "eval_date": eval_date,
        "total_games": matched,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "brier_score": round(brier_score, 4),
        "roi": round(roi, 2),
        "n_bets": n_bets,
        "profit": round(profit, 2),
        "details": details,
    }

    # 5. Store in nba_daily_eval
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO nba_daily_eval
                    (eval_date, total_games, correct, accuracy, brier_score, roi, details)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (eval_date, matched, correct, accuracy, brier_score, roi,
                  json.dumps(details, default=str)))
            conn.commit()
        print(f"[EVAL] Stored daily eval in nba_daily_eval")
    except Exception as e:
        print(f"[EVAL] Error storing eval: {e}")
        conn.rollback()
    finally:
        conn.close()

    # 6. Print results
    print(f"\n{'='*60}")
    print(f"  EVALUATION — {eval_date}")
    print(f"{'='*60}")
    print(f"  Games evaluated: {matched}")
    print(f"  Correct:         {correct}/{matched} ({accuracy:.1%})")
    print(f"  Brier Score:     {brier_score:.4f}")
    print(f"  Bets placed:     {n_bets}")
    print(f"  ROI:             {roi:+.1f}%")
    print(f"  Profit:          {profit:+.2f} units")
    print(f"{'='*60}\n")

    for d in details:
        check = "✓" if d["correct"] else "✗"
        print(f"  {check} {d['away']} @ {d['home']}  "
              f"pred={d['predicted']:.0%}  actual={'HOME' if d['actual_home_win'] else 'AWAY'}  "
              f"brier={d['brier']:.4f}  score={d['score']}")

    return eval_result


# ══════════════════════════════════════════════════════════════════════════════
# BACKFILL — Evaluate all dates with stored but unevaluated predictions
# ══════════════════════════════════════════════════════════════════════════════

def backfill_evaluations():
    """Find all dates with unevaluated predictions and evaluate them."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT game_date FROM nba_predictions
                WHERE actual_home_win IS NULL AND game_date < CURRENT_DATE
                ORDER BY game_date
            """)
            dates = [row[0].isoformat() for row in cur.fetchall()]
    finally:
        conn.close()

    if not dates:
        print("[EVAL] No unevaluated past predictions found")
        return

    print(f"[EVAL] Backfilling {len(dates)} dates: {dates}")
    for d in dates:
        evaluate_date(d)
        time.sleep(1)  # Rate limit ESPN


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate NBA predictions against actual results")
    parser.add_argument("--date", type=str, default=None, help="Date to evaluate (YYYY-MM-DD). Default: yesterday")
    parser.add_argument("--backfill", action="store_true", help="Evaluate all unevaluated past predictions")
    args = parser.parse_args()

    if args.backfill:
        backfill_evaluations()
    else:
        eval_date = args.date or (date.today() - timedelta(days=1)).isoformat()
        evaluate_date(eval_date)


if __name__ == "__main__":
    main()
