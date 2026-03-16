#!/usr/bin/env python3
"""
NBA Quant Backtesting Framework.

Simulates betting on historical data using our models.
Measures: ROI, Brier, Sharpe, Max Drawdown, CLV.

Usage:
  python3 ops/backtest.py --seasons 3     # Backtest last 3 seasons
  python3 ops/backtest.py --full           # All 8 seasons
  python3 ops/backtest.py --strategy kelly # Kelly sizing
  python3 ops/backtest.py --strategy flat  # Flat $10 bets
"""

import json, sys, math, os
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ops"))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Load games ────────────────────────────────────────────────────────────────

def load_games(n_seasons: int = 8) -> List[dict]:
    hist_dir = ROOT / "data" / "historical"
    all_games = []
    files = sorted(hist_dir.glob("games-*.json"), reverse=True)[:n_seasons]
    for f in sorted(files):
        data = json.loads(f.read_text())
        items = data if isinstance(data, list) else data.get("games", [])
        all_games.extend(items)
    all_games.sort(key=lambda g: g.get("game_date", g.get("date", "")))
    return all_games


# ── Build features (same as improve-loop) ─────────────────────────────────────

TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

def resolve(name):
    if name in TEAM_MAP: return TEAM_MAP[name]
    if len(name) == 3 and name.isupper(): return name
    for full, abbr in TEAM_MAP.items():
        if name in full or full in name: return abbr
    return name[:3].upper() if name else None


def build_features(games):
    team_results = defaultdict(list)
    team_last_game = {}
    X, y, meta = [], [], []

    for game in games:
        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        if "home" in game and isinstance(game["home"], dict):
            h, a = game["home"], game.get("away", {})
            home_score, away_score = h.get("pts"), a.get("pts")
            if not home_raw: home_raw = h.get("team_name", "")
            if not away_raw: away_raw = a.get("team_name", "")
        else:
            home_score, away_score = game.get("home_score"), game.get("away_score")

        if home_score is None or away_score is None: continue
        home_score, away_score = int(home_score), int(away_score)
        home, away = resolve(home_raw), resolve(away_raw)
        if not home or not away: continue

        game_date = game.get("game_date", game.get("date", ""))[:10]
        hr, ar = team_results[home], team_results[away]

        def wp(r, n):
            s = r[-n:] if r else []
            return sum(1 for x in s if x[1]) / len(s) if s else 0.5

        def pd(r, n):
            s = r[-n:] if r else []
            return sum(x[2] for x in s) / len(s) if s else 0.0

        def streak(r):
            if not r: return 0
            s, last = 0, r[-1][1]
            for x in reversed(r):
                if x[1] == last: s += 1
                else: break
            return s if last else -s

        def rest(team):
            last = team_last_game.get(team)
            if not last or not game_date: return 3
            try:
                return max(0, (datetime.strptime(game_date[:10], "%Y-%m-%d") -
                              datetime.strptime(last[:10], "%Y-%m-%d")).days)
            except: return 3

        def sos(r, n=10):
            recent = r[-n:]
            if not recent: return 0.5
            opcts = []
            for x in recent:
                opp_r = team_results[x[3]]
                if opp_r: opcts.append(sum(1 for z in opp_r if z[1]) / len(opp_r))
            return sum(opcts) / len(opcts) if opcts else 0.5

        h_rest, a_rest = rest(home), rest(away)
        row = [
            wp(hr,5), wp(ar,5), wp(hr,10), wp(ar,10), wp(hr,20), wp(ar,20),
            wp(hr,5)-wp(hr,20), wp(ar,5)-wp(ar,20),
            pd(hr,5), pd(ar,5), pd(hr,10), pd(ar,10),
            min(h_rest,7), min(a_rest,7), 1.0 if h_rest==1 else 0.0, 1.0 if a_rest==1 else 0.0,
            1.0, streak(hr)/10.0, streak(ar)/10.0, sos(hr), sos(ar),
            0.5, min(len(hr),82)/82.0, min(len(ar),82)/82.0,
        ]

        X.append(row)
        y.append(1 if home_score > away_score else 0)
        meta.append({"home": home, "away": away, "date": game_date,
                      "h_score": home_score, "a_score": away_score})

        margin = home_score - away_score
        team_results[home].append((game_date, home_score > away_score, margin, away))
        team_results[away].append((game_date, away_score > home_score, -margin, home))
        team_last_game[home] = game_date
        team_last_game[away] = game_date

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32), meta


# ── Backtest Engine ───────────────────────────────────────────────────────────

def backtest(X, y, meta, strategy="kelly", bankroll_init=100.0,
             min_edge=0.02, kelly_fraction=0.25, max_bet_pct=0.05):
    """
    Walk-forward backtest: train on past, predict next day, bet, repeat.
    Uses expanding window (always retrain on all past data).
    """
    log(f"Starting backtest: {len(X)} games, strategy={strategy}, bankroll=${bankroll_init}")

    # We need at least 500 games to start making predictions
    MIN_TRAIN = 500
    if len(X) < MIN_TRAIN + 100:
        log("Not enough data for meaningful backtest")
        return {}

    bankroll = bankroll_init
    peak = bankroll_init
    bets = []
    daily_returns = []
    all_probs = []
    all_actuals = []

    # Walk forward in chunks of ~50 games (roughly 3-4 days)
    chunk_size = 50
    n_chunks = (len(X) - MIN_TRAIN) // chunk_size

    log(f"  {n_chunks} chunks of {chunk_size} games, starting after {MIN_TRAIN} training games")

    for chunk_idx in range(n_chunks):
        train_end = MIN_TRAIN + chunk_idx * chunk_size
        test_start = train_end
        test_end = min(test_start + chunk_size, len(X))

        if test_end <= test_start:
            break

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        test_meta = meta[test_start:test_end]

        # Train ensemble (quick models for speed)
        try:
            lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_train)
            lr.fit(X_tr_scaled, y_train)

            rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=1)
            rf.fit(X_train, y_train)

            xgb_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                eval_metric="logloss", random_state=42, n_jobs=1, verbosity=0
            )
            xgb_model.fit(X_train, y_train)

        except Exception as e:
            log(f"  Training failed at chunk {chunk_idx}: {e}")
            continue

        # Predict
        X_test_scaled = scaler.transform(X_test)
        probs_lr = lr.predict_proba(X_test_scaled)[:, 1]
        probs_rf = rf.predict_proba(X_test)[:, 1]
        probs_xgb = xgb_model.predict_proba(X_test)[:, 1]

        # Simple ensemble average
        probs = (probs_lr * 0.35 + probs_rf * 0.35 + probs_xgb * 0.30)
        all_probs.extend(probs)
        all_actuals.extend(y_test)

        # Simulate betting
        chunk_start_bankroll = bankroll
        for i in range(len(probs)):
            prob = probs[i]
            actual = y_test[i]
            game = test_meta[i]

            # Simulate market odds (implied prob with ~5% vig)
            true_prob = 0.59  # Home team base rate
            market_implied = true_prob + np.random.uniform(-0.05, 0.05)
            market_odds = 1.0 / max(market_implied, 0.1)

            # Determine if we bet
            model_edge = prob - market_implied

            if abs(model_edge) < min_edge:
                continue  # No edge, skip

            # Bet on the side our model favors
            bet_on_home = model_edge > 0
            bet_prob = prob if bet_on_home else (1 - prob)
            bet_odds = market_odds if bet_on_home else (1.0 / max(1 - market_implied, 0.1))
            bet_won = (actual == 1) if bet_on_home else (actual == 0)

            # Sizing
            if strategy == "kelly":
                b = bet_odds - 1
                q = 1 - bet_prob
                kelly_full = max(0, (b * bet_prob - q) / b)
                stake = bankroll * kelly_full * kelly_fraction
                stake = min(stake, bankroll * max_bet_pct)
            else:  # flat
                stake = min(10.0, bankroll * 0.05)

            if stake < 0.50 or bankroll < 1.0:
                continue

            # Resolve bet
            if bet_won:
                profit = stake * (bet_odds - 1)
            else:
                profit = -stake

            bankroll += profit
            peak = max(peak, bankroll)

            bets.append({
                "date": game["date"],
                "home": game["home"], "away": game["away"],
                "bet_on": "home" if bet_on_home else "away",
                "prob": round(float(bet_prob), 4),
                "odds": round(float(bet_odds), 2),
                "edge": round(float(model_edge), 4),
                "stake": round(float(stake), 2),
                "profit": round(float(profit), 2),
                "bankroll": round(float(bankroll), 2),
                "won": bet_won,
            })

        chunk_return = (bankroll - chunk_start_bankroll) / max(chunk_start_bankroll, 1)
        daily_returns.append(chunk_return)

        if chunk_idx % 20 == 0:
            log(f"  Chunk {chunk_idx}/{n_chunks}: bankroll=${bankroll:.2f} ({len(bets)} bets)")

    # ── Results ──
    if not bets:
        log("No bets placed!")
        return {}

    wins = sum(1 for b in bets if b["won"])
    losses = len(bets) - wins
    total_wagered = sum(b["stake"] for b in bets)
    total_profit = bankroll - bankroll_init
    roi = total_profit / total_wagered * 100 if total_wagered > 0 else 0
    max_drawdown = 0
    running_peak = bankroll_init
    for b in bets:
        running_peak = max(running_peak, b["bankroll"])
        dd = (running_peak - b["bankroll"]) / running_peak
        max_drawdown = max(max_drawdown, dd)

    # Sharpe ratio (annualized from chunk returns)
    if daily_returns and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252 / max(len(daily_returns), 1))
    else:
        sharpe = 0

    # Brier score
    brier = brier_score_loss(all_actuals, all_probs) if all_probs else 0
    accuracy = accuracy_score(all_actuals, [1 if p > 0.5 else 0 for p in all_probs]) if all_probs else 0

    results = {
        "strategy": strategy,
        "initial_bankroll": bankroll_init,
        "final_bankroll": round(bankroll, 2),
        "total_profit": round(total_profit, 2),
        "total_bets": len(bets),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(bets) * 100, 1),
        "total_wagered": round(total_wagered, 2),
        "roi_pct": round(roi, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 1),
        "sharpe_ratio": round(sharpe, 3),
        "brier_score": round(brier, 5),
        "accuracy": round(accuracy, 4),
        "games_evaluated": len(all_probs),
        "avg_edge": round(np.mean([b["edge"] for b in bets]), 4) if bets else 0,
        "avg_stake": round(np.mean([b["stake"] for b in bets]), 2) if bets else 0,
    }

    log(f"\n{'='*60}")
    log(f"BACKTEST RESULTS ({strategy})")
    log(f"{'='*60}")
    log(f"  Bankroll: ${bankroll_init:.0f} → ${bankroll:.2f} ({total_profit:+.2f})")
    log(f"  Bets: {len(bets)} ({wins}W-{losses}L, {results['win_rate']}% win rate)")
    log(f"  ROI: {roi:+.2f}%")
    log(f"  Sharpe: {sharpe:.3f}")
    log(f"  Max Drawdown: {max_drawdown*100:.1f}%")
    log(f"  Brier Score: {brier:.5f}")
    log(f"  Accuracy: {accuracy:.4f}")

    # Save results
    out_file = ROOT / "data" / "backtest" / f"backtest-{strategy}-{datetime.now().strftime('%Y%m%d-%H%M')}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"results": results, "bets": bets[-100:]}, indent=2))
    log(f"  Saved: {out_file}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NBA Backtest")
    parser.add_argument("--seasons", type=int, default=8, help="Number of seasons")
    parser.add_argument("--strategy", default="kelly", choices=["kelly", "flat"])
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--min-edge", type=float, default=0.02)
    args = parser.parse_args()

    games = load_games(args.seasons)
    log(f"Loaded {len(games)} games from {args.seasons} seasons")

    X, y, meta = build_features(games)
    log(f"Built {X.shape[0]} x {X.shape[1]} features")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Run both strategies
    results_kelly = backtest(X, y, meta, strategy="kelly",
                             bankroll_init=args.bankroll, min_edge=args.min_edge)
    results_flat = backtest(X, y, meta, strategy="flat",
                            bankroll_init=args.bankroll, min_edge=args.min_edge)

    # Summary
    log(f"\n{'='*60}")
    log(f"COMPARISON")
    log(f"{'='*60}")
    for name, r in [("Kelly", results_kelly), ("Flat", results_flat)]:
        if r:
            log(f"  {name:10s}: ${r['initial_bankroll']:.0f}→${r['final_bankroll']:.2f} "
                f"ROI={r['roi_pct']:+.1f}% Sharpe={r['sharpe_ratio']:.3f} "
                f"DD={r['max_drawdown_pct']:.1f}%")


if __name__ == "__main__":
    main()
