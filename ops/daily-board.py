#!/usr/bin/env python3
"""
NBA Quant Daily Board — Live Trading Dashboard
================================================
Displays everything a quant trader needs in one terminal view:

1. MODEL STATUS: Training metrics, feature importance, Brier scores
2. TODAY'S GAMES: Schedule with tip-off times
3. ODDS COMPARISON: All bookmakers side-by-side (FR + US)
4. VALUE BETS: Recommended bets with Kelly sizing and bankroll allocation
5. RECENT P&L: Last 7 days performance tracking
6. ELO RANKINGS: Current power rankings

Usage:
  python3 ops/daily-board.py              # One-shot display
  python3 ops/daily-board.py --loop 300   # Refresh every 5 min
  python3 ops/daily-board.py --compact    # Compact mode for small terminals
"""

import os, sys, json, time, ssl, urllib.request, math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "ops"))

# Load env
for env_file in [ROOT / ".env.local", Path("/home/termius/mon-ipad/.env.local")]:
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

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

DATA_DIR = ROOT / "data"
PREDICTIONS_DIR = DATA_DIR / "predictions"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models"
TRAINING_DIR = DATA_DIR / "training"
HISTORICAL_DIR = DATA_DIR / "historical"
BANKROLL_DIR = DATA_DIR / "bankroll"

from power_ratings import NBA_TEAMS, get_team, predict_matchup

# ── Terminal colors ──────────────────────────────────────────
class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    BG_GREEN = "\033[42m"
    BG_RED = "\033[41m"
    BG_BLUE = "\033[44m"
    BG_YELLOW = "\033[43m"

def colorize(text, color):
    return f"{color}{text}{C.RESET}"

def bar(width=80, char="═"):
    return char * width

def header(title, width=80):
    pad = max(0, width - len(title) - 4)
    return f"\n{C.BOLD}{C.CYAN}╔{'═' * (width-2)}╗\n║ {title}{' ' * pad} ║\n╚{'═' * (width-2)}╝{C.RESET}"


# ══════════════════════════════════════════════════════════════
# SECTION 1: MODEL STATUS
# ══════════════════════════════════════════════════════════════

def show_model_status():
    print(header("MODEL STATUS — Training Metrics"))

    # Load ensemble weights
    weights_file = MODELS_DIR / "ensemble-weights.json"
    best_file = MODELS_DIR / "best-weights.json"
    wf = best_file if best_file.exists() else weights_file

    if wf.exists():
        data = json.loads(wf.read_text())
        raw_weights = data.get("weights", {})
        # Handle nested weights structure
        if isinstance(raw_weights.get("weights"), dict):
            brier_scores = raw_weights.get("brier_scores", {})
            weights = raw_weights["weights"]
        else:
            weights = raw_weights
            brier_scores = data.get("brier_scores", {})
        print(f"\n  {C.BOLD}Ensemble Weights{C.RESET} (lower Brier = better):")
        print(f"  {'Model':<25} {'Weight':>8} {'Brier':>8} {'Status':>10}")
        print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*10}")
        for model, w in sorted(weights.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
            if not isinstance(w, (int, float)):
                continue
            brier = brier_scores.get(model, "?")
            if isinstance(brier, (int, float)):
                status = colorize("★ BEST", C.GREEN) if brier < 0.22 else colorize("OK", C.YELLOW) if brier < 0.24 else colorize("WEAK", C.RED)
                print(f"  {model:<25} {w:>8.3f} {brier:>8.4f} {status}")
            else:
                print(f"  {model:<25} {w:>8.3f} {'?':>8} {'?':>10}")
    else:
        print(f"\n  {C.RED}No trained models found. Run: python3 ops/karpathy-loop.py --full{C.RESET}")

    # Load training history
    hist_file = TRAINING_DIR / "training-history.jsonl"
    if hist_file.exists():
        events = []
        for line in hist_file.read_text().splitlines()[-10:]:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
        if events:
            print(f"\n  {C.BOLD}Recent Training Events:{C.RESET}")
            for e in events[-5:]:
                ts = e.get("ts", "")[:16]
                ev_type = e.get("type", e.get("event", "?"))
                detail = e.get("detail", e.get("accuracy", ""))
                print(f"  {C.DIM}{ts}{C.RESET} {ev_type}: {detail}")

    # Elo rankings
    elo_file = MODELS_DIR / "elo-ratings.json"
    if elo_file.exists():
        elo_data = json.loads(elo_file.read_text())
        ratings = elo_data.get("ratings", {})
        sorted_teams = sorted(ratings.items(), key=lambda x: -x[1])[:10]
        print(f"\n  {C.BOLD}Top 10 Elo Rankings:{C.RESET}")
        for i, (team, rating) in enumerate(sorted_teams, 1):
            name = NBA_TEAMS.get(team, {}).get("name", team)
            delta = rating - 1500
            color = C.GREEN if delta > 50 else C.YELLOW if delta > 0 else C.RED
            print(f"  {i:>3}. {team} {name:<28} {colorize(f'{rating:.0f}', color)} ({delta:+.0f})")


# ══════════════════════════════════════════════════════════════
# SECTION 2: TODAY'S GAMES + ODDS
# ══════════════════════════════════════════════════════════════

def load_latest_odds() -> List[dict]:
    """Load most recent odds data from snapshots or API."""
    # Try latest snapshot
    snapshots = sorted(DATA_DIR.glob("odds-*.json"), reverse=True)
    if snapshots:
        try:
            data = json.loads(snapshots[0].read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass

    # Try live fetch
    if ODDS_API_KEY:
        try:
            url = (f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
                   f"?apiKey={ODDS_API_KEY}&regions=us,eu&markets=h2h,spreads,totals"
                   f"&oddsFormat=decimal&dateFormat=iso")
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=20) as resp:
                data = json.loads(resp.read())
                if isinstance(data, list):
                    # Save snapshot
                    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
                    (DATA_DIR / f"odds-{ts}.json").write_text(json.dumps(data, indent=2))
                    return data
        except Exception:
            pass

    return []


def load_schedule() -> List[dict]:
    """Load today's schedule from nba_api."""
    sched_file = HISTORICAL_DIR / "schedule-today.json"
    if sched_file.exists():
        try:
            data = json.loads(sched_file.read_text())
            return data.get("games", [])
        except Exception:
            pass
    return []


def show_todays_games():
    print(header("TODAY'S GAMES — Odds Comparison (All Bookmakers)"))

    odds_data = load_latest_odds()
    schedule = load_schedule()

    if not odds_data and not schedule:
        print(f"\n  {C.YELLOW}No games data available. Run: python3 ops/ingest-nba-data.py --quick{C.RESET}")
        return []

    # Use odds data as primary source if available
    games = odds_data if odds_data else []

    if not games:
        # Fallback to schedule
        print(f"\n  {C.BOLD}Schedule ({len(schedule)} games):{C.RESET}")
        for g in schedule:
            tip = g.get("game_time_utc", "")[:16].replace("T", " ")
            status = g.get("game_status", "")
            print(f"  {g.get('away_team','?')} @ {g.get('home_team','?')}  |  {tip} UTC  |  {status}")
        return schedule

    # Group bookmaker odds per game
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_games = [g for g in games if g.get("commence_time", "")[:10] >= today]

    if not today_games:
        today_games = games[:12]  # Show most recent

    # Key bookmakers to display (FR + US)
    KEY_BOOKS = ["pinnacle", "draftkings", "fanduel", "betway", "winamax",
                 "betclic", "parions_sport", "unibet_eu", "williamhill"]
    BOOK_LABELS = {
        "pinnacle": "PIN", "draftkings": "DK", "fanduel": "FD",
        "betway": "BW", "winamax": "WIN", "betclic": "BC",
        "parions_sport": "PS", "unibet_eu": "UNI", "williamhill": "WH",
        "betmgm": "MGM", "bovada": "BOV", "pointsbetus": "PBU",
        "betus": "BU", "mybookieag": "MYB", "betonlineag": "BOL",
    }

    for game in today_games:
        home = game.get("home_team", "?")
        away = game.get("away_team", "?")
        tip = game.get("commence_time", "")[:16].replace("T", " ")

        # Get our model prediction
        h_abbrev, _ = get_team(home)
        a_abbrev, _ = get_team(away)
        model_pred = None
        if h_abbrev and a_abbrev:
            model_pred = predict_matchup(h_abbrev, a_abbrev)

        print(f"\n  {C.BOLD}{C.WHITE}{'─'*76}{C.RESET}")
        print(f"  {C.BOLD}{away} @ {home}{C.RESET}  |  {C.DIM}{tip} UTC{C.RESET}")

        if model_pred:
            prob = model_pred["home_win_prob"]
            if prob > 1:
                prob /= 100
            print(f"  {C.CYAN}Model: {home} {prob*100:.1f}% | {away} {(1-prob)*100:.1f}% | "
                  f"Spread: {model_pred.get('predicted_diff', 0):+.1f}{C.RESET}")

        # Extract and display bookmaker odds
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            print(f"  {C.DIM}No bookmaker data{C.RESET}")
            continue

        # H2H odds table
        print(f"\n  {'Book':>5}  {'Home':>7}  {'Away':>7}  {'Spread':>7}  {'Total':>7}  {'Value?':>8}")
        print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}")

        best_home_odds = 0
        best_away_odds = 0
        best_home_book = ""
        best_away_book = ""

        for bk in bookmakers:
            bk_key = bk.get("key", "")
            bk_label = BOOK_LABELS.get(bk_key, bk_key[:4].upper())

            h2h_home = h2h_away = spread_val = total_val = None

            for mkt in bk.get("markets", []):
                mk = mkt.get("key", "")
                for o in mkt.get("outcomes", []):
                    if mk == "h2h":
                        if o.get("name") == home:
                            h2h_home = o.get("price")
                        elif o.get("name") == away:
                            h2h_away = o.get("price")
                    elif mk == "spreads" and o.get("name") == home:
                        spread_val = o.get("point")
                    elif mk == "totals" and o.get("name") == "Over":
                        total_val = o.get("point")

            if h2h_home and h2h_away:
                # Highlight best odds
                if h2h_home > best_home_odds:
                    best_home_odds = h2h_home
                    best_home_book = bk_label
                if h2h_away > best_away_odds:
                    best_away_odds = h2h_away
                    best_away_book = bk_label

                # Calculate implied prob and check value
                impl_home = 1 / h2h_home
                value_flag = ""
                if model_pred:
                    our_prob = model_pred["home_win_prob"]
                    if our_prob > 1:
                        our_prob /= 100
                    edge_home = our_prob - impl_home
                    edge_away = (1 - our_prob) - (1 / h2h_away)
                    if edge_home > 0.03:
                        value_flag = colorize(f"+{edge_home*100:.1f}%H", C.GREEN)
                    elif edge_away > 0.03:
                        value_flag = colorize(f"+{edge_away*100:.1f}%A", C.GREEN)

                spread_str = f"{spread_val:+.1f}" if spread_val is not None else "  —"
                total_str = f"{total_val:.1f}" if total_val is not None else "  —"

                print(f"  {bk_label:>5}  {h2h_home:>7.2f}  {h2h_away:>7.2f}  {spread_str:>7}  {total_str:>7}  {value_flag:>8}")

        if best_home_odds > 0:
            print(f"  {C.GREEN}Best: {home} @ {best_home_odds:.2f} ({best_home_book}) | "
                  f"{away} @ {best_away_odds:.2f} ({best_away_book}){C.RESET}")

    return today_games


# ══════════════════════════════════════════════════════════════
# SECTION 3: VALUE BETS — Kelly Sizing
# ══════════════════════════════════════════════════════════════

def show_value_bets():
    print(header("VALUE BETS — Today's Recommended Picks"))

    # Load latest picks
    picks_files = sorted(PREDICTIONS_DIR.glob("picks-*.json"), reverse=True)
    if not picks_files:
        # Generate from current odds + model
        print(f"\n  {C.YELLOW}Generating picks from model + latest odds...{C.RESET}")
        picks = generate_fresh_picks()
        if not picks:
            print(f"  {C.RED}No value bets found. Market may be efficient today.{C.RESET}")
            return
    else:
        try:
            data = json.loads(picks_files[0].read_text())
            picks = data.get("picks", [])
            ts = data.get("timestamp", picks_files[0].stem)
            print(f"\n  {C.DIM}Last updated: {ts}{C.RESET}")
        except Exception:
            picks = []

    if not picks:
        print(f"  {C.YELLOW}No value bets identified{C.RESET}")
        return

    # Load bankroll
    bankroll = 100.0
    bankroll_files = sorted(BANKROLL_DIR.glob("daily-*.json"), reverse=True)
    if bankroll_files:
        try:
            br = json.loads(bankroll_files[0].read_text())
            bankroll = br.get("closing_bankroll", br.get("bankroll", 100.0))
        except Exception:
            pass

    print(f"\n  {C.BOLD}Bankroll: ${bankroll:.2f}{C.RESET}")
    print(f"\n  {'#':>3}  {'Bet':<35}  {'Book':>6}  {'Odds':>6}  {'Edge':>7}  {'Stake':>8}  {'EV':>7}")
    print(f"  {'─'*3}  {'─'*35}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*7}")

    total_stake = 0
    total_ev = 0
    bet_picks = [p for p in picks if p.get("is_bet")]

    for i, pick in enumerate(bet_picks[:15], 1):
        desc = pick.get("description", "?")[:35]
        book = pick.get("bookmaker", "?")[:6]
        odds = pick.get("odds", 0)
        edge = pick.get("edge", "0%")
        if isinstance(edge, str):
            edge_pct = float(edge.replace("%", "")) if edge.replace("%", "").replace(".", "").replace("-", "").isdigit() else 0
        else:
            edge_pct = edge
        stake = pick.get("stake", 0)
        ev = stake * (odds - 1) * (edge_pct / 100) if edge_pct > 0 and odds > 1 else 0

        total_stake += stake
        total_ev += ev

        edge_color = C.GREEN if edge_pct > 5 else C.YELLOW if edge_pct > 2 else C.WHITE
        print(f"  {i:>3}  {desc:<35}  {book:>6}  {odds:>6.2f}  "
              f"{colorize(f'{edge_pct:>5.1f}%', edge_color)}  "
              f"${stake:>7.2f}  ${ev:>6.2f}")

    print(f"\n  {C.BOLD}Total exposure: ${total_stake:.2f} ({total_stake/bankroll*100:.1f}% of bankroll){C.RESET}")
    if total_ev > 0:
        print(f"  {C.GREEN}Expected value: +${total_ev:.2f}{C.RESET}")


def generate_fresh_picks() -> List[dict]:
    """Generate picks from model predictions vs current odds."""
    odds_data = load_latest_odds()
    if not odds_data:
        return []

    picks = []
    bankroll = 100.0

    for game in odds_data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        h_abbrev, _ = get_team(home)
        a_abbrev, _ = get_team(away)
        if not h_abbrev or not a_abbrev:
            continue

        pred = predict_matchup(h_abbrev, a_abbrev)
        our_prob = pred["home_win_prob"]
        if our_prob > 1:
            our_prob /= 100

        for bk in game.get("bookmakers", []):
            bk_key = bk.get("key", "")
            for mkt in bk.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                for o in mkt.get("outcomes", []):
                    name = o.get("name", "")
                    price = o.get("price", 0)
                    if price <= 1:
                        continue

                    implied = 1 / price
                    if name == home:
                        edge = our_prob - implied
                        prob = our_prob
                    elif name == away:
                        edge = (1 - our_prob) - implied
                        prob = 1 - our_prob
                    else:
                        continue

                    if edge > 0.02:  # 2% minimum edge
                        # Kelly fraction (1/4 Kelly for safety)
                        kelly = max(0, (prob * price - 1) / (price - 1)) * 0.25
                        kelly = min(kelly, 0.05)  # Cap at 5%
                        stake = round(bankroll * kelly, 2)

                        picks.append({
                            "description": f"{name} ML vs {away if name == home else home}",
                            "bookmaker": bk_key,
                            "odds": price,
                            "estimated_prob": f"{prob*100:.1f}%",
                            "edge": f"{edge*100:.1f}%",
                            "kelly_fraction": f"{kelly*100:.2f}%",
                            "stake": stake,
                            "is_bet": True,
                            "reason": f"+EV: edge {edge*100:.1f}%, Kelly {kelly*100:.2f}%",
                        })

    # Sort by edge, take top 10
    picks.sort(key=lambda p: float(p["edge"].replace("%", "")), reverse=True)
    return picks[:10]


# ══════════════════════════════════════════════════════════════
# SECTION 4: P&L TRACKER
# ══════════════════════════════════════════════════════════════

def show_pnl():
    print(header("P&L — Last 7 Days Performance"))

    bankroll_files = sorted(BANKROLL_DIR.glob("daily-*.json"))
    if not bankroll_files:
        print(f"\n  {C.YELLOW}No bankroll history yet. First bets pending.{C.RESET}")
        print(f"  Starting bankroll: $100.00")
        return

    # Show last 7 entries
    recent = bankroll_files[-7:]
    print(f"\n  {'Date':<12} {'Bets':>5} {'Won':>5} {'ROI':>8} {'Bankroll':>10} {'P&L':>8}")
    print(f"  {'─'*12} {'─'*5} {'─'*5} {'─'*8} {'─'*10} {'─'*8}")

    prev_bankroll = 100.0
    for f in recent:
        try:
            data = json.loads(f.read_text())
            date = f.stem.replace("daily-", "")
            bets = data.get("total_bets", 0)
            won = data.get("won", 0)
            br = data.get("closing_bankroll", data.get("bankroll", prev_bankroll))
            roi = data.get("roi", 0)
            pnl = br - prev_bankroll

            color = C.GREEN if pnl >= 0 else C.RED
            print(f"  {date:<12} {bets:>5} {won:>5} {roi:>7.1f}% "
                  f"{colorize(f'${br:>9.2f}', color)} {colorize(f'${pnl:>+7.2f}', color)}")
            prev_bankroll = br
        except Exception:
            pass

    # Show evaluation results if available
    pred_file = PREDICTIONS_DIR / "predictions.jsonl"
    if pred_file.exists():
        preds = []
        for line in pred_file.read_text().splitlines():
            try:
                preds.append(json.loads(line))
            except Exception:
                pass
        verified = [p for p in preds if p.get("actual_winner")]
        if verified:
            correct = sum(1 for p in verified if p.get("correct"))
            total = len(verified)
            print(f"\n  {C.BOLD}Prediction Track Record: {correct}/{total} ({correct/total*100:.1f}%){C.RESET}")
            # CLV
            clvs = [p.get("clv", 0) for p in verified if p.get("clv") is not None]
            if clvs:
                avg_clv = sum(clvs) / len(clvs)
                clv_color = C.GREEN if avg_clv > 0 else C.RED
                print(f"  Average CLV: {colorize(f'{avg_clv:+.2f}%', clv_color)} "
                      f"({'beating market' if avg_clv > 0 else 'below market'})")


# ══════════════════════════════════════════════════════════════
# SECTION 5: RESEARCH & FEATURES
# ══════════════════════════════════════════════════════════════

def show_features():
    """Show feature importance and model details."""
    print(header("FEATURE IMPORTANCE — What Drives Predictions"))

    # Load feature importance from training
    feat_files = sorted(TRAINING_DIR.glob("*.jsonl"))
    if not feat_files:
        return

    # Check if we have cached feature importance
    fi_file = MODELS_DIR / "feature-importance.json"
    if fi_file.exists():
        try:
            fi = json.loads(fi_file.read_text())
            features = fi.get("features", {})
            if features:
                sorted_f = sorted(features.items(), key=lambda x: -x[1])[:15]
                print(f"\n  {'Feature':<35} {'Importance':>10} {'Bar'}")
                print(f"  {'─'*35} {'─'*10} {'─'*20}")
                max_imp = sorted_f[0][1] if sorted_f else 1
                for name, imp in sorted_f:
                    bar_len = int(20 * imp / max_imp)
                    print(f"  {name:<35} {imp:>10.4f} {C.GREEN}{'█' * bar_len}{C.RESET}")
        except Exception:
            pass

    # Show total features count
    try:
        from karpathy_loop import FEATURE_NAMES
        print(f"\n  Total features: {len(FEATURE_NAMES)}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def render_board(compact=False):
    """Render the full dashboard."""
    os.system("clear" if os.name == "posix" else "cls")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"{C.BOLD}{C.MAGENTA}")
    print(f"  ╔═══════════════════════════════════════════════════════════════════════╗")
    print(f"  ║   NOMOS NBA QUANT — Daily Trading Board                             ║")
    print(f"  ║   {now}                                              ║")
    print(f"  ╚═══════════════════════════════════════════════════════════════════════╝")
    print(f"{C.RESET}")

    show_model_status()
    show_todays_games()
    show_value_bets()
    show_pnl()

    if not compact:
        show_features()

    print(f"\n{C.DIM}  [Ctrl+C to exit | Refresh: --loop 300]{C.RESET}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NBA Quant Daily Trading Board")
    parser.add_argument("--loop", type=int, metavar="SECONDS", help="Auto-refresh interval")
    parser.add_argument("--compact", action="store_true", help="Compact mode")
    args = parser.parse_args()

    if args.loop:
        while True:
            try:
                render_board(compact=args.compact)
                print(f"  {C.DIM}Next refresh in {args.loop}s...{C.RESET}")
                time.sleep(args.loop)
            except KeyboardInterrupt:
                print(f"\n{C.YELLOW}Board stopped.{C.RESET}")
                break
    else:
        render_board(compact=args.compact)


if __name__ == "__main__":
    main()
