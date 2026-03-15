#!/usr/bin/env python3
"""
Bankroll Manager — Persistent bankroll state + bet recording + P&L tracking.

Tony Bloom / Starlizard approach:
- Track every bet with full Kelly metadata
- Record outcomes (win/loss/push)
- Compute running P&L, ROI, CLV
- Daily snapshots for compound growth tracking
- Wire to prediction storage for backtesting
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BANKROLL_DIR = DATA / "bankroll"
PREDICTIONS_DIR = DATA / "predictions"
BETS_FILE = BANKROLL_DIR / "bets.jsonl"
STATE_FILE = BANKROLL_DIR / "state.json"
DAILY_FILE = BANKROLL_DIR / "daily-snapshots.jsonl"

# Ensure directories
BANKROLL_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BankrollState:
    """Persistent bankroll state."""
    balance: float = 1000.0
    initial_balance: float = 1000.0
    currency: str = "USD"
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    pending: int = 0
    total_wagered: float = 0.0
    total_profit: float = 0.0
    peak_balance: float = 1000.0
    trough_balance: float = 1000.0
    max_drawdown_pct: float = 0.0
    streak_current: int = 0          # positive = wins, negative = losses
    streak_best: int = 0
    streak_worst: int = 0
    daily_bets_today: int = 0
    daily_profit_today: float = 0.0
    last_bet_ts: str = ""
    last_updated: str = ""
    created: str = ""

    def save(self):
        self.last_updated = datetime.now(timezone.utc).isoformat()
        if not self.created:
            self.created = self.last_updated
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls):
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        state = cls()
        state.save()
        return state


@dataclass
class BetRecord:
    """A single bet record for the journal."""
    bet_id: str
    timestamp: str
    game_id: str
    description: str
    market: str                  # h2h, spread, total
    selection: str               # home, away, over, under
    bookmaker: str
    decimal_odds: float
    estimated_prob: float
    implied_prob: float
    edge_pct: float
    kelly_fraction: float
    stake: float
    potential_profit: float
    status: str = "pending"      # pending, won, lost, push, cancelled
    result_odds: float = 0.0     # closing odds (for CLV)
    actual_profit: float = 0.0
    settled_at: str = ""
    notes: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def place_bet(game_id: str, description: str, market: str, selection: str,
              bookmaker: str, decimal_odds: float, estimated_prob: float,
              kelly_fraction: float = 0.0, notes: str = "") -> dict:
    """
    Record a new bet in the bankroll.

    Returns: dict with bet details and updated bankroll state.
    """
    state = BankrollState.load()

    # Calculate stake using Kelly if fraction provided, else use edge-based sizing
    sys.path.insert(0, str(ROOT / "models"))
    from kelly import evaluate_bet, BetOpportunity, implied_probability

    opp = BetOpportunity(
        game_id=game_id,
        description=description,
        market=market,
        selection=selection,
        decimal_odds=decimal_odds,
        estimated_prob=estimated_prob,
        bookmaker=bookmaker,
    )

    kelly_result = evaluate_bet(opp, state.balance, kelly_fraction or 0.25)

    if not kelly_result.is_bet:
        return {
            "action": "PASS",
            "reason": kelly_result.reason,
            "edge": kelly_result.edge,
            "balance": state.balance,
        }

    stake = kelly_result.recommended_bet
    potential_profit = round(stake * (decimal_odds - 1.0), 2)

    # Create bet record
    bet_id = f"bet-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{game_id[:8]}"
    record = BetRecord(
        bet_id=bet_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        game_id=game_id,
        description=description,
        market=market,
        selection=selection,
        bookmaker=bookmaker,
        decimal_odds=decimal_odds,
        estimated_prob=estimated_prob,
        implied_prob=round(1.0 / decimal_odds, 4),
        edge_pct=round(kelly_result.edge * 100, 2),
        kelly_fraction=kelly_result.recommended_fraction,
        stake=stake,
        potential_profit=potential_profit,
        notes=notes,
    )

    # Write to bets journal
    with open(BETS_FILE, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")

    # Update state
    state.balance -= stake
    state.total_bets += 1
    state.pending += 1
    state.total_wagered += stake
    state.daily_bets_today += 1
    state.last_bet_ts = record.timestamp
    state.trough_balance = min(state.trough_balance, state.balance)
    state.save()

    return {
        "action": "BET",
        "bet_id": bet_id,
        "description": description,
        "bookmaker": bookmaker,
        "odds": decimal_odds,
        "estimated_prob": f"{estimated_prob*100:.1f}%",
        "edge": f"{kelly_result.edge*100:.1f}%",
        "kelly": f"{kelly_result.recommended_fraction*100:.2f}%",
        "stake": stake,
        "potential_profit": potential_profit,
        "balance_after": round(state.balance, 2),
        "pending_bets": state.pending,
    }


def settle_bet(bet_id: str, outcome: str, closing_odds: float = 0.0) -> dict:
    """
    Settle a pending bet.

    Args:
        bet_id: The bet ID to settle
        outcome: 'won', 'lost', 'push'
        closing_odds: Closing line odds (for CLV calculation)

    Returns: dict with settlement details.
    """
    state = BankrollState.load()

    # Find and update the bet
    if not BETS_FILE.exists():
        return {"error": "No bets file found"}

    bets = []
    target = None
    for line in BETS_FILE.read_text().strip().split("\n"):
        if not line:
            continue
        bet = json.loads(line)
        if bet["bet_id"] == bet_id:
            target = bet
            bet["status"] = outcome
            bet["settled_at"] = datetime.now(timezone.utc).isoformat()
            bet["result_odds"] = closing_odds

            if outcome == "won":
                profit = round(bet["stake"] * (bet["decimal_odds"] - 1.0), 2)
                bet["actual_profit"] = profit
                state.balance += bet["stake"] + profit
                state.wins += 1
                state.total_profit += profit
                state.streak_current = max(state.streak_current, 0) + 1
                state.streak_best = max(state.streak_best, state.streak_current)
            elif outcome == "lost":
                bet["actual_profit"] = -bet["stake"]
                state.losses += 1
                state.total_profit -= bet["stake"]
                state.streak_current = min(state.streak_current, 0) - 1
                state.streak_worst = min(state.streak_worst, state.streak_current)
            elif outcome == "push":
                bet["actual_profit"] = 0.0
                state.balance += bet["stake"]  # refund
                state.pushes += 1

            state.pending -= 1
        bets.append(bet)

    if not target:
        return {"error": f"Bet {bet_id} not found"}

    # Rewrite bets file
    with open(BETS_FILE, "w") as f:
        for bet in bets:
            f.write(json.dumps(bet) + "\n")

    # Update peak/trough/drawdown
    state.peak_balance = max(state.peak_balance, state.balance)
    state.trough_balance = min(state.trough_balance, state.balance)
    if state.peak_balance > 0:
        dd = (state.peak_balance - state.trough_balance) / state.peak_balance * 100
        state.max_drawdown_pct = max(state.max_drawdown_pct, dd)
    state.daily_profit_today += target.get("actual_profit", 0)
    state.save()

    # CLV calculation
    clv = None
    if closing_odds > 0 and target:
        opening_implied = 1.0 / target["decimal_odds"]
        closing_implied = 1.0 / closing_odds
        clv = round((closing_implied - opening_implied) * 100, 2)

    return {
        "bet_id": bet_id,
        "outcome": outcome,
        "profit": target.get("actual_profit", 0),
        "clv_pct": clv,
        "balance": round(state.balance, 2),
        "total_profit": round(state.total_profit, 2),
        "win_rate": f"{state.wins}/{state.wins + state.losses}" if (state.wins + state.losses) > 0 else "0/0",
        "roi_pct": round((state.total_profit / max(state.total_wagered, 1)) * 100, 2),
    }


def daily_snapshot():
    """Save daily bankroll snapshot for compound growth tracking."""
    state = BankrollState.load()

    snapshot = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "balance": state.balance,
        "daily_profit": state.daily_profit_today,
        "daily_bets": state.daily_bets_today,
        "total_bets": state.total_bets,
        "win_rate": round(state.wins / max(state.wins + state.losses, 1) * 100, 1),
        "roi_pct": round((state.total_profit / max(state.total_wagered, 1)) * 100, 2),
        "peak": state.peak_balance,
        "drawdown_pct": state.max_drawdown_pct,
        "growth_pct": round((state.balance / state.initial_balance - 1) * 100, 2),
    }

    with open(DAILY_FILE, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

    # Reset daily counters
    state.daily_bets_today = 0
    state.daily_profit_today = 0.0
    state.save()

    return snapshot


def save_prediction(game_id: str, matchup: str, prediction: dict):
    """
    Save a model prediction for later backtesting.

    prediction should contain: home_win_prob, away_win_prob, spread, total,
    confidence, models_used, timestamp
    """
    pred = {
        "game_id": game_id,
        "matchup": matchup,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **prediction,
    }

    pred_file = PREDICTIONS_DIR / f"pred-{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    with open(pred_file, "a") as f:
        f.write(json.dumps(pred) + "\n")

    return pred


def get_status() -> dict:
    """Get current bankroll status summary."""
    state = BankrollState.load()

    # Count recent bets
    recent_bets = []
    if BETS_FILE.exists():
        for line in BETS_FILE.read_text().strip().split("\n"):
            if not line:
                continue
            recent_bets.append(json.loads(line))

    pending = [b for b in recent_bets if b.get("status") == "pending"]
    settled = [b for b in recent_bets if b.get("status") in ("won", "lost", "push")]

    return {
        "balance": round(state.balance, 2),
        "initial": state.initial_balance,
        "growth_pct": round((state.balance / max(state.initial_balance, 1) - 1) * 100, 2),
        "total_bets": state.total_bets,
        "record": f"{state.wins}W-{state.losses}L-{state.pushes}P",
        "win_rate": round(state.wins / max(state.wins + state.losses, 1) * 100, 1),
        "roi_pct": round((state.total_profit / max(state.total_wagered, 1)) * 100, 2),
        "total_wagered": round(state.total_wagered, 2),
        "total_profit": round(state.total_profit, 2),
        "pending": len(pending),
        "peak": state.peak_balance,
        "max_drawdown": f"{state.max_drawdown_pct:.1f}%",
        "streak": state.streak_current,
        "best_streak": state.streak_best,
        "worst_streak": state.streak_worst,
        "last_bet": state.last_bet_ts,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Bankroll Manager")
    sub = parser.add_subparsers(dest="command")

    # Status
    sub.add_parser("status", help="Show bankroll status")

    # Init
    init_p = sub.add_parser("init", help="Initialize bankroll")
    init_p.add_argument("--balance", type=float, default=1000.0)

    # Place bet
    bet_p = sub.add_parser("bet", help="Record a new bet")
    bet_p.add_argument("--game", required=True, help="Game ID")
    bet_p.add_argument("--desc", required=True, help="Bet description")
    bet_p.add_argument("--market", default="h2h", help="h2h, spread, total")
    bet_p.add_argument("--selection", default="home", help="home, away, over, under")
    bet_p.add_argument("--book", required=True, help="Bookmaker name")
    bet_p.add_argument("--odds", type=float, required=True, help="Decimal odds")
    bet_p.add_argument("--prob", type=float, required=True, help="Estimated win probability")

    # Settle
    settle_p = sub.add_parser("settle", help="Settle a bet")
    settle_p.add_argument("--id", required=True, help="Bet ID")
    settle_p.add_argument("--outcome", required=True, choices=["won", "lost", "push"])
    settle_p.add_argument("--closing-odds", type=float, default=0.0)

    # Daily snapshot
    sub.add_parser("snapshot", help="Save daily snapshot")

    # List bets
    sub.add_parser("bets", help="List recent bets")

    args = parser.parse_args()

    if args.command == "status":
        status = get_status()
        print(f"\n{'='*50}")
        print(f"  NOMOS NBA BANKROLL — Tony Bloom Mode")
        print(f"{'='*50}")
        print(f"  Balance:     ${status['balance']:,.2f} ({status['growth_pct']:+.1f}%)")
        print(f"  Record:      {status['record']}")
        print(f"  Win Rate:    {status['win_rate']:.1f}%")
        print(f"  ROI:         {status['roi_pct']:+.2f}%")
        print(f"  Wagered:     ${status['total_wagered']:,.2f}")
        print(f"  Profit:      ${status['total_profit']:+,.2f}")
        print(f"  Peak:        ${status['peak']:,.2f}")
        print(f"  Drawdown:    {status['max_drawdown']}")
        print(f"  Streak:      {status['streak']:+d} (best {status['best_streak']}, worst {status['worst_streak']})")
        print(f"  Pending:     {status['pending']}")
        print(f"{'='*50}\n")

    elif args.command == "init":
        state = BankrollState(
            balance=args.balance,
            initial_balance=args.balance,
            peak_balance=args.balance,
            trough_balance=args.balance,
        )
        state.save()
        print(f"Bankroll initialized: ${args.balance:,.2f}")

    elif args.command == "bet":
        result = place_bet(
            game_id=args.game,
            description=args.desc,
            market=args.market,
            selection=args.selection,
            bookmaker=args.book,
            decimal_odds=args.odds,
            estimated_prob=args.prob,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "settle":
        result = settle_bet(args.id, args.outcome, args.closing_odds)
        print(json.dumps(result, indent=2))

    elif args.command == "snapshot":
        snap = daily_snapshot()
        print(json.dumps(snap, indent=2))

    elif args.command == "bets":
        if BETS_FILE.exists():
            bets = [json.loads(l) for l in BETS_FILE.read_text().strip().split("\n") if l]
            for b in bets[-10:]:
                status_icon = {"won": "+", "lost": "-", "push": "=", "pending": "?"}
                icon = status_icon.get(b["status"], "?")
                profit_str = f"${b.get('actual_profit', 0):+.2f}" if b["status"] != "pending" else "pending"
                print(f"  [{icon}] {b['description']:<35s} | {b['bookmaker']:>10s} | {b['decimal_odds']:.2f} | ${b['stake']:.2f} | {profit_str}")
        else:
            print("  No bets recorded yet.")

    else:
        parser.print_help()
