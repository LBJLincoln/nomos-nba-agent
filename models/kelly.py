#!/usr/bin/env python3
"""
Kelly Criterion — Optimal bankroll allocation for sports betting.

Core formula: f* = (bp - q) / b
  where:
    b = decimal odds - 1 (net profit per unit wagered)
    p = estimated probability of winning
    q = 1 - p (probability of losing)
    f* = fraction of bankroll to wager

Starlizard approach:
- Fractional Kelly (1/4 to 1/2 Kelly) to reduce variance
- Maximum bet cap (5% of bankroll)
- Multiple simultaneous bet optimization (independent events)
- Daily compound growth target: 20-30% over long run
- Never bet if edge < minimum threshold (2%)
"""

import math, json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime, timezone


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

FRACTIONAL_KELLY = 0.25       # Use 1/4 Kelly for safety (Starlizard uses 1/4 to 1/3)
MAX_BET_FRACTION = 0.05       # Never bet more than 5% of bankroll
MIN_EDGE_THRESHOLD = 0.02     # Minimum 2% edge to bet
MIN_ODDS = 1.20               # Don't bet below 1.20 decimal (too little juice)
MAX_ODDS = 10.0               # Don't bet above 10.0 decimal (too risky)
DEFAULT_BANKROLL = 1000.0     # Starting bankroll in units


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BetOpportunity:
    """A potential bet to evaluate."""
    game_id: str
    description: str           # e.g., "Boston Celtics ML"
    market: str                # "h2h", "spread", "total"
    selection: str             # "home", "away", "over", "under"
    decimal_odds: float        # From bookmaker
    estimated_prob: float      # Our model's probability
    bookmaker: str             # Which book offers this
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class KellyResult:
    """Result of Kelly calculation for a single bet."""
    opportunity: dict
    full_kelly: float          # Full Kelly fraction
    fractional_kelly: float    # After applying fraction
    recommended_fraction: float  # After caps
    recommended_bet: float     # In currency units
    edge: float                # Our edge percentage
    expected_value: float      # Expected value per unit
    implied_prob: float        # Bookmaker's implied probability
    is_bet: bool               # Should we bet?
    reason: str                # Why or why not


@dataclass
class MultiKellyResult:
    """Result for multiple simultaneous bets."""
    bets: List[KellyResult]
    total_exposure: float      # Total % of bankroll at risk
    expected_portfolio_ev: float  # Expected value of all bets combined
    bankroll: float
    timestamp: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# CORE KELLY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def implied_probability(decimal_odds):
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def decimal_to_american(decimal_odds):
    """Convert decimal odds to American format."""
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))


def american_to_decimal(american_odds):
    """Convert American odds to decimal format."""
    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    else:
        return (100.0 / abs(american_odds)) + 1.0


def kelly_fraction(decimal_odds, estimated_prob):
    """
    Calculate the Kelly fraction.

    f* = (bp - q) / b
    where b = decimal_odds - 1, p = estimated_prob, q = 1 - p

    Returns:
        f*: fraction of bankroll to bet (can be negative = don't bet)
    """
    b = decimal_odds - 1.0  # Net profit per unit wagered
    p = estimated_prob
    q = 1.0 - p

    if b <= 0:
        return 0.0

    f_star = (b * p - q) / b
    return f_star


def edge_percentage(decimal_odds, estimated_prob):
    """
    Calculate our edge over the bookmaker.
    Edge = (estimated_prob * decimal_odds) - 1
    Positive edge = +EV bet
    """
    return (estimated_prob * decimal_odds) - 1.0


def expected_value(decimal_odds, estimated_prob, stake=1.0):
    """
    Calculate expected value of a bet.
    EV = (prob_win * profit) - (prob_lose * stake)
    """
    prob_win = estimated_prob
    prob_lose = 1.0 - estimated_prob
    profit = stake * (decimal_odds - 1.0)
    ev = (prob_win * profit) - (prob_lose * stake)
    return ev


def evaluate_bet(opportunity, bankroll=DEFAULT_BANKROLL,
                 kelly_fraction_mult=FRACTIONAL_KELLY,
                 max_fraction=MAX_BET_FRACTION,
                 min_edge=MIN_EDGE_THRESHOLD):
    """
    Full Kelly evaluation for a single bet opportunity.

    Returns KellyResult with recommendation.
    """
    odds = opportunity.decimal_odds
    prob = opportunity.estimated_prob
    impl_prob = implied_probability(odds)

    # Calculate Kelly
    full_k = kelly_fraction(odds, prob)
    frac_k = full_k * kelly_fraction_mult

    # Calculate edge
    edge = edge_percentage(odds, prob)
    ev = expected_value(odds, prob)

    # Decision logic
    is_bet = True
    reason = ""

    if edge < min_edge:
        is_bet = False
        reason = f"Edge trop faible: {edge*100:.1f}% < {min_edge*100:.1f}% minimum"
    elif full_k <= 0:
        is_bet = False
        reason = f"Kelly negatif ({full_k:.4f}) — pas de valeur"
    elif odds < MIN_ODDS:
        is_bet = False
        reason = f"Cotes trop basses: {odds:.2f} < {MIN_ODDS:.2f}"
    elif odds > MAX_ODDS:
        is_bet = False
        reason = f"Cotes trop elevees: {odds:.2f} > {MAX_ODDS:.2f} (risque)"
    elif prob < 0.08 or prob > 0.97:
        is_bet = False
        reason = f"Probabilite estimee aberrante: {prob*100:.1f}%"
    elif abs(prob - impl_prob) > 0.35:
        is_bet = False
        reason = f"Desaccord modele/marche trop grand: modele {prob*100:.1f}% vs marche {impl_prob*100:.1f}% (delta {abs(prob-impl_prob)*100:.0f}pp)"
    else:
        reason = f"+EV confirme: edge {edge*100:.1f}%, Kelly {frac_k*100:.2f}%"

    # Apply caps
    recommended = min(frac_k, max_fraction) if is_bet else 0.0
    recommended = max(recommended, 0.0)
    bet_amount = recommended * bankroll

    return KellyResult(
        opportunity=asdict(opportunity) if hasattr(opportunity, '__dataclass_fields__') else vars(opportunity),
        full_kelly=round(full_k, 6),
        fractional_kelly=round(frac_k, 6),
        recommended_fraction=round(recommended, 6),
        recommended_bet=round(bet_amount, 2),
        edge=round(edge, 4),
        expected_value=round(ev, 4),
        implied_prob=round(impl_prob, 4),
        is_bet=is_bet,
        reason=reason,
    )


def evaluate_multiple_bets(opportunities, bankroll=DEFAULT_BANKROLL,
                           kelly_fraction_mult=FRACTIONAL_KELLY,
                           max_total_exposure=0.25):
    """
    Evaluate multiple simultaneous bets with portfolio constraints.

    Starlizard approach: optimize across all bets, ensuring total exposure
    doesn't exceed max_total_exposure of bankroll.

    Args:
        opportunities: list of BetOpportunity
        bankroll: current bankroll
        kelly_fraction_mult: Kelly fraction to use
        max_total_exposure: max fraction of bankroll at risk simultaneously

    Returns: MultiKellyResult
    """
    results = []
    for opp in opportunities:
        result = evaluate_bet(opp, bankroll, kelly_fraction_mult)
        results.append(result)

    # Sort by edge (best bets first)
    results.sort(key=lambda r: r.edge, reverse=True)

    # Apply portfolio constraint: scale down if total exposure exceeds limit
    total_exposure = sum(r.recommended_fraction for r in results if r.is_bet)

    if total_exposure > max_total_exposure:
        scale_factor = max_total_exposure / total_exposure
        for r in results:
            if r.is_bet:
                r.recommended_fraction = round(r.recommended_fraction * scale_factor, 6)
                r.recommended_bet = round(r.recommended_fraction * bankroll, 2)

    total_exposure_final = sum(r.recommended_fraction for r in results if r.is_bet)
    total_ev = sum(r.expected_value * r.recommended_fraction * bankroll for r in results if r.is_bet)

    return MultiKellyResult(
        bets=results,
        total_exposure=round(total_exposure_final, 4),
        expected_portfolio_ev=round(total_ev, 2),
        bankroll=bankroll,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# COMPOUND GROWTH PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compound_projection(bankroll, daily_edge_pct, days=365, bets_per_day=3):
    """
    Project compound bankroll growth.

    Args:
        bankroll: starting bankroll
        daily_edge_pct: average daily edge as decimal (e.g., 0.03 = 3%)
        days: projection period
        bets_per_day: average bets per day
    """
    projections = []
    current = bankroll

    for day in range(1, days + 1):
        # Daily growth from compounding: each bet compounds
        daily_growth = 1.0 + (daily_edge_pct / bets_per_day)
        for _ in range(bets_per_day):
            current *= daily_growth

        if day % 30 == 0 or day == 1 or day == days:
            projections.append({
                "day": day,
                "bankroll": round(current, 2),
                "growth_pct": round((current / bankroll - 1) * 100, 1),
                "multiple": round(current / bankroll, 2),
            })

    return {
        "starting_bankroll": bankroll,
        "daily_edge_pct": daily_edge_pct,
        "bets_per_day": bets_per_day,
        "projections": projections,
        "final_bankroll": round(current, 2),
        "total_return_pct": round((current / bankroll - 1) * 100, 1),
        "annualized_return_pct": round(((current / bankroll) ** (365 / days) - 1) * 100, 1),
    }


def format_kelly_report(kelly_result, lang="fr"):
    """Format a Kelly evaluation into a readable report."""
    r = kelly_result
    if lang == "fr":
        lines = [
            f"{'─'*55}",
            f"Pari: {r.opportunity.get('description', 'N/A')}",
            f"Book: {r.opportunity.get('bookmaker', 'N/A')} | Cotes: {r.opportunity.get('decimal_odds', 0):.2f}",
            f"",
            f"  Proba estimee:  {r.opportunity.get('estimated_prob', 0)*100:.1f}%",
            f"  Proba implicite: {r.implied_prob*100:.1f}%",
            f"  Edge:           {r.edge*100:.1f}%",
            f"  EV/unite:       {r.expected_value:+.4f}",
            f"",
            f"  Kelly complet:  {r.full_kelly*100:.2f}%",
            f"  Kelly 1/4:      {r.fractional_kelly*100:.2f}%",
            f"  Mise recommandee: {r.recommended_fraction*100:.2f}% = {r.recommended_bet:.2f}$",
            f"",
            f"  Decision: {'PARIER' if r.is_bet else 'PASSER'}",
            f"  Raison:   {r.reason}",
            f"{'─'*55}",
        ]
        return "\n".join(lines)
    else:
        lines = [
            f"{'─'*55}",
            f"Bet: {r.opportunity.get('description', 'N/A')}",
            f"Book: {r.opportunity.get('bookmaker', 'N/A')} | Odds: {r.opportunity.get('decimal_odds', 0):.2f}",
            f"  Edge: {r.edge*100:.1f}% | EV: {r.expected_value:+.4f}",
            f"  Kelly: {r.fractional_kelly*100:.2f}% | Bet: {r.recommended_bet:.2f}$",
            f"  {'BET' if r.is_bet else 'PASS'}: {r.reason}",
            f"{'─'*55}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kelly Criterion Calculator")
    parser.add_argument("--odds", type=float, help="Decimal odds (e.g., 2.10)")
    parser.add_argument("--prob", type=float, help="Estimated probability (e.g., 0.55)")
    parser.add_argument("--bankroll", type=float, default=1000, help="Current bankroll")
    parser.add_argument("--fraction", type=float, default=0.25, help="Kelly fraction (default 1/4)")
    parser.add_argument("--projection", action="store_true", help="Show 365-day compound projection")
    parser.add_argument("--edge", type=float, default=0.03, help="Daily edge for projection (default 3%)")
    args = parser.parse_args()

    if args.odds and args.prob:
        opp = BetOpportunity(
            game_id="manual",
            description=f"Manual bet @ {args.odds:.2f}",
            market="h2h",
            selection="home",
            decimal_odds=args.odds,
            estimated_prob=args.prob,
            bookmaker="manual",
        )
        result = evaluate_bet(opp, args.bankroll, args.fraction)
        print(format_kelly_report(result))

    elif args.projection:
        proj = compound_projection(args.bankroll, args.edge)
        print(f"\nProjection compound — Bankroll: {args.bankroll}$ | Edge: {args.edge*100:.1f}%/jour")
        print(f"{'='*55}")
        for p in proj["projections"]:
            print(f"  Jour {p['day']:>4d}: {p['bankroll']:>12,.2f}$ ({p['growth_pct']:+.1f}% | x{p['multiple']:.1f})")
        print(f"\nRendement annualise: {proj['annualized_return_pct']:.1f}%")
        print(f"Multiple final: x{proj['final_bankroll']/args.bankroll:.1f}")

    else:
        # Demo
        print("\nKelly Criterion — Exemples:")
        print("="*55)
        examples = [
            (2.10, 0.55, "Favori leger"),
            (1.91, 0.56, "Spread -110 avec 56% edge"),
            (3.50, 0.35, "Underdog value"),
            (1.50, 0.60, "Heavy favorite"),
            (2.00, 0.48, "Mauvais pari (neg EV)"),
        ]
        for odds, prob, desc in examples:
            opp = BetOpportunity("demo", desc, "h2h", "home", odds, prob, "demo")
            result = evaluate_bet(opp, 1000, 0.25)
            edge_str = f"{result.edge*100:+.1f}%"
            bet_str = f"{result.recommended_bet:.2f}$" if result.is_bet else "PASS"
            print(f"  {desc:<25s} | Cotes {odds:.2f} | Prob {prob*100:.0f}% | Edge {edge_str:>6s} | {bet_str}")
