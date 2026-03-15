"""Tests for Kelly multi-bet portfolio optimization with realistic NBA slates."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.kelly import evaluate_multiple_bets, BetOpportunity


def _slate():
    """Realistic 3-game NBA slate: Celtics, Nuggets, Thunder."""
    return [
        BetOpportunity("g1", "Celtics ML vs Nets", "h2h", "home", 1.45, 0.75, "draftkings"),
        BetOpportunity("g2", "Nuggets +4.5", "spread", "away", 1.91, 0.58, "fanduel"),
        BetOpportunity("g3", "Thunder ML vs Spurs", "h2h", "home", 1.67, 0.68, "betmgm"),
    ]


def test_portfolio_total_exposure_within_cap():
    """Total exposure must not exceed max_total_exposure (25% default)."""
    result = evaluate_multiple_bets(_slate(), bankroll=2000)
    assert result.total_exposure <= 0.25


def test_portfolio_bets_sorted_by_edge():
    """Bets should be returned sorted by edge descending."""
    result = evaluate_multiple_bets(_slate(), bankroll=2000)
    edges = [b.edge for b in result.bets]
    assert edges == sorted(edges, reverse=True)


def test_portfolio_no_negative_ev_bets():
    """No bet in the portfolio should have a negative recommended amount."""
    result = evaluate_multiple_bets(_slate(), bankroll=2000)
    for b in result.bets:
        assert b.recommended_bet >= 0
