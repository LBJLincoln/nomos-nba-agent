"""Tests for Kelly Criterion module with realistic NBA betting scenarios."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.kelly import (
    kelly_fraction, implied_probability, edge_percentage,
    expected_value, evaluate_bet, BetOpportunity,
)


def _make_opp(odds, prob, desc="Test bet"):
    return BetOpportunity("g1", desc, "h2h", "home", odds, prob, "fanduel")


def test_kelly_fraction_positive_edge():
    """Celtics ML -150 (1.67), model says 65% → positive Kelly."""
    f = kelly_fraction(1.67, 0.65)
    assert f > 0, "Should recommend a bet when model edge exists"
    assert round(f, 4) == round((0.67 * 0.65 - 0.35) / 0.67, 4)


def test_kelly_fraction_negative_edge():
    """Lakers spread at -110 (1.91), model says only 48% → no bet."""
    f = kelly_fraction(1.91, 0.48)
    assert f < 0, "Negative Kelly when model prob < implied prob"


def test_implied_probability():
    assert round(implied_probability(2.00), 2) == 0.50
    assert round(implied_probability(1.91), 4) == round(1 / 1.91, 4)


def test_edge_percentage_value_bet():
    """Nuggets +180 (2.80), model gives 42% → edge = 0.42*2.80-1 = 0.176."""
    e = edge_percentage(2.80, 0.42)
    assert round(e, 3) == 0.176


def test_expected_value():
    ev = expected_value(2.10, 0.55, stake=1.0)
    assert ev > 0, "+EV when prob exceeds implied"


def test_evaluate_bet_recommends_bet():
    """Bucks ML -130 (1.77), model 62% → should bet with fractional Kelly."""
    opp = _make_opp(1.77, 0.62, "Bucks ML")
    result = evaluate_bet(opp, bankroll=1000)
    assert result.is_bet is True
    assert 0 < result.recommended_bet <= 50  # capped at 5% of 1000


def test_evaluate_bet_rejects_no_edge():
    """Knicks ML +100 (2.00), model 48% → no edge, pass."""
    opp = _make_opp(2.00, 0.48, "Knicks ML")
    result = evaluate_bet(opp, bankroll=1000)
    assert result.is_bet is False
    assert result.recommended_bet == 0
