"""Tests for Monte Carlo simulation predictor with realistic NBA matchups."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.predictor import monte_carlo_predict


def test_mc_okc_home_vs_was_heavy_favorite():
    """OKC at home vs WAS — MC should show strong home favorite with reasonable stats."""
    result = monte_carlo_predict("OKC", "WAS", n_sims=500)
    assert result is not None
    assert result["home_win_prob"] > 0.75, "OKC should dominate WAS in simulations"
    assert result["predicted_margin"] > 10, "Expected double-digit home margin"
    assert 180 < result["predicted_total"] < 260, "Total should be in realistic NBA range"
    assert result["avg_home_score"] > result["avg_away_score"]


def test_mc_close_matchup_spread_covers():
    """BOS vs CLE — elite matchup should produce spread cover probs near 50% at 0."""
    result = monte_carlo_predict("BOS", "CLE", n_sims=500)
    assert result is not None
    assert 0.30 < result["spread_cover_probs"][0] < 0.80
    assert result["margin_stdev"] > 5, "Score variance should reflect NBA game randomness"


def test_mc_invalid_team_returns_none():
    assert monte_carlo_predict("FAKE", "BOS") is None
