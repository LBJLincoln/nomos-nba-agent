"""Tests for Power Ratings model with realistic NBA matchup scenarios."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.power_ratings import predict_matchup, get_rest_adjustment, get_injury_adjustment


def test_predict_matchup_home_favorite():
    """OKC (best team) hosting WAS (worst) → OKC heavy favorite."""
    p = predict_matchup("OKC", "WAS")
    assert p["predicted_diff"] > 15, "OKC should be 15+ pt favorite at home vs WAS"
    assert p["home_win_prob"] > 0.90
    assert p["confidence"] == "VERY HIGH"
    assert 180 < p["predicted_total"] < 240


def test_predict_matchup_close_game():
    """BOS vs CLE — two elite East teams → tight matchup."""
    p = predict_matchup("BOS", "CLE")
    assert abs(p["predicted_diff"]) < 10, "Elite matchup should be within 10 pts"
    assert 0.35 < p["home_win_prob"] < 0.85


def test_back_to_back_penalty():
    """Away team on B2B should get penalized vs rested home team."""
    normal = predict_matchup("DEN", "DAL")
    b2b = predict_matchup("DEN", "DAL", away_context={"rest_days": 0})
    assert b2b["predicted_diff"] > normal["predicted_diff"], "B2B away team widens home edge"


def test_rest_adjustment_values():
    assert get_rest_adjustment(0) == -4.0
    assert get_rest_adjustment(2) == 0.0
    assert get_rest_adjustment(99) == 0.8  # capped at 4+


def test_injury_adjustment_realistic_nba():
    """Losing a superstar C (Jokic) hurts more than losing a rotation SG."""
    # Superstar center out — largest possible impact
    star_out = get_injury_adjustment([
        {"player": "Nikola Jokic", "tier": "superstar", "position": "C"},
    ])
    assert star_out < -6.0, "Superstar C injury should exceed -6.0 pts"

    # Rotation shooting guard out — minimal impact
    bench_out = get_injury_adjustment([
        {"player": "Bench SG", "tier": "rotation", "position": "SG"},
    ])
    assert -1.0 < bench_out < 0.0, "Rotation SG loss should be minor"

    # Multiple injuries stack: All-Star PG + starter SF
    multi = get_injury_adjustment([
        {"player": "Tyrese Haliburton", "tier": "all_star", "position": "PG"},
        {"player": "Starter SF", "tier": "starter", "position": "SF"},
    ])
    assert multi < bench_out, "Two injuries should hurt more than one rotation player"
    assert multi < -5.0
