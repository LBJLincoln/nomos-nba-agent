"""Tests for calculate_power_rating — contextual adjustments with realistic NBA data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.power_ratings import calculate_power_rating


def test_home_court_adds_three_points():
    """Home court should add exactly 3.0 pts to adjusted power."""
    home = calculate_power_rating("BOS", {"is_home": True, "rest_days": 2})
    away = calculate_power_rating("BOS", {"is_home": False, "rest_days": 2})
    assert home["adjustments"]["home_court"] == 3.0
    assert away["adjustments"]["home_court"] == 0.0
    assert home["adjusted_power"] - away["adjusted_power"] == 3.0


def test_denver_altitude_bonus_at_home():
    """Denver gets +1.5 altitude bonus only when playing at home."""
    den_home = calculate_power_rating("DEN", {"is_home": True, "rest_days": 2})
    den_away = calculate_power_rating("DEN", {"is_home": False, "rest_days": 2})
    assert den_home["adjustments"]["altitude"] == 1.5
    assert den_away["adjustments"]["altitude"] == 0.0


def test_injury_stacking_reduces_power():
    """Multiple injuries should stack and significantly reduce adjusted power."""
    healthy = calculate_power_rating("MIL", {"is_home": True, "rest_days": 2})
    injured = calculate_power_rating("MIL", {
        "is_home": True,
        "rest_days": 2,
        "injuries": [
            {"player": "Giannis Antetokounmpo", "tier": "superstar", "position": "PF"},
            {"player": "Damian Lillard", "tier": "all_star", "position": "PG"},
        ],
    })
    assert injured["adjusted_power"] < healthy["adjusted_power"] - 9.0


def test_unknown_team_returns_none():
    """An invalid team abbreviation should return None."""
    assert calculate_power_rating("XXX") is None
