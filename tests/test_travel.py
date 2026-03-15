"""Tests for travel adjustment — cross-country road trips penalize more than short hops."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.power_ratings import get_travel_adjustment, haversine_miles


def test_cross_country_trip_penalizes_heavily():
    """Portland → Miami (~2700 mi) should give a significant fatigue penalty."""
    adj = get_travel_adjustment("Portland", "Miami")
    assert adj < -0.5, f"Cross-country travel adj {adj} should be < -0.5"


def test_short_hop_minimal_penalty():
    """Brooklyn → Philadelphia (~95 mi) should barely penalize."""
    adj = get_travel_adjustment("Brooklyn", "Philadelphia")
    assert -0.15 < adj < 0.0, f"Short-hop adj {adj} should be near zero"


def test_same_city_no_penalty():
    """LA Clippers arena → LA Lakers arena — same city, ~0 penalty."""
    adj = get_travel_adjustment("Los Angeles", "Los Angeles")
    assert adj == 0.0, "Same-city travel should be zero"


def test_haversine_boston_to_san_francisco():
    """BOS → GSW distance should be roughly 2700 miles."""
    dist = haversine_miles(42.36, -71.06, 37.77, -122.39)
    assert 2600 < dist < 2800, f"BOS-GSW distance {dist:.0f} mi outside expected range"


def test_unknown_city_returns_zero():
    """Unknown city name should return 0.0, not crash."""
    adj = get_travel_adjustment("Atlantis", "Denver")
    assert adj == 0.0
