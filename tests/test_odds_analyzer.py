"""Tests for Odds Analyzer — analyze_game_odds with realistic NBA data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.odds_analyzer import analyze_game_odds, _match_team_name


def _make_game(home="Boston Celtics", away="New York Knicks",
               home_h2h=1.55, away_h2h=2.50):
    """Build a minimal game dict with two bookmakers."""
    return {
        "id": "test123",
        "home_team": home,
        "away_team": away,
        "commence_time": "2026-03-15T00:00:00Z",
        "bookmakers": [
            {
                "key": "pinnacle",
                "markets": [{"key": "h2h", "outcomes": [
                    {"name": home, "price": home_h2h},
                    {"name": away, "price": away_h2h},
                ]}],
            },
            {
                "key": "fanduel",
                "markets": [{"key": "h2h", "outcomes": [
                    {"name": home, "price": home_h2h + 0.05},
                    {"name": away, "price": away_h2h - 0.05},
                ]}],
            },
        ],
    }


def test_analyze_finds_best_odds():
    """Best H2H odds should come from the book offering the highest price."""
    game = _make_game(home_h2h=1.55, away_h2h=2.50)
    result = analyze_game_odds(game)
    best = result["markets"]["h2h"]["best_odds"]
    assert best["Boston Celtics"]["bookmaker"] == "fanduel"   # 1.60 > 1.55
    assert best["New York Knicks"]["bookmaker"] == "pinnacle"  # 2.50 > 2.45


def test_analyze_implied_probs_sum_near_100():
    """Average implied probs across books should be close to 100% (with vig)."""
    result = analyze_game_odds(_make_game())
    probs = result["markets"]["h2h"]["implied_probs"]
    total = sum(probs.values())
    assert 0.95 < total < 1.15, f"Implied prob sum {total} should be near 1.0"


def test_match_team_name_abbreviations():
    assert _match_team_name("Boston Celtics") == "BOS"
    assert _match_team_name("Golden State Warriors") == "GSW"
    assert _match_team_name("Unknown Fake Team") is None


def test_arbitrage_detection():
    """Detect arbitrage when best odds across books imply < 100% total."""
    game = {
        "id": "arb_test",
        "home_team": "Boston Celtics",
        "away_team": "New York Knicks",
        "commence_time": "2026-03-15T00:00:00Z",
        "bookmakers": [
            {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Boston Celtics", "price": 2.20},
                {"name": "New York Knicks", "price": 1.70},
            ]}]},
            {"key": "draftkings", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Boston Celtics", "price": 1.80},
                {"name": "New York Knicks", "price": 2.15},
            ]}]},
        ],
    }
    # Best: BOS@2.20 (pinnacle) + NYK@2.15 (draftkings) = 1/2.20+1/2.15 ≈ 0.92
    result = analyze_game_odds(game)
    arb = result["markets"]["h2h"]["arbitrage"]
    assert arb["is_arb"] is True
    assert arb["profit_pct"] > 0
    assert arb["margin"] < 1.0
