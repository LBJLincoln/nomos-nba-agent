"""Smoke test for NBAFeatureEngine — verifies build() doesn't crash."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.engine import NBAFeatureEngine


def test_engine_instantiates():
    """Engine should create feature names on init."""
    engine = NBAFeatureEngine(include_market=False)
    assert len(engine.feature_names) > 100, "Should have 100+ feature names"
    assert all(isinstance(n, str) for n in engine.feature_names)


def test_engine_builds_from_synthetic_games():
    """Engine should produce features from minimal synthetic games."""
    engine = NBAFeatureEngine(include_market=False)
    games = []
    teams = ["BOS", "NYK", "MIA", "PHI"]
    for i in range(40):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        games.append({
            "game_date": f"2026-01-{(i % 28) + 1:02d}",
            "home_team": home,
            "away_team": away,
            "home_score": 105 + (i % 15),
            "away_score": 100 + ((i * 3) % 17),
        })
    X, y, names = engine.build(games)
    assert X.shape[0] > 0, "Should produce at least 1 row"
    assert X.shape[1] == len(names), "Feature count must match names"
    assert len(names) > 50, "Should have substantial features"
    assert set(y).issubset({0, 1}), "Labels should be 0 or 1"
