"""Tests for predictor.py — ELO, Poisson PMF, and Bayesian update."""

import pytest
import math
from models.predictor import elo_expected, poisson_pmf, bayesian_update, update_with_injury


class TestEloExpected:
    def test_equal_ratings_returns_50_percent(self):
        assert elo_expected(1500, 1500) == pytest.approx(0.5)

    def test_higher_rating_favored(self):
        # 200-point ELO gap ≈ 76% expected (standard ELO formula)
        assert elo_expected(1700, 1500) == pytest.approx(0.7597, abs=0.01)

    def test_symmetry(self):
        assert elo_expected(1600, 1400) + elo_expected(1400, 1600) == pytest.approx(1.0)


class TestPoissonPmf:
    def test_mode_near_lambda(self):
        # For lambda=110, P(110) should be highest around 110
        assert poisson_pmf(110, 110.0) > poisson_pmf(100, 110.0)

    def test_zero_lambda(self):
        assert poisson_pmf(0, 0) == 1.0
        assert poisson_pmf(5, 0) == 0.0

    def test_known_value(self):
        # P(k=3, λ=3) = e^-3 * 3^3 / 3! = 0.2240
        assert poisson_pmf(3, 3.0) == pytest.approx(0.2240, abs=0.001)


class TestBayesianUpdate:
    def test_neutral_evidence(self):
        # Equal likelihoods → posterior == prior
        assert bayesian_update(0.6, 0.5, 0.5) == pytest.approx(0.6)

    def test_strong_evidence_shifts_probability(self):
        # Strong evidence for true → posterior > prior
        posterior = bayesian_update(0.5, 0.9, 0.1)
        assert posterior == pytest.approx(0.9)

    def test_injury_reduces_home_prob(self):
        # Superstar injury on home team should reduce home win prob
        prior = 0.65
        post = update_with_injury(prior, "superstar", "home")
        assert post < prior
