#!/usr/bin/env python3
"""
Game Predictor — Multi-model prediction engine for NBA games.

Models:
1. Poisson model for total points prediction
2. ELO-based win probability
3. Monte Carlo simulation (1000+ iterations)
4. Bayesian updating with new information
5. Ensemble: weighted combination of all models

Tony Bloom / Starlizard approach:
- Multiple independent models voting
- Confidence intervals, not point estimates
- Bayesian updating as information arrives
- Track model accuracy over time
"""

import math, json, random, hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

from models.power_ratings import predict_matchup, NBA_TEAMS, get_team, SIGMOID_SCALE


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ELO parameters (calibrated to NBA)
ELO_K_FACTOR = 20.0          # How much a single game shifts ELO
ELO_HOME_ADVANTAGE = 100.0   # ELO points for home court (~3.0 real points)
ELO_INITIAL = 1500.0         # Starting ELO for all teams

# Poisson parameters
NBA_AVG_TEAM_SCORE = 113.5   # League average points per game 2025-26
POISSON_HOME_BOOST = 2.5     # Additional points for home team in Poisson

# Monte Carlo
MC_SIMULATIONS = 1000        # Number of simulations per prediction
MC_SCORE_STDEV = 12.0        # Standard deviation of team scores (empirical NBA)

# Ensemble weights (sum to 1.0)
ENSEMBLE_WEIGHTS = {
    "power_ratings": 0.35,    # Our power ratings model
    "elo": 0.20,              # ELO model
    "poisson": 0.15,          # Poisson model
    "monte_carlo": 0.30,      # Monte Carlo simulation
}

# ══════════════════════════════════════════════════════════════════════════════
# ELO RATING MODEL
# ══════════════════════════════════════════════════════════════════════════════

# Current ELO ratings (initialized from power ratings for bootstrapping)
_elo_ratings = {}

def _init_elo():
    """Initialize ELO ratings from power ratings."""
    global _elo_ratings
    for abbrev, team in NBA_TEAMS.items():
        # Convert power rating to ELO scale
        # Power rating of +10 ~ ELO of 1700, -10 ~ ELO of 1300
        _elo_ratings[abbrev] = ELO_INITIAL + (team["base_power"] * 20)

_init_elo()


def elo_expected(rating_a, rating_b):
    """Expected score for player A (0 to 1)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_win_probability(home_team, away_team):
    """
    Calculate win probability using ELO ratings.
    Home team gets ELO_HOME_ADVANTAGE added.
    """
    home_abbrev, _ = get_team(home_team)
    away_abbrev, _ = get_team(away_team)

    if not home_abbrev or not away_abbrev:
        return 0.5, 0.5

    home_elo = _elo_ratings.get(home_abbrev, ELO_INITIAL) + ELO_HOME_ADVANTAGE
    away_elo = _elo_ratings.get(away_abbrev, ELO_INITIAL)

    home_prob = elo_expected(home_elo, away_elo)
    return home_prob, 1.0 - home_prob


def elo_update(home_team, away_team, home_won, margin=0):
    """
    Update ELO ratings after a game result.
    Uses margin of victory as multiplier (MOV factor).
    """
    home_abbrev, _ = get_team(home_team)
    away_abbrev, _ = get_team(away_team)

    if not home_abbrev or not away_abbrev:
        return

    home_elo = _elo_ratings[home_abbrev]
    away_elo = _elo_ratings[away_abbrev]

    expected_home = elo_expected(home_elo + ELO_HOME_ADVANTAGE, away_elo)
    actual_home = 1.0 if home_won else 0.0

    # MOV factor: amplify K for blowouts, reduce for close games
    # Based on FiveThirtyEight's NBA ELO methodology
    mov_mult = math.log(max(abs(margin), 1) + 1) * (2.2 / (1.0 + 0.001 * abs(home_elo - away_elo)))
    k = ELO_K_FACTOR * max(mov_mult, 0.5)

    _elo_ratings[home_abbrev] += k * (actual_home - expected_home)
    _elo_ratings[away_abbrev] += k * (expected_home - actual_home)


def get_elo_rankings():
    """Return all teams sorted by ELO."""
    rankings = []
    for abbrev, elo in sorted(_elo_ratings.items(), key=lambda x: -x[1]):
        team = NBA_TEAMS.get(abbrev, {})
        rankings.append({
            "team": abbrev,
            "name": team.get("name", abbrev),
            "elo": round(elo, 1),
        })
    return rankings


# ══════════════════════════════════════════════════════════════════════════════
# POISSON MODEL
# ══════════════════════════════════════════════════════════════════════════════

def poisson_pmf(k, lam):
    """Probability mass function for Poisson distribution (log-space to avoid overflow)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    # Use log-space: log(P) = k*log(lam) - lam - log(k!)
    log_p = k * math.log(lam) - lam - math.lgamma(k + 1)
    return math.exp(log_p)


def poisson_predict(home_team, away_team):
    """
    Predict game outcome using Poisson model.

    Uses team offensive/defensive ratings to estimate expected scores,
    then uses Poisson distribution for score probabilities.

    Returns:
        home_expected: expected home score
        away_expected: expected away score
        home_win_prob: probability of home win
        over_under_probs: dict of total point probabilities
    """
    home_abbrev, home_data = get_team(home_team)
    away_abbrev, away_data = get_team(away_team)

    if not home_data or not away_data:
        return None

    # Calculate expected scores using offensive/defensive rating interaction
    # home_expected = (home_ortg / league_avg) * (away_drtg / league_avg) * league_avg + home_boost
    league_avg = NBA_AVG_TEAM_SCORE
    home_expected = (home_data["ortg"] / league_avg) * (away_data["drtg"] / league_avg) * league_avg + POISSON_HOME_BOOST
    away_expected = (away_data["ortg"] / league_avg) * (home_data["drtg"] / league_avg) * league_avg

    # Calculate win/draw probabilities using Poisson PMFs
    # Sum over all possible score combinations
    home_win_prob = 0.0
    away_win_prob = 0.0
    total_probs = defaultdict(float)

    # NBA scores typically range 80-140; we check 70-160 for completeness
    max_score = 160
    for h_score in range(70, max_score):
        p_home = poisson_pmf(h_score, home_expected)
        if p_home < 1e-10:
            continue
        for a_score in range(70, max_score):
            p_away = poisson_pmf(a_score, away_expected)
            if p_away < 1e-10:
                continue
            joint = p_home * p_away
            total = h_score + a_score
            total_probs[total] += joint

            if h_score > a_score:
                home_win_prob += joint
            elif a_score > h_score:
                away_win_prob += joint

    # Normalize (there's a tiny remainder from ties, which don't happen in NBA due to OT)
    total_prob = home_win_prob + away_win_prob
    if total_prob > 0:
        home_win_prob /= total_prob
        away_win_prob /= total_prob

    # Over/under analysis for common totals
    predicted_total = home_expected + away_expected
    over_probs = {}
    for line in [int(predicted_total) - 5, int(predicted_total), int(predicted_total) + 5]:
        over_prob = sum(p for t, p in total_probs.items() if t > line)
        under_prob = sum(p for t, p in total_probs.items() if t <= line)
        norm = over_prob + under_prob
        if norm > 0:
            over_probs[line] = round(over_prob / norm, 4)

    return {
        "model": "poisson",
        "home_expected": round(home_expected, 1),
        "away_expected": round(away_expected, 1),
        "predicted_total": round(predicted_total, 1),
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(away_win_prob, 4),
        "over_under_probs": over_probs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def monte_carlo_predict(home_team, away_team, n_sims=MC_SIMULATIONS,
                        home_context=None, away_context=None):
    """
    Monte Carlo simulation for game outcome.

    Simulates n_sims games, each with:
    - Random home score ~ Normal(home_expected, stdev)
    - Random away score ~ Normal(away_expected, stdev)
    - Clamped to minimum 70 points

    Returns distribution of outcomes.
    """
    # Get power rating prediction as base
    prediction = predict_matchup(home_team, away_team, home_context, away_context)
    if not prediction:
        return None

    home_mean = prediction["home_expected_pts"]
    away_mean = prediction["away_expected_pts"]

    # Run simulations
    home_wins = 0
    away_wins = 0
    home_scores = []
    away_scores = []
    margins = []
    totals = []
    spread_covers = defaultdict(int)  # spread_line -> times home covered

    for _ in range(n_sims):
        # Sample scores from normal distribution
        h_score = max(70, np.random.normal(home_mean, MC_SCORE_STDEV))
        a_score = max(70, np.random.normal(away_mean, MC_SCORE_STDEV))

        # Round to integers (real scores are integers)
        h_score = round(h_score)
        a_score = round(a_score)

        # Handle ties (go to OT — slight home advantage)
        while h_score == a_score:
            # OT averages ~10 points per team
            h_ot = max(0, round(np.random.normal(5.2, 2.5)))
            a_ot = max(0, round(np.random.normal(4.8, 2.5)))
            h_score += h_ot
            a_score += a_ot

        home_scores.append(h_score)
        away_scores.append(a_score)
        margin = h_score - a_score
        margins.append(margin)
        totals.append(h_score + a_score)

        if h_score > a_score:
            home_wins += 1
        else:
            away_wins += 1

        # Track spread coverage for common lines
        for spread_line in [-10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10]:
            if margin > -spread_line:  # Home covers when margin > -spread (spread is negative for favorites)
                spread_covers[spread_line] += 1

    home_win_prob = home_wins / n_sims
    margins_arr = np.array(margins)
    totals_arr = np.array(totals)

    return {
        "model": "monte_carlo",
        "n_simulations": n_sims,
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(1 - home_win_prob, 4),
        "predicted_margin": round(float(np.mean(margins_arr)), 1),
        "margin_stdev": round(float(np.std(margins_arr)), 1),
        "predicted_total": round(float(np.mean(totals_arr)), 1),
        "total_stdev": round(float(np.std(totals_arr)), 1),
        "margin_percentiles": {
            "5th": round(float(np.percentile(margins_arr, 5)), 1),
            "25th": round(float(np.percentile(margins_arr, 25)), 1),
            "median": round(float(np.median(margins_arr)), 1),
            "75th": round(float(np.percentile(margins_arr, 75)), 1),
            "95th": round(float(np.percentile(margins_arr, 95)), 1),
        },
        "total_percentiles": {
            "5th": round(float(np.percentile(totals_arr, 5)), 1),
            "25th": round(float(np.percentile(totals_arr, 25)), 1),
            "median": round(float(np.median(totals_arr)), 1),
            "75th": round(float(np.percentile(totals_arr, 75)), 1),
            "95th": round(float(np.percentile(totals_arr, 95)), 1),
        },
        "spread_cover_probs": {
            k: round(v / n_sims, 4)
            for k, v in sorted(spread_covers.items())
        },
        "avg_home_score": round(float(np.mean(home_scores)), 1),
        "avg_away_score": round(float(np.mean(away_scores)), 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BAYESIAN UPDATER
# ══════════════════════════════════════════════════════════════════════════════

def bayesian_update(prior_prob, likelihood_given_true, likelihood_given_false):
    """
    Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)

    Args:
        prior_prob: P(home_wins) — our prior belief
        likelihood_given_true: P(evidence | home_wins)
        likelihood_given_false: P(evidence | home_loses)

    Returns:
        posterior: updated probability
    """
    p_b = (likelihood_given_true * prior_prob) + (likelihood_given_false * (1 - prior_prob))
    if p_b == 0:
        return prior_prob
    posterior = (likelihood_given_true * prior_prob) / p_b
    return posterior


def update_with_injury(prior_prob, injured_player_tier="starter", team="home"):
    """
    Update win probability when an injury is reported.

    Likelihoods are calibrated from historical injury impact data.
    """
    # How likely is a team to win given their star is injured?
    impact = {
        "superstar": {"true": 0.35, "false": 0.65},  # Superstar out: 35% win rate
        "all_star":  {"true": 0.42, "false": 0.58},
        "starter":   {"true": 0.46, "false": 0.54},
        "rotation":  {"true": 0.49, "false": 0.51},
    }

    tier_data = impact.get(injured_player_tier, impact["rotation"])

    if team == "home":
        # Home team injured — reduces their win probability
        return bayesian_update(prior_prob, tier_data["true"], tier_data["false"])
    else:
        # Away team injured — increases home win probability
        return bayesian_update(prior_prob, tier_data["false"], tier_data["true"])


def update_with_line_movement(prior_prob, opening_spread, current_spread, is_home_favorite=True):
    """
    Update probability based on line movement (sharp money indicator).

    If line moves toward a team, sharp money is on them.
    Each point of line movement ~ 3% probability shift.
    """
    line_move = current_spread - opening_spread  # Negative = moved toward home

    if is_home_favorite:
        # Line moving more negative = more home favorite = sharp money on home
        prob_shift = -line_move * 0.03
    else:
        prob_shift = line_move * 0.03

    updated = prior_prob + prob_shift
    return max(0.05, min(0.95, updated))


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_predict(home_team, away_team, home_context=None, away_context=None):
    """
    Combine all models into an ensemble prediction.

    Each model votes with a weight, and the final probability is the
    weighted average. Also generates confidence intervals.
    """
    results = {}

    # 1. Power Ratings model
    pr_pred = predict_matchup(home_team, away_team, home_context, away_context)
    if pr_pred:
        results["power_ratings"] = {
            "home_win_prob": pr_pred["home_win_prob"],
            "predicted_spread": pr_pred["spread"],
            "predicted_total": pr_pred["predicted_total"],
        }

    # 2. ELO model
    elo_home, elo_away = elo_win_probability(home_team, away_team)
    results["elo"] = {
        "home_win_prob": round(elo_home, 4),
    }

    # 3. Poisson model
    poisson = poisson_predict(home_team, away_team)
    if poisson:
        results["poisson"] = {
            "home_win_prob": poisson["home_win_prob"],
            "predicted_total": poisson["predicted_total"],
        }

    # 4. Monte Carlo simulation
    mc = monte_carlo_predict(home_team, away_team, MC_SIMULATIONS, home_context, away_context)
    if mc:
        results["monte_carlo"] = {
            "home_win_prob": mc["home_win_prob"],
            "predicted_spread": -mc["predicted_margin"],
            "predicted_total": mc["predicted_total"],
            "margin_stdev": mc["margin_stdev"],
            "percentiles": mc["margin_percentiles"],
            "spread_covers": mc["spread_cover_probs"],
        }

    # Ensemble weighted average
    ensemble_prob = 0.0
    total_weight = 0.0
    model_probs = []

    for model_name, weight in ENSEMBLE_WEIGHTS.items():
        if model_name in results and "home_win_prob" in results[model_name]:
            prob = results[model_name]["home_win_prob"]
            ensemble_prob += prob * weight
            total_weight += weight
            model_probs.append(prob)

    if total_weight > 0:
        ensemble_prob /= total_weight
    else:
        ensemble_prob = 0.5

    # Confidence: based on model agreement
    if len(model_probs) >= 2:
        prob_stdev = float(np.std(model_probs))
        # Low stdev = high agreement = high confidence
        if prob_stdev < 0.03:
            confidence = "VERY HIGH"
        elif prob_stdev < 0.06:
            confidence = "HIGH"
        elif prob_stdev < 0.10:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        model_agreement = round(1.0 - prob_stdev, 4)
    else:
        confidence = "LOW"
        model_agreement = 0.5
        prob_stdev = 0.15

    # Get spread/total from Monte Carlo (most reliable for these)
    predicted_spread = mc["predicted_margin"] if mc else (pr_pred["spread"] if pr_pred else 0)
    predicted_total = mc["predicted_total"] if mc else (pr_pred["predicted_total"] if pr_pred else 220)

    home_abbrev, home_data = get_team(home_team)
    away_abbrev, away_data = get_team(away_team)

    return {
        "home_team": home_abbrev,
        "home_name": home_data["name"] if home_data else home_team,
        "away_team": away_abbrev,
        "away_name": away_data["name"] if away_data else away_team,
        "ensemble_home_win_prob": round(ensemble_prob, 4),
        "ensemble_away_win_prob": round(1 - ensemble_prob, 4),
        "predicted_spread": round(predicted_spread, 1),
        "predicted_total": round(predicted_total, 1),
        "confidence": confidence,
        "model_agreement": model_agreement,
        "individual_models": results,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "confidence_interval_90": {
            "home_win_prob_low": round(max(0.01, ensemble_prob - 1.645 * prob_stdev), 4),
            "home_win_prob_high": round(min(0.99, ensemble_prob + 1.645 * prob_stdev), 4),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def format_prediction_report(prediction, lang="fr"):
    """Format ensemble prediction into a detailed report."""
    p = prediction
    lines = [
        f"\n{'='*70}",
        f"PREDICTION ENSEMBLE — {p['away_name']} @ {p['home_name']}",
        f"{'='*70}",
        f"",
        f"  Probabilite de victoire:",
        f"    {p['home_name']}: {p['ensemble_home_win_prob']*100:.1f}%",
        f"    {p['away_name']}: {p['ensemble_away_win_prob']*100:.1f}%",
        f"",
        f"  Spread prevu:  {p['predicted_spread']:+.1f}",
        f"  Total prevu:   {p['predicted_total']:.1f}",
        f"  Confiance:     {p['confidence']} (accord modeles: {p['model_agreement']*100:.1f}%)",
        f"",
        f"  IC 90%: [{p['confidence_interval_90']['home_win_prob_low']*100:.1f}% — {p['confidence_interval_90']['home_win_prob_high']*100:.1f}%]",
        f"",
        f"  {'─'*65}",
        f"  Modeles individuels:",
    ]

    for model_name, weight in ENSEMBLE_WEIGHTS.items():
        if model_name in p["individual_models"]:
            m = p["individual_models"][model_name]
            prob = m.get("home_win_prob", "N/A")
            prob_str = f"{prob*100:.1f}%" if isinstance(prob, float) else prob
            lines.append(f"    {model_name:<18s} (poids {weight:.0%}): {prob_str}")

    # Monte Carlo details
    mc = p["individual_models"].get("monte_carlo", {})
    if mc.get("percentiles"):
        lines.extend([
            f"",
            f"  Monte Carlo ({MC_SIMULATIONS} sims):",
            f"    Marge: {mc.get('percentiles', {}).get('5th', 'N/A')} a {mc.get('percentiles', {}).get('95th', 'N/A')} (5e-95e percentile)",
        ])

    if mc.get("spread_covers"):
        lines.append(f"    Couverture spread:")
        for spread, prob in list(mc["spread_covers"].items())[:5]:
            lines.append(f"      Home +{spread}: {prob*100:.1f}%")

    lines.append(f"\n{'='*70}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def save_prediction(prediction):
    """Save prediction for backtesting."""
    pred_dir = ROOT / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    game_id = f"{prediction['away_team']}-{prediction['home_team']}-{ts}"

    pred_file = pred_dir / f"pred-{game_id}.json"
    pred_file.write_text(json.dumps(prediction, indent=2, ensure_ascii=False, default=str))

    # Also append to JSONL log
    log_file = pred_dir / "predictions.jsonl"
    with open(log_file, "a") as f:
        compact = {
            "game_id": game_id,
            "home": prediction["home_team"],
            "away": prediction["away_team"],
            "home_prob": prediction["ensemble_home_win_prob"],
            "spread": prediction["predicted_spread"],
            "total": prediction["predicted_total"],
            "confidence": prediction["confidence"],
            "ts": prediction["timestamp"],
        }
        f.write(json.dumps(compact) + "\n")

    return pred_file


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NBA Game Predictor")
    parser.add_argument("--matchup", nargs=2, metavar=("HOME", "AWAY"), help="Predict: HOME AWAY")
    parser.add_argument("--elo", action="store_true", help="Show ELO rankings")
    parser.add_argument("--poisson", action="store_true", help="Use Poisson model only")
    parser.add_argument("--mc", action="store_true", help="Use Monte Carlo only")
    parser.add_argument("--sims", type=int, default=MC_SIMULATIONS, help="Monte Carlo simulations")
    parser.add_argument("--save", action="store_true", help="Save prediction to file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.elo:
        rankings = get_elo_rankings()
        print(f"\nELO Rankings — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
        print("="*50)
        for i, r in enumerate(rankings, 1):
            print(f"  {i:2d}. {r['team']} {r['name']:<30s} {r['elo']:.0f}")

    elif args.matchup:
        home, away = args.matchup

        if args.poisson:
            result = poisson_predict(home, away)
            if result:
                print(json.dumps(result, indent=2))
            else:
                print("Equipe introuvable")

        elif args.mc:
            MC_SIMULATIONS = args.sims
            result = monte_carlo_predict(home, away, args.sims)
            if result:
                print(json.dumps(result, indent=2))
            else:
                print("Equipe introuvable")

        else:
            # Full ensemble
            prediction = ensemble_predict(home, away)
            if args.json:
                print(json.dumps(prediction, indent=2, default=str))
            else:
                print(format_prediction_report(prediction))

            if args.save:
                f = save_prediction(prediction)
                print(f"\nSaved to: {f}")

    else:
        # Demo: predict a few matchups
        matchups = [("BOS", "NYK"), ("OKC", "DEN"), ("LAL", "GSW")]
        for home, away in matchups:
            pred = ensemble_predict(home, away)
            h_name = pred["home_name"]
            a_name = pred["away_name"]
            print(f"\n{a_name} @ {h_name}: "
                  f"{pred['ensemble_home_win_prob']*100:.1f}% home | "
                  f"Spread {pred['predicted_spread']:+.1f} | "
                  f"Total {pred['predicted_total']:.0f} | "
                  f"{pred['confidence']}")
