#!/usr/bin/env python3
"""
MOVDA Elo Rating System — arXiv:2506.00348
Margin of Victory Differential Analysis

Computes MOVDA ratings for all NBA teams chronologically.
Output: per-game features (movda_home, movda_away, movda_diff, movda_win_prob, mov_surprise_ewm)

Usage:
  python3 scripts/compute_movda_ratings.py          # compute & print summary
  python3 scripts/compute_movda_ratings.py --export  # export to CSV for engine integration
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

# ── Constants ──────────────────────────────────────────────────────
INITIAL_ELO = 1500.0
K_FACTOR = 20.0          # FiveThirtyEight NBA standard
C = 400.0                # Elo scale factor
SEASON_REVERT = 0.25     # Revert 25% toward mean between seasons
LAMBDA_GRID = np.arange(0.1, 3.1, 0.1)  # Grid search for lambda
EWM_ALPHA = 0.3          # EWMA decay for MOV surprise signal


def load_games_from_cache():
    """Load games chronologically from cached JSON files (same as HF Spaces)."""
    from pathlib import Path
    hist_dir = Path(os.path.dirname(__file__)) / ".." / "data" / "historical"
    if not hist_dir.exists():
        hist_dir = Path(os.path.dirname(__file__)) / ".." / "hf-space" / "data" / "historical"

    all_games = []
    season_map = {
        "2018-19": "2018-19", "2019-20": "2019-20", "2020-21": "2020-21",
        "2021-22": "2021-22", "2022-23": "2022-23", "2023-24": "2023-24",
        "2024-25": "2024-25", "2025-26": "2025-26",
    }

    for f in sorted(hist_dir.glob("games-*.json")):
        season = f.stem.replace("games-", "")
        data = json.loads(f.read_text())
        games = data if isinstance(data, list) else data.get("games", [])
        for g in games:
            home_score = g.get("home_score", g.get("home", {}).get("pts"))
            away_score = g.get("away_score", g.get("away", {}).get("pts"))
            if home_score is None or away_score is None:
                continue
            # Resolve team abbreviations
            home_abbrev = g.get("home_abbrev") or g.get("home_team", "")[:3].upper()
            away_abbrev = g.get("away_abbrev") or g.get("away_team", "")[:3].upper()
            all_games.append({
                "date": g.get("game_date", g.get("date", "")),
                "home": home_abbrev,
                "away": away_abbrev,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "season": season,
            })

    all_games.sort(key=lambda g: g["date"])
    return all_games


def standard_elo_pass(games):
    """Run standard Elo to get deltaR for each game (needed for MOVDA parameter fitting)."""
    ratings = defaultdict(lambda: INITIAL_ELO)
    results = []
    prev_season = None

    for g in games:
        # Season revert
        if prev_season and g["season"] != prev_season:
            for team in list(ratings.keys()):
                ratings[team] = INITIAL_ELO + (1 - SEASON_REVERT) * (ratings[team] - INITIAL_ELO)
        prev_season = g["season"]

        home, away = g["home"], g["away"]
        delta_r = ratings[home] - ratings[away]
        e_a = 1.0 / (1.0 + 10.0 ** (-delta_r / C))
        t_mov = g["home_score"] - g["away_score"]
        s_a = 1.0 if t_mov > 0 else (0.0 if t_mov < 0 else 0.5)

        results.append({
            **g,
            "delta_r": delta_r,
            "e_a": e_a,
            "s_a": s_a,
            "t_mov": t_mov,
            "r_home_before": ratings[home],
            "r_away_before": ratings[away],
        })

        # Update ratings
        update = K_FACTOR * (s_a - e_a)
        ratings[home] += update
        ratings[away] -= update

    return results


def e_mov_func(x, alpha, beta, gamma, delta):
    """Expected MOV function (Eq. 4 from paper).
    x[0] = deltaR, x[1] = I_HA (+1 home, -1 away)
    """
    return alpha * np.tanh(beta * x[0]) + gamma + delta * x[1]


def fit_movda_params(elo_results, train_frac=0.7):
    """Fit alpha, beta, gamma, delta on training data."""
    n_train = int(len(elo_results) * train_frac)
    train = elo_results[:n_train]

    delta_r_arr = np.array([g["delta_r"] for g in train])
    i_ha_arr = np.ones(len(train))  # All from home perspective
    t_mov_arr = np.array([g["t_mov"] for g in train], dtype=float)

    x_data = np.array([delta_r_arr, i_ha_arr])

    try:
        popt, _ = curve_fit(
            e_mov_func, x_data, t_mov_arr,
            p0=[15.0, 0.005, 0.0, 3.0],  # Initial guesses
            maxfev=10000,
        )
        alpha, beta, gamma, delta_param = popt
        return {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta_param}
    except Exception as e:
        print(f"WARNING: curve_fit failed: {e}. Using defaults.")
        return {"alpha": 15.0, "beta": 0.004, "gamma": 0.0, "delta": 3.0}


def run_movda(games, params, lam):
    """Run full MOVDA rating system with given parameters and lambda."""
    alpha, beta, gamma, delta_param = params["alpha"], params["beta"], params["gamma"], params["delta"]

    ratings = defaultdict(lambda: INITIAL_ELO)
    mov_surprise_ewm = defaultdict(float)  # Per-team EWMA of MOV surprise
    results = []
    prev_season = None

    for g in games:
        # Season revert
        if prev_season and g["season"] != prev_season:
            for team in list(ratings.keys()):
                ratings[team] = INITIAL_ELO + (1 - SEASON_REVERT) * (ratings[team] - INITIAL_ELO)
        prev_season = g["season"]

        home, away = g["home"], g["away"]
        delta_r = ratings[home] - ratings[away]

        # Standard Elo win probability
        e_a = 1.0 / (1.0 + 10.0 ** (-delta_r / C))

        # Expected MOV (Eq. 4)
        e_mov = alpha * np.tanh(beta * delta_r) + gamma + delta_param * 1.0  # I_HA=+1 for home

        # Actual MOV
        t_mov = g["home_score"] - g["away_score"]

        # MOV deviation (Eq. 7) — the surprise signal
        delta_mov = t_mov - e_mov

        # Binary outcome
        s_a = 1.0 if t_mov > 0 else (0.0 if t_mov < 0 else 0.5)

        # Store features BEFORE updating
        results.append({
            "date": g["date"],
            "home": home,
            "away": away,
            "movda_home": ratings[home],
            "movda_away": ratings[away],
            "movda_diff": delta_r,
            "movda_win_prob": e_a,
            "mov_surprise": delta_mov,
            "mov_surprise_ewm_home": mov_surprise_ewm[home],
            "mov_surprise_ewm_away": mov_surprise_ewm[away],
            "e_mov": e_mov,
            "t_mov": t_mov,
            "s_a": s_a,
            "e_a": e_a,
        })

        # MOVDA update (Eq. 8-9)
        elo_update = K_FACTOR * (s_a - e_a)
        movda_update = elo_update + lam * delta_mov
        ratings[home] += movda_update
        ratings[away] -= movda_update

        # Update EWMA of MOV surprise per team
        mov_surprise_ewm[home] = EWM_ALPHA * delta_mov + (1 - EWM_ALPHA) * mov_surprise_ewm[home]
        mov_surprise_ewm[away] = EWM_ALPHA * (-delta_mov) + (1 - EWM_ALPHA) * mov_surprise_ewm[away]

    return results


def evaluate_brier(results, start_idx, end_idx):
    """Compute Brier score on a slice of results."""
    subset = results[start_idx:end_idx]
    if not subset:
        return 1.0
    brier = sum((r["e_a"] - r["s_a"]) ** 2 for r in subset) / len(subset)
    return brier


def grid_search_lambda(games, params, train_frac=0.7, val_frac=0.2):
    """Grid search lambda on validation set."""
    n_train = int(len(games) * train_frac)
    n_val = int(len(games) * (train_frac + val_frac))

    best_lam = 0.0
    best_brier = 1.0

    for lam in LAMBDA_GRID:
        results = run_movda(games, params, lam)
        brier = evaluate_brier(results, n_train, n_val)
        if brier < best_brier:
            best_brier = brier
            best_lam = lam

    # Also test lambda=0 (standard Elo)
    results_elo = run_movda(games, params, 0.0)
    brier_elo = evaluate_brier(results_elo, n_train, n_val)

    print(f"  Standard Elo (lambda=0): Brier={brier_elo:.5f}")
    print(f"  Best MOVDA (lambda={best_lam:.1f}): Brier={best_brier:.5f}")
    print(f"  Delta: {brier_elo - best_brier:+.5f} ({(brier_elo - best_brier)/brier_elo*100:+.2f}%)")

    return best_lam


def main():
    parser = argparse.ArgumentParser(description="Compute MOVDA ratings for NBA games")
    parser.add_argument("--export", action="store_true", help="Export to CSV")
    args = parser.parse_args()

    print("=== MOVDA Elo Rating System (arXiv:2506.00348) ===\n")

    # Step 1: Load games
    print("[1/5] Loading games from Supabase...")
    games = load_games_from_cache()
    print(f"  {len(games)} games loaded")

    # Step 2: Standard Elo pass (for parameter fitting)
    print("[2/5] Running standard Elo pass...")
    elo_results = standard_elo_pass(games)

    # Step 3: Fit MOVDA parameters
    print("[3/5] Fitting MOVDA parameters (alpha, beta, gamma, delta)...")
    params = fit_movda_params(elo_results)
    print(f"  alpha={params['alpha']:.4f} (amplitude)")
    print(f"  beta={params['beta']:.6f} (steepness)")
    print(f"  gamma={params['gamma']:.4f} (offset)")
    print(f"  delta={params['delta']:.4f} (home advantage)")

    # Step 4: Grid search lambda
    print("[4/5] Grid searching lambda on validation set...")
    best_lambda = grid_search_lambda(games, params)
    print(f"  Best lambda: {best_lambda:.1f}")

    # Step 5: Final run with best lambda
    print("[5/5] Running MOVDA with optimal parameters...")
    results = run_movda(games, params, best_lambda)

    # Summary
    n_test = int(len(results) * 0.1)
    test_brier = evaluate_brier(results, len(results) - n_test, len(results))
    elo_results_final = run_movda(games, params, 0.0)
    elo_test_brier = evaluate_brier(elo_results_final, len(results) - n_test, len(results))

    print(f"\n=== RESULTS (test set: last {n_test} games) ===")
    print(f"  Standard Elo Brier: {elo_test_brier:.5f}")
    print(f"  MOVDA Brier:        {test_brier:.5f}")
    print(f"  Improvement:        {elo_test_brier - test_brier:+.5f} ({(elo_test_brier - test_brier)/elo_test_brier*100:+.2f}%)")

    # Features for engine integration
    print(f"\n  Features per game: movda_home, movda_away, movda_diff, movda_win_prob, mov_surprise_ewm_home, mov_surprise_ewm_away")
    print(f"  Total games with ratings: {len(results)}")

    if args.export:
        import csv
        out_path = os.path.join(os.path.dirname(__file__), "..", "data", "results", "movda-ratings.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "date", "home", "away",
                "movda_home", "movda_away", "movda_diff", "movda_win_prob",
                "mov_surprise", "mov_surprise_ewm_home", "mov_surprise_ewm_away",
                "e_mov", "t_mov",
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({k: round(v, 6) if isinstance(v, float) else v
                                 for k, v in r.items() if k in writer.fieldnames})
        print(f"\n  Exported to: {out_path}")

    # Save params for engine integration
    params_path = os.path.join(os.path.dirname(__file__), "..", "data", "results", "movda-params.json")
    with open(params_path, "w") as f:
        json.dump({
            "alpha": params["alpha"],
            "beta": params["beta"],
            "gamma": params["gamma"],
            "delta": params["delta"],
            "lambda": best_lambda,
            "k_factor": K_FACTOR,
            "c": C,
            "season_revert": SEASON_REVERT,
            "ewm_alpha": EWM_ALPHA,
            "n_games": len(games),
            "test_brier_elo": elo_test_brier,
            "test_brier_movda": test_brier,
        }, f, indent=2)
    print(f"  Params saved to: {params_path}")


if __name__ == "__main__":
    main()
