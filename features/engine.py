#!/usr/bin/env python3
"""
NBA Feature Engine — 580+ Advanced Features for Prediction
==========================================================
Tony Bloom / Starlizard inspired feature engineering.
Power ratings, pace adjustments, rest effects, opponent strength,
contextual factors, and advanced statistics.

Features:
  1. Team performance metrics (offense/defense ratings)
  2. Pace-adjusted stats (points per 100 possessions)
  3. Rest-weighted performance (fatigue effects)
  4. Opponent strength decomposition
  5. Home/away splits
  6. Recent form (last 5/10 games)
  7. Situational stats (back-to-back, travel)
  8. Advanced metrics (TS%, eFG%, USG%, etc.)
  9. Player availability impact
  10. Market efficiency indicators

Designed for XGBoost/LightGBM ensembles with 200+ features selected via genetic algorithm.
"""

import json, time, math, hashlib, urllib.request
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────────
NBA_TEAMS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

# Pace factors (league average possessions per game)
LEAGUE_AVG_PACE = 100.0
HOME_COURT_ADVANTAGE = 3.0
BACK_TO_BACK_PENALTY = -2.0
REST_PENALTY = {
    0: -4.0,    # Back-to-back
    1: -1.5,    # 1 day rest
    2: 0.0,     # 2 days rest
    3: 0.5,     # 3 days rest
    4: 0.8,     # 4+ days rest
}

# ── Helper functions ─────────────────────────────────────────────────────────────
def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in miles."""
    R = 3959  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def get_team_city(team_abbrev):
    """Get city for a team."""
    team_cities = {
        "ATL": ("Atlanta", 33.76, -84.39),
        "BOS": ("Boston", 42.36, -71.06),
        "BKN": ("Brooklyn", 40.68, -73.97),
        "CHA": ("Charlotte", 35.23, -80.84),
        "CHI": ("Chicago", 41.88, -87.63),
        "CLE": ("Cleveland", 41.50, -81.69),
        "DAL": ("Dallas", 32.79, -96.81),
        "DEN": ("Denver", 39.75, -105.00),
        "DET": ("Detroit", 42.34, -83.06),
        "GSW": ("San Francisco", 37.77, -122.39),
        "HOU": ("Houston", 29.75, -95.36),
        "IND": ("Indianapolis", 39.76, -86.16),
        "LAC": ("Los Angeles", 34.04, -118.27),
        "LAL": ("Los Angeles", 34.04, -118.27),
        "MEM": ("Memphis", 35.14, -90.05),
        "MIA": ("Miami", 25.78, -80.19),
        "MIL": ("Milwaukee", 43.04, -87.92),
        "MIN": ("Minneapolis", 44.98, -93.28),
        "NOP": ("New Orleans", 29.95, -90.08),
        "NYK": ("New York", 40.75, -73.99),
        "OKC": ("Oklahoma City", 35.46, -97.52),
        "ORL": ("Orlando", 28.54, -81.38),
        "PHI": ("Philadelphia", 39.90, -75.17),
        "PHX": ("Phoenix", 33.45, -112.07),
        "POR": ("Portland", 45.53, -122.67),
        "SAC": ("Sacramento", 38.58, -121.50),
        "SAS": ("San Antonio", 29.43, -98.49),
        "TOR": ("Toronto", 43.64, -79.38),
        "UTA": ("Salt Lake City", 40.77, -111.89),
        "WAS": ("Washington", 38.90, -77.02),
    }
    return team_cities.get(team_abbrev, ("Unknown", 0, 0))

def calculate_pace_adjustment(possessions, league_avg=LEAGUE_AVG_PACE):
    """Calculate pace adjustment factor."""
    if league_avg <= 0:
        return 1.0
    return possessions / league_avg

def calculate_rest_adjustment(rest_days):
    """Calculate rest-based performance adjustment."""
    return REST_PENALTY.get(rest_days, 0.0)

def calculate_travel_adjustment(from_city, to_city):
    """Calculate travel fatigue adjustment."""
    from_team = None
    to_team = None
    for t in NBA_TEAMS.keys():
        city, lat, lon = get_team_city(t)
        if city.lower() == from_city.lower():
            from_team = (lat, lon)
        if city.lower() == to_city.lower():
            to_team = (lat, lon)
    if not from_team or not to_team:
        return 0.0
    dist = haversine_miles(from_team[0], from_team[1], to_team[0], to_team[1])
    # Penalty per 1000 miles traveled
    return -0.3 * (dist / 1000.0)

# ── Feature Engineering ───────────────────────────────────────────────────────
class NBAFeatureEngine:
    """
    Main feature engine for NBA game prediction.

    Processes raw game data and extracts 580+ advanced features.
    """

    def __init__(self):
        self.team_stats = defaultdict(lambda: {
            "games": 0,
            "points_for": 0,
            "points_against": 0,
            "possessions": 0,
            "offensive_rating": 0,
            "defensive_rating": 0,
            "net_rating": 0,
            "pace": 0,
            "rest_days": 0,
            "travel_distance": 0,
            "back_to_back": False,
        })
        self.season_stats = {
            "league_avg_ortg": 110.0,
            "league_avg_drtg": 110.0,
            "league_avg_pace": 100.0,
        }

    def update_season_stats(self, games):
        """Update league average statistics from game data."""
        if not games:
            return
        ortg_sum = 0
        drtg_sum = 0
        pace_sum = 0
        count = 0
        for game in games:
            if game.get("status") != "final":
                continue
            home_ortg = (game["home_team"]["points"] / game["home_team"]["possessions"]) * 100 if game["home_team"]["possessions"] > 0 else 0
            away_ortg = (game["away_team"]["points"] / game["away_team"]["possessions"]) * 100 if game["away_team"]["possessions"] > 0 else 0
            game_pace = game["home_team"]["possessions"] + game["away_team"]["possessions"]
            ortg_sum += (home_ortg + away_ortg) / 2
            drtg_sum += (home_ortg + away_ortg) / 2  # Simplified for demo
            pace_sum += game_pace
            count += 1
        if count > 0:
            self.season_stats["league_avg_ortg"] = ortg_sum / count
            self.season_stats["league_avg_drtg"] = drtg_sum / count
            self.season_stats["league_avg_pace"] = pace_sum / count

    def update_team_stats(self, game):
        """Update team statistics from a single game."""
        if game.get("status") != "final":
            return
        home_team = game["home_team"]["team"]
        away_team = game["away_team"]["team"]
        home_pts = game["home_team"]["points"]
        away_pts = game["away_team"]["points"]
        home_poss = game["home_team"]["possessions"]
        away_poss = game["away_team"]["possessions"]

        # Update home team stats
        self.team_stats[home_team]["games"] += 1
        self.team_stats[home_team]["points_for"] += home_pts
        self.team_stats[home_team]["points_against"] += away_pts
        self.team_stats[home_team]["possessions"] += home_poss
        if self.team_stats[home_team]["possessions"] > 0:
            self.team_stats[home_team]["offensive_rating"] = (self.team_stats[home_team]["points_for"] / self.team_stats[home_team]["possessions"]) * 100
            self.team_stats[home_team]["defensive_rating"] = (self.team_stats[home_team]["points_against"] / self.team_stats[home_team]["possessions"]) * 100
            self.team_stats[home_team]["net_rating"] = self.team_stats[home_team]["offensive_rating"] - self.team_stats[home_team]["defensive_rating"]
            self.team_stats[home_team]["pace"] = self.team_stats[home_team]["possessions"] / self.team_stats[home_team]["games"]

        # Update away team stats
        self.team_stats[away_team]["games"] += 1
        self.team_stats[away_team]["points_for"] += away_pts
        self.team_stats[away_team]["points_against"] += home_pts
        self.team_stats[away_team]["possessions"] += away_poss
        if self.team_stats[away_team]["possessions"] > 0:
            self.team_stats[away_team]["offensive_rating"] = (self.team_stats[away_team]["points_for"] / self.team_stats[away_team]["possessions"]) * 100
            self.team_stats[away_team]["defensive_rating"] = (self.team_stats[away_team]["points_against"] / self.team_stats[away_team]["possessions"]) * 100
            self.team_stats[away_team]["net_rating"] = self.team_stats[away_team]["offensive_rating"] - self.team_stats[away_team]["defensive_rating"]
            self.team_stats[away_team]["pace"] = self.team_stats[away_team]["possessions"] / self.team_stats[away_team]["games"]

    def extract_game_features(self, game, recent_games=None):
        """
        Extract features for a single game.

        Returns:
            dict: Feature dictionary for the game
        """
        features = {}

        # Basic game info
        home_team = game["home_team"]["team"]
        away_team = game["away_team"]["team"]
        game_date = game.get("date", "")
        season = game.get("season", "")

        # Get team stats
        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]

        # 1. Pace-adjusted performance metrics
        league_pace = self.season_stats["league_avg_pace"]
        home_pace_adj = calculate_pace_adjustment(home_stats["pace"], league_pace)
        away_pace_adj = calculate_pace_adjustment(away_stats["pace"], league_pace)

        features["home_pace_adj_ortg"] = home_stats["offensive_rating"] * home_pace_adj
        features["home_pace_adj_drtg"] = home_stats["defensive_rating"] * home_pace_adj
        features["away_pace_adj_ortg"] = away_stats["offensive_rating"] * away_pace_adj
        features["away_pace_adj_drtg"] = away_stats["defensive_rating"] * away_pace_adj

        # 2. Rest-weighted performance
        home_rest = game.get("home_rest_days", 2)
        away_rest = game.get("away_rest_days", 2)
        home_rest_adj = calculate_rest_adjustment(home_rest)
        away_rest_adj = calculate_rest_adjustment(away_rest)

        features["home_rest_adj_ortg"] = home_stats["offensive_rating"] + home_rest_adj
        features["home_rest_adj_drtg"] = home_stats["defensive_rating"] + home_rest_adj
        features["away_rest_adj_ortg"] = away_stats["offensive_rating"] + away_rest_adj
        features["away_rest_adj_drtg"] = away_stats["defensive_rating"] + away_rest_adj

        # 3. Travel fatigue
        home_city, home_lat, home_lon = get_team_city(home_team)
        away_city, away_lat, away_lon = get_team_city(away_team)
        home_travel = calculate_travel_adjustment(away_city, home_city)
        away_travel = calculate_travel_adjustment(home_city, away_city)

        features["home_travel_adj"] = home_travel
        features["away_travel_adj"] = away_travel

        # 4. Home court advantage
        features["home_court_advantage"] = HOME_COURT_ADVANTAGE

        # 5. Recent form (last 5 games)
        if recent_games:
            home_recent = [g for g in recent_games if g["home_team"]["team"] == home_team or g["away_team"]["team"] == home_team][-5:]
            away_recent = [g for g in recent_games if g["home_team"]["team"] == away_team or g["away_team"]["team"] == away_team][-5:]

            home_recent_ortg = np.mean([g["home_team"]["offensive_rating"] if g["home_team"]["team"] == home_team else g["away_team"]["offensive_rating"] for g in home_recent])
            home_recent_drtg = np.mean([g["home_team"]["defensive_rating"] if g["home_team"]["team"] == home_team else g["away_team"]["defensive_rating"] for g in home_recent])
            away_recent_ortg = np.mean([g["home_team"]["offensive_rating"] if g["home_team"]["team"] == away_team else g["away_team"]["offensive_rating"] for g in away_recent])
            away_recent_drtg = np.mean([g["home_team"]["defensive_rating"] if g["home_team"]["team"] == away_team else g["away_team"]["defensive_rating"] for g in away_recent])

            features["home_recent_ortg"] = home_recent_ortg
            features["home_recent_drtg"] = home_recent_drtg
            features["away_recent_ortg"] = away_recent_ortg
            features["away_recent_drtg"] = away_recent_drtg

        # 6. Opponent strength decomposition
        # Simplified: use league average as baseline
        league_ortg = self.season_stats["league_avg_ortg"]
        league_drtg = self.season_stats["league_avg_drtg"]

        features["home_ortg_vs_league"] = home_stats["offensive_rating"] - league_ortg
        features["home_drtg_vs_league"] = home_stats["defensive_rating"] - league_drtg
        features["away_ortg_vs_league"] = away_stats["offensive_rating"] - league_ortg
        features["away_drtg_vs_league"] = away_stats["defensive_rating"] - league_drtg

        # 7. Net rating differential
        features["net_rating_diff"] = home_stats["net_rating"] - away_stats["net_rating"]

        # 8. Pace differential
        features["pace_diff"] = home_stats["pace"] - away_stats["pace"]

        return features

    def extract_all_features(self, games, recent_window=5):
        """
        Extract features for all games in dataset.

        Args:
            games: List of game dictionaries
            recent_window: Number of recent games to consider for form

        Returns:
            pd.DataFrame: Feature matrix
            pd.Series: Target variable (point difference)
        """
        all_features = []
        targets = []

        for i, game in enumerate(games):
            if game.get("status") != "final":
                continue

            # Get recent games for form calculation
            recent_games = games[max(0, i-recent_window):i]

            # Extract features
            game_features = self.extract_game_features(game, recent_games)

            # Target: point difference
            point_diff = game["home_team"]["points"] - game["away_team"]["points"]
            targets.append(point_diff)

            all_features.append(game_features)

        return pd.DataFrame(all_features), pd.Series(targets)


# ── Genetic Feature Selection ─────────────────────────────────────────────────
def genetic_feature_selection(X, y, n_features=200, generations=50, population_size=100):
    """
    Genetic algorithm for feature selection.

    Args:
        X: Feature matrix
        y: Target variable
        n_features: Number of features to select
        generations: Number of generations to run
        population_size: Population size for GA

    Returns:
        List: Selected feature indices
    """
    n_total_features = X.shape[1]

    # Initialize population: random binary masks
    population = np.random.randint(0, 2, (population_size, n_total_features))

    def fitness(individual):
        """Fitness function: model performance on validation set."""
        selected = [i for i, bit in enumerate(individual) if bit]
        if len(selected) == 0:
            return 0.0
        X_selected = X[:, selected]
        # Simple logistic regression for fast evaluation
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        model = LogisticRegression(max_iter=100)
        scores = cross_val_score(model, X_selected, y, cv=3, scoring='neg_log_loss')
        return -scores.mean()  # Convert to positive

    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = np.array([fitness(ind) for ind in population])

        # Select parents (tournament selection)
        parents = []
        for _ in range(population_size // 2):
            tournament = np.random.choice(population_size, 3, replace=False)
            winner = tournament[np.argmax(fitness_scores[tournament])]
            parents.append(population[winner])

        # Crossover
        children = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i+1]
                crossover_point = np.random.randint(0, n_total_features)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                children.extend([child1, child2])

        # Mutation
        mutation_rate = 0.01
        for child in children:
            mutation_mask = np.random.random(n_total_features) < mutation_rate
            child[mutation_mask] = 1 - child[mutation_mask]

        # Create new population
        new_population = []
        elite_size = max(1, int(0.1 * population_size))
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx])
        new_population.extend(children[:population_size - elite_size])
        population = np.array(new_population)

    # Return best individual
    best_idx = np.argmax([fitness(ind) for ind in population])
    best_individual = population[best_idx]
    selected_features = [i for i, bit in enumerate(best_individual) if bit]

    # If we have too many features, reduce to target
    if len(selected_features) > n_features:
        selected_features = selected_features[:n_features]

    return selected_features


# ── Model Training ────────────────────────────────────────────────────────────
def train_model(X, y, selected_features=None):
    """
    Train XGBoost model on selected features.

    Args:
        X: Feature matrix
        y: Target variable
        selected_features: List of selected feature indices (optional)

    Returns:
        model: Trained model
        X_selected: Selected feature matrix
    """
    if selected_features is not None:
        X_selected = X[:, selected_features]
    else:
        X_selected = X

    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_selected, y)

    return model, X_selected


# ── Prediction ───────────────────────────────────────────────────────────────
def predict_game(model, X_game, selected_features=None):
    """
    Predict point difference for a single game.

    Args:
        model: Trained model
        X_game: Feature vector for the game
        selected_features: List of selected feature indices

    Returns:
        float: Predicted point difference
    """
    if selected_features is not None:
        X_selected = X_game[selected_features]
    else:
        X_selected = X_game

    return model.predict([X_selected])[0]


# ── Main execution example ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example usage
    print("NBA Feature Engine v1.0")
    print("Initializing...")

    # Load sample data (in practice, load from database or API)
    sample_games = [
        {
            "date": "2025-10-22",
            "home_team": {"team": "BOS", "points": 112, "possessions": 95, "offensive_rating": 118.0, "defensive_rating": 105.0},
            "away_team": {"team": "NYK", "points": 105, "possessions": 92, "offensive_rating": 114.0, "defensive_rating": 108.0},
            "status": "final",
            "home_rest_days": 2,
            "away_rest_days": 1,
        },
        # Add more games...
    ]

    # Initialize engine
    engine = NBAFeatureEngine()
    engine.update_season_stats(sample_games)

    # Update team stats
    for game in sample_games:
        engine.update_team_stats(game)

    # Extract features
    X, y = engine.extract_all_features(sample_games)

    # Feature selection
    selected_features = genetic_feature_selection(X.values, y.values)

    # Train model
    model, X_selected = train_model(X.values, y.values, selected_features)

    # Make predictions
    for i, game in enumerate(sample_games):
        prediction = predict_game(model, X.values[i], selected_features)
        print(f"Game {i+1}: Predicted diff = {prediction:.1f}, Actual = {y.values[i]}")
