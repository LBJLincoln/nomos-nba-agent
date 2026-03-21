import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

def compute_head_to_head_record(games: pd.DataFrame, team_id: int, opponent_id: int, season: str = None) -> Dict[str, int]:
    """
    Calculate head-to-head record between two teams.

    Parameters:
    games (pd.DataFrame): Game data with columns ['team_id', 'opponent_id', 'points', 'opponent_points', 'game_date', 'season']
    team_id (int): Team ID
    opponent_id (int): Opponent ID
    season (str): Optional season filter (e.g., '2023-24')

    Returns:
    Dict[str, int]: Dictionary with head-to-head statistics
    """
    team_games = games[(games['team_id'] == team_id) & (games['opponent_id'] == opponent_id)].copy()

    if season:
        team_games = team_games[team_games['season'] == season]

    if team_games.empty:
        return {
            'games_played': 0,
            'team_wins': 0,
            'opponent_wins': 0,
            'team_points': 0,
            'opponent_points': 0,
            'team_win_pct': 0.0,
            'avg_margin': 0.0
        }

    team_games['team_won'] = (team_games['points'] > team_games['opponent_points']).astype(int)
    team_games['margin'] = team_games['points'] - team_games['opponent_points']

    return {
        'games_played': len(team_games),
        'team_wins': team_games['team_won'].sum(),
        'opponent_wins': len(team_games) - team_games['team_won'].sum(),
        'team_points': team_games['points'].sum(),
        'opponent_points': team_games['opponent_points'].sum(),
        'team_win_pct': team_games['team_won'].mean(),
        'avg_margin': team_games['margin'].mean()
    }

def compute_season_series_record(games: pd.DataFrame, team_id: int, season: str = None) -> Dict[str, Dict[str, int]]:
    """
    Calculate season series records for a team against all opponents.

    Parameters:
    games (pd.DataFrame): Game data
    team_id (int): Team ID
    season (str): Optional season filter

    Returns:
    Dict[str, Dict[str, int]]: Dictionary with opponent_id as keys and head-to-head records as values
    """
    team_games = games[games['team_id'] == team_id].copy()

    if season:
        team_games = team_games[team_games['season'] == season]

    opponent_ids = team_games['opponent_id'].unique()
    season_series = {}

    for opponent_id in opponent_ids:
        season_series[str(opponent_id)] = compute_head_to_head_record(games, team_id, opponent_id, season)

    return season_series

def compute_style_matchup_metrics(games: pd.DataFrame, team_id: int, opponent_id: int) -> Dict[str, float]:
    """
    Calculate style matchup metrics between two teams.

    Parameters:
    games (pd.DataFrame): Game data with pace and defensive efficiency columns
    team_id (int): Team ID
    opponent_id (int): Opponent ID

    Returns:
    Dict[str, float]: Dictionary with style matchup metrics
    """
    team_games = games[(games['team_id'] == team_id) & (games['opponent_id'] == opponent_id)].copy()

    if team_games.empty:
        return {
            'pace_diff': 0.0,
            'def_eff_diff': 0.0,
            'style_mismatch_score': 0.0,
            'pace_advantage': 0.0,
            'defense_advantage': 0.0
        }

    team_pace = team_games['pace'].mean()
    opponent_pace = team_games['opponent_pace'].mean()
    team_def_eff = team_games['defensive_efficiency'].mean()
    opponent_def_eff = team_games['opponent_defensive_efficiency'].mean()

    pace_diff = team_pace - opponent_pace
    def_eff_diff = team_def_eff - opponent_def_eff

    # Style mismatch score (higher = more mismatch)
    style_mismatch_score = (np.abs(pace_diff) + np.abs(def_eff_diff)) / 2

    # Advantage indicators
    pace_advantage = 1 if pace_diff > 0 else -1 if pace_diff < 0 else 0
    defense_advantage = 1 if def_eff_diff < 0 else -1 if def_eff_diff > 0 else 0

    return {
        'pace_diff': pace_diff,
        'def_eff_diff': def_eff_diff,
        'style_mismatch_score': style_mismatch_score,
        'pace_advantage': pace_advantage,
        'defense_advantage': defense_advantage
    }

def compute_home_away_splits(games: pd.DataFrame, team_id: int, opponent_id: int) -> Dict[str, Dict[str, float]]:
    """
    Calculate home/away performance splits against a specific opponent.

    Parameters:
    games (pd.DataFrame): Game data with location information
    team_id (int): Team ID
    opponent_id (int): Opponent ID

    Returns:
    Dict[str, Dict[str, float]]: Dictionary with home/away splits
    """
    team_games = games[(games['team_id'] == team_id) & (games['opponent_id'] == opponent_id)].copy()

    if team_games.empty:
        return {
            'home': {'games': 0, 'win_pct': 0.0, 'avg_points': 0.0, 'avg_margin': 0.0},
            'away': {'games': 0, 'win_pct': 0.0, 'avg_points': 0.0, 'avg_margin': 0.0},
            'neutral': {'games': 0, 'win_pct': 0.0, 'avg_points': 0.0, 'avg_margin': 0.0}
        }

    team_games['team_won'] = (team_games['points'] > team_games['opponent_points']).astype(int)
    team_games['margin'] = team_games['points'] - team_games['opponent_points']

    # Split by location
    home_games = team_games[team_games['location'] == 'home']
    away_games = team_games[team_games['location'] == 'away']
    neutral_games = team_games[team_games['location'].isin(['neutral', 'none'])]

    def compute_split_stats(games_subset: pd.DataFrame) -> Dict[str, float]:
        if games_subset.empty:
            return {'games': 0, 'win_pct': 0.0, 'avg_points': 0.0, 'avg_margin': 0.0}

        return {
            'games': len(games_subset),
            'win_pct': games_subset['team_won'].mean(),
            'avg_points': games_subset['points'].mean(),
            'avg_margin': games_subset['margin'].mean()
        }

    return {
        'home': compute_split_stats(home_games),
        'away': compute_split_stats(away_games),
        'neutral': compute_split_stats(neutral_games)
    }

def compute_historical_matchup_trends(games: pd.DataFrame, team_id: int, opponent_id: int, window: int = 5) -> Dict[str, float]:
    """
    Calculate recent trends in head-to-head matchups.

    Parameters:
    games (pd.DataFrame): Game data
    team_id (int): Team ID
    opponent_id (int): Opponent ID
    window (int): Number of recent games to consider

    Returns:
    Dict[str, float]: Dictionary with trend metrics
    """
    team_games = games[(games['team_id'] == team_id) & (games['opponent_id'] == opponent_id)].copy()
    team_games = team_games.sort_values('game_date', ascending=False).head(window)

    if team_games.empty:
        return {
            'recent_win_pct': 0.0,
            'recent_avg_margin': 0.0,
            'trend_direction': 0.0,
            'momentum_score': 0.0
        }

    team_games['team_won'] = (team_games['points'] > team_games['opponent_points']).astype(int)
    team_games['margin'] = team_games['points'] - team_games['opponent_points']

    recent_win_pct = team_games['team_won'].mean()
    recent_avg_margin = team_games['margin'].mean()

    # Trend direction (positive = improving, negative = declining)
    if len(team_games) > 1:
        trend_direction = np.polyfit(np.arange(len(team_games)), team_games['margin'], 1)[0]
    else:
        trend_direction = 0.0

    # Momentum score (weighted recent performance)
    weights = np.arange(1, len(team_games) + 1)
    momentum_score = np.average(team_games['margin'], weights=weights)

    return {
        'recent_win_pct': recent_win_pct,
        'recent_avg_margin': recent_avg_margin,
        'trend_direction': trend_direction,
        'momentum_score': momentum_score
    }

def generate_matchup_profile(games: pd.DataFrame, team_id: int, opponent_id: int, season: str = None) -> Dict[str, any]:
    """
    Generate comprehensive matchup profile between two teams.

    Parameters:
    games (pd.DataFrame): Game data
    team_id (int): Team ID
    opponent_id (int): Opponent ID
    season (str): Optional season filter

    Returns:
    Dict[str, any]: Complete matchup profile
    """
    profile = {
        'team_id': team_id,
        'opponent_id': opponent_id,
        'season': season,
        'head_to_head': compute_head_to_head_record(games, team_id, opponent_id, season),
        'style_matchup': compute_style_matchup_metrics(games, team_id, opponent_id),
        'home_away_splits': compute_home_away_splits(games, team_id, opponent_id),
        'historical_trends': compute_historical_matchup_trends(games, team_id, opponent_id)
    }

    return profile

def compute_matchup_advantage_score(profile: Dict[str, any]) -> float:
    """
    Calculate a single matchup advantage score from profile data.

    Parameters:
    profile (Dict[str, any]): Matchup profile

    Returns:
    float: Advantage score (positive = team advantage, negative = opponent advantage)
    """
    score = 0.0

    # Head-to-head record
    if profile['head_to_head']['games_played'] > 0:
        score += (profile['head_to_head']['team_win_pct'] - 0.5) * 0.3

    # Style matchup
    if profile['style_matchup']['style_mismatch_score'] > 0.5:
        score += profile['style_matchup']['pace_advantage'] * 0.2
        score += profile['style_matchup']['defense_advantage'] * 0.2

    # Home/away advantage
    if profile['home_away_splits']['home']['games'] > 0:
        score += (profile['home_away_splits']['home']['win_pct'] - 0.5) * 0.15
    if profile['home_away_splits']['away']['games'] > 0:
        score += (profile['home_away_splits']['away']['win_pct'] - 0.5) * 0.15

    # Recent trends
    if profile['historical_trends']['recent_win_pct'] > 0.5:
        score += (profile['historical_trends']['recent_win_pct'] - 0.5) * 0.15
    if abs(profile['historical_trends']['trend_direction']) > 0.5:
        score += np.sign(profile['historical_trends']['trend_direction']) * 0.1

    return score

# Example usage:
# games = pd.DataFrame({
#     'team_id': [1,1,1,2,2,2],
#     'opponent_id': [2,2,2,1,1,1],
#     'points': [100, 110, 95, 90, 105, 100],
#     'opponent_points': [90, 100, 105, 100, 95, 110],
#     'game_date': pd.date_range('2024-01-01', periods=6),
#     'season': ['2023-24']*6,
#     'location': ['home', 'away', 'neutral', 'away', 'home', 'neutral'],
#     'pace': [100, 105, 95, 95, 100, 98],
#     'defensive_efficiency': [105, 108, 102, 102, 105, 104],
#     'opponent_pace': [95, 98, 100, 100, 95, 97],
#     'opponent_defensive_efficiency': [102, 105, 108, 108, 102, 105]
# })
#
# profile = generate_matchup_profile(games, team_id=1, opponent_id=2, season='2023-24')
# advantage_score = compute_matchup_advantage_score(profile)
# print(f"Matchup advantage score: {advantage_score:.3f}")

