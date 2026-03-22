import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List

def compute_advanced_momentum_features(games: pd.DataFrame, team_id: int) -> Dict[str, float]:
    """
    Compute advanced momentum features for a team.
    
    Parameters:
    games (pd.DataFrame): Game data with columns ['team_id', 'opponent_id', 'points', 'opponent_points', 'game_date']
    team_id (int): Team ID to analyze
    
    Returns:
    Dict[str, float]: Dictionary of momentum features
    """
    team_games = games[games['team_id'] == team_id].copy()
    team_games['margin'] = team_games['points'] - team_games['opponent_points']
    team_games['result'] = (team_games['margin'] > 0).astype(int)
    
    if len(team_games) < 5:
        return {
            'weighted_win_streak': 0.0,
            'margin_trend': 0.0,
            'avg_margin': 0.0,
            'margin_volatility': 0.0,
            'momentum_power': 0.0,
            'momentum_log': 0.0
        }
    
    # Calculate weighted win streak based on opponent strength
    team_games['opponent_strength'] = games.groupby('opponent_id')['points'].transform('mean') / 100
    team_games['weighted_result'] = team_games['result'] * team_games['opponent_strength']
    
    # Calculate streaks
    team_games['streak'] = (team_games['result'].astype(int).diff().fillna(1)!= 0).cumsum()
    streaks = team_games.groupby('streak')['weighted_result'].agg(['first', 'size', 'sum'])
    
    # Filter only winning streaks
    winning_streaks = streaks[streaks['sum'] == streaks['size']]
    
    if not winning_streaks.empty:
        weighted_streak = (winning_streaks['size'] * winning_streaks['first']).sum()
    else:
        weighted_streak = 0.0
    
    # Recent margin trends (last 5 games)
    recent_games = team_games.tail(5)
    X = np.arange(len(recent_games))
    y = recent_games['margin'].values
    if len(X) > 1:
        A = np.vstack([X, np.ones(len(X))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        slope = 0.0
    
    # Momentum metrics
    momentum = recent_games['margin'].mean()
    volatility = recent_games['margin'].std()
    
    # Advanced momentum features
    momentum_power = weighted_streak ** 1.3 if weighted_streak > 0 else 0.0
    momentum_log = np.log1p(weighted_streak) if weighted_streak > 0 else 0.0
    
    return {
        'weighted_win_streak': weighted_streak,
        'margin_trend': slope,
        'avg_margin': momentum,
        'margin_volatility': volatility,
        'momentum_power': momentum_power,
        'momentum_log': momentum_log
    }

def compute_rest_travel_impact(games: pd.DataFrame, team_id: int, current_date: datetime) -> Dict[str, float]:
    """
    Compute rest and travel impact features.
    
    Parameters:
    games (pd.DataFrame): Game data with columns ['team_id', 'game_date', 'location', 'timezone']
    team_id (int): Team ID to analyze
    current_date (datetime): Current game date
    
    Returns:
    Dict[str, float]: Dictionary of rest/travel features
    """
    team_games = games[games['team_id'] == team_id].copy()
    team_games['game_date'] = pd.to_datetime(team_games['game_date'])
    
    # Calculate rest days since last game
    last_game = team_games[team_games['game_date'] < current_date].sort_values('game_date', ascending=False).head(1)
    
    if last_game.empty:
        rest_days = 7
        is_back_to_back = False
        travel_distance = 0
        timezone_diff = 0
    else:
        last_game_date = last_game['game_date'].iloc[0]
        rest_days = (current_date - last_game_date).days
        is_back_to_back = rest_days == 1
        
        # Mock travel distance (in practice, use real coordinates)
        last_location = last_game['location'].iloc[0]
        current_location = 'Chicago'  # Replace with actual location
        travel_distance = 0 if last_location == current_location else 500
        
        # Timezone difference
        last_timezone = last_game['timezone'].iloc[0]
        current_timezone = -6  # Replace with actual timezone
        timezone_diff = abs(current_timezone - last_timezone)
    
    # Rest quality score
    rest_quality = 1.0
    
    if rest_days < 1:
        rest_quality *= 0.7
    elif rest_days == 1:
        rest_quality *= 0.9
    elif rest_days > 4:
        rest_quality *= 1.1
    
    if is_back_to_back:
        rest_quality *= 0.6
    
    if travel_distance > 1000:
        rest_quality *= 0.8
    elif travel_distance > 500:
        rest_quality *= 0.9
    
    # Timezone adjustment
    timezone_adjustment = max(0.8, 1.0 - (timezone_diff * 0.1))
    
    return {
        'rest_days': rest_days,
        'is_back_to_back': 1.0 if is_back_to_back else 0.0,
        'travel_distance': travel_distance,
        'timezone_adjustment': timezone_adjustment,
        'rest_quality_score': rest_quality
    }

def compute_opponent_strength_decomposition(games: pd.DataFrame, team_id: int, opponent_id: int) -> Dict[str, float]:
    """
    Decompose opponent strength into multiple components.
    
    Parameters:
    games (pd.DataFrame): Game data with team and opponent stats
    team_id (int): Team ID
    opponent_id (int): Opponent ID
    
    Returns:
    Dict[str, float]: Dictionary of opponent strength features
    """
    # Get opponent's recent performance
    opponent_games = games[games['team_id'] == opponent_id].copy()
    
    if len(opponent_games) < 5:
        return {
            'opponent_strength': 0.5,
            'strength_diff': 0.0,
            'strength_ratio': 1.0,
            'is_strength_favored': 0.0,
            'strength_dominance': 0.0
        }
    
    # Calculate opponent strength as win rate
    opponent_games['result'] = (opponent_games['points'] > opponent_games['opponent_points']).astype(int)
    opponent_strength = opponent_games['result'].mean()
    
    # Get team's strength for comparison
    team_games = games[games['team_id'] == team_id].copy()
    team_games['result'] = (team_games['points'] > team_games['opponent_points']).astype(int)
    team_strength = team_games['result'].mean()
    
    strength_diff = team_strength - opponent_strength
    strength_ratio = team_strength / opponent_strength if opponent_strength > 0 else 1.0
    is_strength_favored = 1.0 if team_strength > opponent_strength else 0.0
    strength_dominance = strength_diff
    
    return {
        'opponent_strength': opponent_strength,
        'strength_diff': strength_diff,
        'strength_ratio': strength_ratio,
        'is_strength_favored': is_strength_favored,
        'strength_dominance': strength_dominance
    }

def compute_clutch_time_statistics(games: pd.DataFrame, team_id: int) -> Dict[str, float]:
    """
    Compute clutch time performance statistics.
    
    Parameters:
    games (pd.DataFrame): Game data with quarter and game time information
    team_id (int): Team ID to analyze
    
    Returns:
    Dict[str, float]: Dictionary of clutch time features
    """
    team_games = games[games['team_id'] == team_id].copy()
    
    if len(team_games) < 5:
        return {
            'clutch_opportunity': 0.0,
            'clutch_efficiency': 0.0,
            'late_game_points_rate': 0.0,
            'is_clutch_team': 0.0
        }
    
    # Define clutch time (last 5 minutes of close games in 4th quarter)
    team_games['clutch_time'] = ((team_games['quarter'] == 4) & (team_games['game_time'] <= 5)).astype(int)
    team_games['clutch_margin'] = team_games['points'] - team_games['opponent_points']
    team_games['clutch_close_game'] = (np.abs(team_games['clutch_margin']) <= 5).astype(int)
    
    clutch_opportunity = (team_games['clutch_time'] & team_games['clutch_close_game']).sum()
    total_close_games = team_games['clutch_close_game'].sum()
    
    if total_close_games > 0:
        clutch_opportunity_rate = clutch_opportunity / total_close_games
    else:
        clutch_opportunity_rate = 0.0
    
    # Clutch efficiency
    clutch_games = team_games[team_games['clutch_time'] == 1]
    if len(clutch_games) > 0:
        clutch_efficiency = clutch_games['points'].sum() / (clutch_games['points'].sum() + clutch_games['opponent_points'].sum() + 1)
    else:
        clutch_efficiency = 0.0
    
    # Late game points rate
    late_game_games = team_games[team_games['quarter'] >= 4]
    if len(late_game_games) > 0:
        late_game_points_rate = late_game_games['points'].sum() / late_game_games['game_time'].sum()
    else:
        late_game_points_rate = 0.0
    
    # Clutch team indicator (based on clutch efficiency)
    is_clutch_team = 1.0 if clutch_efficiency > 0.55 else 0.0
    
    return {
        'clutch_opportunity': clutch_opportunity_rate,
        'clutch_efficiency': clutch_efficiency,
        'late_game_points_rate': late_game_points_rate,
        'is_clutch_team': is_clutch_team
    }

def compute_home_court_advantage(games: pd.DataFrame, team_id: int, location: str) -> Dict[str, float]:
    """
    Compute home court advantage features.
    
    Parameters:
    games (pd.DataFrame): Game data with location information
    team_id (int): Team ID to analyze
    location (str): Game location ('home', 'away', or 'neutral')
    
    Returns:
    Dict[str, float]: Dictionary of home court features
    """
    team_games = games[games['team_id'] == team_id].copy()
    
    if len(team_games) < 5:
        return {
            'home_court_impact': 0.0,
            'home_advantage': 0.0,
            'crowd_impact': 1.0,
            'home_court_differential': 0.0
        }
    
    # Home/away splits
    home_games = team_games[team_games['location'] == 'home']
    away_games = team_games[team_games['location'] == 'away']
    
    if len(home_games) > 0 and len(away_games) > 0:
        home_win_rate = (home_games['points'] > home_games['opponent_points']).mean()
        away_win_rate = (away_games['points'] > away_games['opponent_points']).mean()
        home_court_impact = home_win_rate - away_win_rate
    else:
        home_court_impact = 0.0
    
    # Home advantage indicator
    home_advantage = 0.05 if location == 'home' else -0.05 if location == 'away' else 0.0
    
    # Crowd impact (proxy for fan attendance)
    crowd_impact = 1.02 if location == 'home' else 0.98 if location == 'away' else 1.0
    
    # Home court differential (team vs opponent home strength)
    # Mock data - in practice, use actual home/away records
    team_home_strength = 0.6
    opponent_home_strength = 0.5
    home_court_differential = team_home_strength - opponent_home_strength
    
    return {
        'home_court_impact': home_court_impact,
        'home_advantage': home_advantage,
        'crowd_impact': crowd_impact,
        'home_court_differential': home_court_differential
    }

def generate_complete_feature_set(games: pd.DataFrame, team_id: int, opponent_id: int, 
                                   location: str, game_date: str) -> Dict[str, float]:
    """
    Generate complete feature set combining all advanced features.
    
    Parameters:
    games (pd.DataFrame): Game data
    team_id (int): Team ID
    opponent_id (int): Opponent ID
    location (str): Game location
    game_date (str): Game date as string
    
    Returns:
    Dict[str, float]: Complete feature dictionary
    """
    features = {}
    
    # Convert game_date to datetime
    current_date = datetime.strptime(game_date, '%Y-%m-%d')
    
    # Advanced momentum features
    momentum_features = compute_advanced_momentum_features(games, team_id)
    features.update(momentum_features)
    
    # Rest and travel impact
    rest_features = compute_rest_travel_impact(games, team_id, current_date)
    features.update(rest_features)
    
    # Opponent strength decomposition
    opponent_features = compute_opponent_strength_decomposition(games, team_id, opponent_id)
    features.update(opponent_features)
    
    # Clutch time statistics
    clutch_features = compute_clutch_time_statistics(games, team_id)
    features.update(clutch_features)
    
    # Home court advantage
    home_features = compute_home_court_advantage(games, team_id, location)
    features.update(home_features)
    
    # Basic features
    features['is_home'] = 1.0 if location == 'home' else 0.0
    features['is_away'] = 1.0 if location == 'away' else 0.0
    
    return features

# Example usage:
# games = pd.DataFrame({
#     'team_id': [1,1,1,2,2,2],
#     'opponent_id': [2,3,4,1,3,4],
#     'points': [100,110,95,90,105,100],
#     'opponent_points': [90,100,105,100,95,110],
#     'game_date': ['2024-01-01', '2024-01-05', '2024-01-10', '2024-01-02', '2024-01-06', '2024-01-11'],
#     'location': ['home', 'away', 'neutral', 'away', 'home', 'neutral'],
#     'timezone': [-5, -5, -5, -8, -7, -7],
#     'quarter': [4,4,4,4,4,4],
#     'game_time': [48,48,48,48,48,48]
# })
# 
# features = generate_complete_feature_set(games, team_id=1, opponent_id=2, 
#                                          location='home', game_date='2024-01-15')
# print(features)
