import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from features.helpers.pace_adjusted import compute_pace_adjusted

def compute_rest_days_since_last_game(games: pd.DataFrame, team_id: int, current_date: datetime) -> int:
    """
    Calculate days since last game for a team.
    
    Parameters:
    games (pd.DataFrame): DataFrame with columns ['team_id', 'game_date']
    team_id (int): Team ID to analyze
    current_date (datetime): Current game date
    
    Returns:
    int: Days since last game (0 = same day, 1 = one day rest, etc.)
    """
    team_games = games[games['team_id'] == team_id].copy()
    team_games['game_date'] = pd.to_datetime(team_games['game_date'])
    
    # Get last game before current date
    last_game = team_games[team_games['game_date'] < current_date].sort_values('game_date', ascending=False).head(1)
    
    if last_game.empty:
        return 7  # Default to 7 days if no previous game
    
    last_game_date = last_game['game_date'].iloc[0]
    rest_days = (current_date - last_game_date).days
    
    return rest_days

def detect_back_to_back_games(games: pd.DataFrame, team_id: int, current_date: datetime) -> bool:
    """
    Detect if team is playing back-to-back games.
    
    Parameters:
    games (pd.DataFrame): DataFrame with columns ['team_id', 'game_date']
    team_id (int): Team ID to analyze
    current_date (datetime): Current game date
    
    Returns:
    bool: True if back-to-back, False otherwise
    """
    team_games = games[games['team_id'] == team_id].copy()
    team_games['game_date'] = pd.to_datetime(team_games['game_date'])
    
    # Get last game
    last_game = team_games[team_games['game_date'] < current_date].sort_values('game_date', ascending=False).head(1)
    
    if last_game.empty:
        return False
    
    last_game_date = last_game['game_date'].iloc[0]
    return (current_date - last_game_date).days == 1

def compute_travel_distance_impact(games: pd.DataFrame, team_id: int, current_game: pd.Series) -> float:
    """
    Calculate travel distance impact on performance.
    
    Parameters:
    games (pd.DataFrame): DataFrame with columns ['team_id', 'game_date', 'location']
    team_id (int): Team ID to analyze
    current_game (pd.Series): Current game with 'game_date' and 'location'
    
    Returns:
    float: Travel distance in miles (0 = no travel, higher = more fatigue)
    """
    team_games = games[games['team_id'] == team_id].copy()
    team_games['game_date'] = pd.to_datetime(team_games['game_date'])
    
    # Get last game location
    last_game = team_games[team_games['game_date'] < current_game['game_date']].sort_values('game_date', ascending=False).head(1)
    
    if last_game.empty:
        return 0.0
    
    # Simple city-based distance (replace with actual haversine if coordinates available)
    last_location = last_game['location'].iloc[0]
    current_location = current_game['location']
    
    # Mock distance dictionary (in practice, use real coordinates)
    distances = {
        ('New York', 'Boston'): 215,
        ('Boston', 'New York'): 215,
        ('Los Angeles', 'Phoenix'): 370,
        ('Phoenix', 'Los Angeles'): 370,
        ('Miami', 'Atlanta'): 660,
        ('Atlanta', 'Miami'): 660,
        ('Chicago', 'Cleveland'): 350,
        ('Cleveland', 'Chicago'): 350,
    }
    
    distance = distances.get((last_location, current_location), 500)  # Default to 500 miles
    
    return distance

def compute_timezone_adjustment_factor(games: pd.DataFrame, team_id: int, current_game: pd.Series) -> float:
    """
    Calculate timezone adjustment factor for performance impact.
    
    Parameters:
    games (pd.DataFrame): DataFrame with columns ['team_id', 'game_date', 'timezone']
    team_id (int): Team ID to analyze
    current_game (pd.Series): Current game with 'timezone' and 'game_time'
    
    Returns:
    float: Adjustment factor (1.0 = no adjustment, <1 = negative impact, >1 = positive impact)
    """
    team_games = games[games['team_id'] == team_id].copy()
    team_games['game_date'] = pd.to_datetime(team_games['game_date'])
    
    # Get last game timezone
    last_game = team_games[team_games['game_date'] < current_game['game_date']].sort_values('game_date', ascending=False).head(1)
    
    if last_game.empty:
        return 1.0
    
    last_timezone = last_game['timezone'].iloc[0]
    current_timezone = current_game['timezone']
    
    # Calculate timezone difference
    timezone_diff = abs(current_timezone - last_timezone)
    
    # Adjustment factor: 1.0 = no change, decreases with timezone difference
    adjustment = max(0.8, 1.0 - (timezone_diff * 0.1))
    
    return adjustment

def compute_rest_impact_features(games: pd.DataFrame, team_id: int, current_game: pd.Series) -> dict:
    """
    Compute comprehensive rest impact features for a team.
    
    Parameters:
    games (pd.DataFrame): DataFrame containing game history
    team_id (int): Team ID to analyze
    current_game (pd.Series): Current game data
    
    Returns:
    dict: Dictionary containing all rest impact features
    """
    features = {}
    
    # Calculate rest days
    current_date = pd.to_datetime(current_game['game_date'])
    features['rest_days'] = compute_rest_days_since_last_game(games, team_id, current_date)
    
    # Detect back-to-back
    features['is_back_to_back'] = detect_back_to_back_games(games, team_id, current_date)
    
    # Travel distance
    features['travel_distance'] = compute_travel_distance_impact(games, team_id, current_game)
    
    # Timezone adjustment
    features['timezone_adjustment'] = compute_timezone_adjustment_factor(games, team_id, current_game)
    
    # Rest quality score (composite metric)
    rest_quality = 1.0
    
    if features['rest_days'] < 1:
        rest_quality *= 0.7  # No rest
    elif features['rest_days'] == 1:
        rest_quality *= 0.9  # One day rest
    elif features['rest_days'] > 4:
        rest_quality *= 1.1  # Well rested
    
    if features['is_back_to_back']:
        rest_quality *= 0.6  # Back-to-back penalty
    
    if features['travel_distance'] > 1000:
        rest_quality *= 0.8  # Long travel penalty
    elif features['travel_distance'] > 500:
        rest_quality *= 0.9  # Medium travel penalty
    
    features['rest_quality_score'] = rest_quality
    
    return features

# Example usage:
# games = pd.DataFrame({
#     'team_id': [1,1,1,2,2,2],
#     'game_date': ['2024-01-01', '2024-01-05', '2024-01-10', '2024-01-02', '2024-01-06', '2024-01-11'],
#     'location': ['New York', 'Boston', 'Miami', 'Los Angeles', 'Phoenix', 'Denver'],
#     'timezone': [-5, -5, -5, -8, -7, -7]
# })
# current_game = {'game_date': '2024-01-15', 'location': 'Chicago', 'timezone': -6}
# rest_features = compute_rest_impact_features(games, team_id=1, current_game=current_game)

