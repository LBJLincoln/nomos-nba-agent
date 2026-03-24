import numpy as np
import pandas as pd
from scipy.stats import zscore

def compute_win_streak_weighted_by_opponent_strength(games: pd.DataFrame, team_id: int) -> float:
    """
    Compute weighted win streak based on opponent strength.

    Parameters:
    games (pd.DataFrame): DataFrame containing game results with columns:
        - 'team_id', 'opponent_id', 'points', 'opponent_points', 'result' (1=win, 0=loss)
        - 'opponent_strength' (pre-computed strength metric)
    team_id (int): ID of the team to analyze

    Returns:
    float: Weighted win streak momentum score
    """
    team_games = games[games['team_id'] == team_id].copy()

    # Calculate streak
    team_games['streak'] = (team_games['result'].astype(int).diff().fillna(1)!= 0).cumsum()
    streaks = team_games.groupby('streak')['result'].agg(['first', 'size', 'sum'])

    # Filter only winning streaks
    winning_streaks = streaks[streaks['sum'] == streaks['size']]

    if winning_streaks.empty:
        return 0.0

    # Get opponent strengths for each streak
    streak_opponent_strengths = []
    for idx, row in winning_streaks.iterrows():
        streak_games = team_games[team_games['streak'] == idx]
        avg_opponent_strength = streak_games['opponent_strength'].mean()
        streak_opponent_strengths.append(avg_opponent_strength)

    # Weight by streak length and opponent strength
    weighted_streaks = winning_streaks['size'].values * np.array(streak_opponent_strengths)

    return weighted_streaks.sum()

def compute_recent_margin_trends(games: pd.DataFrame, team_id: int, n_games: int = 5) -> dict:
    """
    Compute recent margin trends and momentum indicators.

    Parameters:
    games (pd.DataFrame): DataFrame containing game results with columns:
        - 'team_id', 'points', 'opponent_points', 'result'
    team_id (int): ID of the team to analyze
    n_games (int): Number of recent games to consider

    Returns:
    dict: Dictionary containing trend metrics
    """
    team_games = games[games['team_id'] == team_id].copy()

    if len(team_games) < n_games:
        return {'trend': 0, 'momentum': 0, 'std_dev': 0}

    recent_games = team_games.tail(n_games).copy()
    recent_games['margin'] = recent_games['points'] - recent_games['opponent_points']

    # Linear trend (slope of margin over time)
    X = np.arange(len(recent_games))
    y = recent_games['margin'].values
    A = np.vstack([X, np.ones(len(X))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    # Momentum (average margin)
    momentum = recent_games['margin'].mean()

    # Volatility (std dev of margins)
    volatility = recent_games['margin'].std()

    return {
        'trend': slope,
        'momentum': momentum,
        'volatility': volatility,
        'games_analyzed': n_games
    }

def detect_hot_cold_streaks(games: pd.DataFrame, team_id: int, threshold: float = 1.5) -> dict:
    """
    Detect hot (overperforming) and cold (underperforming) streaks using z-scores.

    Parameters:
    games (pd.DataFrame): DataFrame containing game results with columns:
        - 'team_id', 'points', 'opponent_points', 'result'
    team_id (int): ID of the team to analyze
    threshold (float): Z-score threshold to define hot/cold

    Returns:
    dict: Dictionary containing hot/cold indicators
    """
    team_games = games[games['team_id'] == team_id].copy()
    team_games['margin'] = team_games['points'] - team_games['opponent_points']

    if len(team_games) < 5:
        return {'hot_streak': False, 'cold_streak': False, 'z_score': 0}

    # Calculate z-scores for margins
    team_games['z_score'] = zscore(team_games['margin'])

    # Check recent games for hot/cold streaks
    recent_games = team_games.tail(5)
    recent_z_scores = recent_games['z_score']

    hot_streak = (recent_z_scores > threshold).all()
    cold_streak = (recent_z_scores < -threshold).all()

    current_z_score = recent_games['z_score'].iloc[-1]

    return {
        'hot_streak': hot_streak,
        'cold_streak': cold_streak,
        'current_z_score': current_z_score,
        'recent_z_scores': recent_z_scores.tolist()
    }

def compute_momentum_features(games: pd.DataFrame, team_id: int) -> dict:
    """
    Compute comprehensive momentum features for a team.

    Parameters:
    games (pd.DataFrame): DataFrame containing game results
    team_id (int): ID of the team to analyze

    Returns:
    dict: Dictionary containing all momentum features
    """
    features = {}

    # Win streak weighted by opponent strength
    features['weighted_win_streak'] = compute_win_streak_weighted_by_opponent_strength(games, team_id)

    # Recent margin trends
    trend_metrics = compute_recent_margin_trends(games, team_id, n_games=5)
    features.update({
        'margin_trend': trend_metrics['trend'],
        'avg_margin': trend_metrics['momentum'],
        'margin_volatility': trend_metrics['volatility']
    })

    # Hot/cold detection
    streak_metrics = detect_hot_cold_streaks(games, team_id)
    features.update({
        'is_hot_streak': streak_metrics['hot_streak'],
        'is_cold_streak': streak_metrics['cold_streak'],
        'current_z_score': streak_metrics['current_z_score']
    })

    return features

# Example usage:
# games = pd.DataFrame({
#     'team_id': [1,1,1,1,1],
#     'opponent_id': [2,3,4,5,6],
#     'points': [100,110,120,95,105],
#     'opponent_points': [90,105,115,100,95],
#     'opponent_strength': [0.8,1.2,1.1,0.9,1.3],
#     'result': [1,1,0,1,1]
# })
# momentum = compute_momentum_features(games, team_id=1)
