import numpy as np
import pandas as pd
from typing import Dict, List

def compute_simple_baseline_features(games: pd.DataFrame, team_id: int) -> Dict[str, float]:
    """
    Compute simple baseline features for a team.
    
    Parameters:
    games (pd.DataFrame): Game data with columns ['team_id', 'opponent_id', 'points', 'opponent_points']
    team_id (int): Team ID to analyze
    
    Returns:
    Dict[str, float]: Baseline feature dictionary
    """
    team_games = games[games['team_id'] == team_id].copy()
    
    if len(team_games) == 0:
        return {
            'baseline_win_prob': 0.5,
            'avg_points': 100.0,
            'avg_opponent_points': 100.0,
            'home_advantage': 0.0
        }
    
    # Simple win probability based on recent performance
    recent_games = team_games.tail(5)
    win_rate = (recent_games['points'] > recent_games['opponent_points']).mean()
    
    # Average points and opponent points
    avg_points = recent_games['points'].mean()
    avg_opponent_points = recent_games['opponent_points'].mean()
    
    # Simple home advantage (if location data available)
    home_games = recent_games[recent_games['location'] == 'home']
    home_advantage = (home_games['points'] > home_games['opponent_points']).mean() - 0.5
    
    return {
        'baseline_win_prob': win_rate,
        'avg_points': avg_points,
        'avg_opponent_points': avg_opponent_points,
        'home_advantage': home_advantage
    }

def compute_opponent_strength_baseline(games: pd.DataFrame, team_id: int, opponent_id: int) -> float:
    """
    Compute opponent strength baseline using simple metrics.
    
    Parameters:
    games (pd.DataFrame): Game data
    team_id (int): Team ID
    opponent_id (int): Opponent ID
    
    Returns:
    float: Opponent strength score (0.0 to 1.0)
    """
    # Get opponent's recent performance against similar teams
    opponent_games = games[games['opponent_id'] == opponent_id].copy()
    
    if len(opponent_games) == 0:
        return 0.5  # Neutral strength
    
    # Simple strength metric: win rate against teams with similar strength
    opponent_win_rate = (opponent_games['points'] > opponent_games['opponent_points']).mean()
    
    # Adjust for strength of schedule
    opponent_strength = opponent_win_rate * 0.8 + 0.5 * 0.2
    
    return opponent_strength

def baseline_prediction(games: pd.DataFrame, team_id: int, opponent_id: int, is_home: bool) -> float:
    """
    Generate baseline prediction using simple heuristics.
    
    Parameters:
    games (pd.DataFrame): Game data
    team_id (int): Team ID
    opponent_id (int): Opponent ID
    is_home (bool): Whether team is home
    
    Returns:
    float: Predicted win probability (0.0 to 1.0)
    """
    # Get baseline features
    team_features = compute_simple_baseline_features(games, team_id)
    opponent_features = compute_simple_baseline_features(games, opponent_id)
    
    # Simple prediction: compare team strength
    team_strength = team_features['baseline_win_prob'] + team_features['home_advantage'] * (1 if is_home else 0)
    opponent_strength = opponent_features['baseline_win_prob'] + opponent_features['home_advantage'] * (0 if is_home else 1)
    
    # Adjust for opponent strength
    opponent_strength_baseline = compute_opponent_strength_baseline(games, team_id, opponent_id)
    opponent_strength *= (1 + (opponent_strength_baseline - 0.5) * 0.5)
    
    # Final prediction using logistic function
    logit = np.log(team_strength / (1 - team_strength)) - np.log(opponent_strength / (1 - opponent_strength))
    win_prob = 1 / (1 + np.exp(-logit))
    
    # Clamp to reasonable range
    win_prob = max(0.05, min(0.95, win_prob))
    
    return win_prob

def evaluate_baseline_performance(games: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate baseline model performance.
    
    Parameters:
    games (pd.DataFrame): Game data with actual results
    predictions (pd.DataFrame): Predicted probabilities
    
    Returns:
    Dict[str, float]: Performance metrics
    """
    results = {}
    
    if len(predictions) == 0:
        return {'brier_score': 1.0, 'accuracy': 0.0}
    
    # Calculate Brier score
    actual = (games['points'] > games['opponent_points']).astype(int)
    probs = predictions['probabilities'].values
    results['brier_score'] = np.mean((probs - actual) ** 2)
    
    # Calculate accuracy
    predicted_labels = (probs > 0.5).astype(int)
    results['accuracy'] = np.mean(predicted_labels == actual)
    
    # Log-loss
    from sklearn.metrics import log_loss
    results['log_loss'] = log_loss(actual, probs, eps=1e-15)
    
    return results

# Example usage:
# games = pd.DataFrame({
#     'team_id': [1,1,2,2],
#     'opponent_id': [2,2,1,1],
#     'points': [100, 110, 95, 105],
#     'opponent_points': [90, 100, 105, 95],
#     'location': ['home', 'away', 'away', 'home']
# })
# 
# # Generate baseline predictions
# predictions = []
# for _, game in games.iterrows():
#     prob = baseline_prediction(games, game['team_id'], game['opponent_id'], game['location'] == 'home')
#     predictions.append({
#         'team_id': game['team_id'],
#         'opponent_id': game['opponent_id'],
#         'probabilities': prob,
#         'predicted_label': 1 if prob > 0.5 else 0
#     })
# 
# # Evaluate performance
# pred_df = pd.DataFrame(predictions)
# metrics = evaluate_baseline_performance(games, pred_df)
# print(f"Baseline Brier Score: {metrics['brier_score']:.4f}")
# print(f"Baseline Accuracy: {metrics['accuracy']:.4f}")
