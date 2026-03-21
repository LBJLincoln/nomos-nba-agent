import numpy as np
from datetime import datetime, timedelta

def calculate_momentum(team: str, game_history: list, window: int = 5) -> float:
    """Calculate momentum score based on recent game performance (weighted average)"""
    if not game_history:
        return 0.0

    team_games = [g for g in reversed(game_history) if g['home_team'] == team or g['away_team'] == team]
    
    if len(team_games) < window:
        window = len(team_games)
        if window == 0:
            return 0.0

    # Use exponential weighting (more recent games weighted higher)
    weights = np.exp(np.linspace(0, 2, window))
    scores = []

    for i, game in enumerate(team_games[:window]):
        is_home = game['home_team'] == team
        team_score = game['home_score' if is_home else 'away_score']
        opp_score = game['away_score' if is_home else 'home_score']
        margin = team_score - opp_score
        scores.append(margin)

    weighted_score = np.average(scores, weights=weights)
    return weighted_score

def calculate_win_rate(team: str, game_history: list, window: int = 5) -> float:
    """Calculate recent win rate"""
    if not game_history:
        return 0.5

    team_games = [g for g in reversed(game_history) if g['home_team'] == team or g['away_team'] == team]
    
    if len(team_games) < window:
        window = len(team_games)
        if window == 0:
            return 0.5

    wins = 0
    for game in team_games[:window]:
        is_home = game['home_team'] == team
        team_score = game['home_score' if is_home else 'away_score']
        opp_score = game['away_score' if is_home else 'home_score']
        if team_score > opp_score:
            wins += 1

    return wins / window
