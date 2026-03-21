import numpy as np
from datetime import datetime, timedelta

def compute_rest_weighted_performance(team_stats, rest_days):
    """
    Weight recent performance by rest days - penalize short rest, reward long rest.
    Returns a multiplier between 0.8-1.2 based on rest quality.
    """
    if rest_days is None or rest_days < 0:
        return 1.0
    
    # Optimal rest range: 2-4 days
    if rest_days < 2:
        return 0.85  # Fatigue penalty
    elif rest_days <= 4:
        return 1.05  # Optimal performance
    elif rest_days <= 6:
        return 1.02  # Good rest
    else:
        return 0.95  # Rust penalty

def apply_rest_weighting(games, team_abbr, rest_col='rest_days'):
    """
    Calculate rest-weighted performance for a team across games.
    """
    weighted_perf = []
    for game in games:
        if game['home_team'] == team_abbr:
            rest = game.get(rest_col)
            pts = game['home_points']
        elif game['away_team'] == team_abbr:
            rest = game.get(rest_col)
            pts = game['away_points']
        else:
            continue
        
        weight = compute_rest_weighted_performance(None, rest)
        weighted_perf.append(pts * weight)
    
    return np.mean(weighted_perf) if weighted_perf else 0.0
