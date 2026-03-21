import numpy as np

def compute_pace_adjusted(pts, pace, league_pace=100.0):
    """Convert points per game to pace-adjusted metric"""
    return (pts / pace) * league_pace if pace > 0 else pts

def compute_pace_adjusted_stats(team_stats, league_pace=100.0):
    """Convert raw team stats to pace-adjusted metrics"""
    pace = team_stats.get('pace', league_pace)
    return {
        'pace_adj_pts': compute_pace_adjusted(team_stats.get('pts', 0), pace, league_pace),
        'pace_adj_opp_pts': compute_pace_adjusted(team_stats.get('opp_pts', 0), pace, league_pace),
        'pace_adj_off_rtg': (team_stats.get('pts', 0) / pace) * 100 if pace > 0 else team_stats.get('pts', 0),
        'pace_adj_def_rtg': (team_stats.get('opp_pts', 0) / pace) * 100 if pace > 0 else team_stats.get('opp_pts', 0),
    }

def rest_weighted_performance(team_stats, rest_days, decay=0.85):
    """Weight recent performance by rest days (more rest = better performance)"""
    if rest_days <= 0:
        return team_stats
    weight = min(1.0, rest_days * 0.1)  # Cap at 1.0 for 10+ rest days
    decay_factor = decay ** (rest_days - 1)
    return {
        'rest_adj_pts': team_stats.get('pts', 0) * (1 + weight * decay_factor),
        'rest_adj_opp_pts': team_stats.get('opp_pts', 0) * (1 - weight * decay_factor * 0.5),
        'rest_adj_off_rtg': team_stats.get('off_rtg', 100) * (1 + weight * decay_factor * 0.1),
        'rest_adj_def_rtg': team_stats.get('def_rtg', 100) * (1 - weight * decay_factor * 0.05),
    }
