import numpy as np

def compute_pace_adjusted(pts, pace, league_pace=100.0):
    """Convert raw points to pace-adjusted metric"""
    return (pts / pace) * league_pace if pace > 0 else pts

def compute_pace_adjusted_stats(stats, pace, league_pace=100.0):
    """Convert multiple stats to pace-adjusted metrics"""
    return {
        'pace_adj_pts': compute_pace_adjusted(stats.get('pts', 0), pace, league_pace),
        'pace_adj_tov': compute_pace_adjusted(stats.get('tov', 0), pace, league_pace),
        'pace_adj_ast': compute_pace_adjusted(stats.get('ast', 0), pace, league_pace),
        'pace_adj_reb': compute_pace_adjusted(stats.get('reb', 0), pace, league_pace),
        'pace_adj_fga': compute_pace_adjusted(stats.get('fga', 0), pace, league_pace),
        'pace_adj_trb': compute_pace_adjusted(stats.get('trb', 0), pace, league_pace),
    }

def rest_weighted_performance(stats, rest_days, decay=0.85):
    """Weight recent performance by rest days with exponential decay"""
    if rest_days <= 0:
        return stats
    weight = decay ** max(0, rest_days - 1)
    return {k: v * weight for k, v in stats.items()}
