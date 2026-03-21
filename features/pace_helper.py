import numpy as np

def compute_pace_adjusted(pts, pace, league_pace=100.0):
    """Convert raw points to pace-adjusted points per 100 possessions."""
    return (pts / pace) * league_pace if pace > 0 else pts

def compute_pace_adjusted_stats(stats, pace, league_pace=100.0):
    """Convert multiple stats to pace-adjusted values."""
    return {
        'pts_pa': compute_pace_adjusted(stats.get('pts', 0), pace, league_pace),
        'reb_pa': compute_pace_adjusted(stats.get('reb', 0), pace, league_pace),
        'ast_pa': compute_pace_adjusted(stats.get('ast', 0), pace, league_pace),
        'stl_pa': compute_pace_adjusted(stats.get('stl', 0), pace, league_pace),
        'blk_pa': compute_pace_adjusted(stats.get('blk', 0), pace, league_pace),
        'tov_pa': compute_pace_adjusted(stats.get('tov', 0), pace, league_pace),
    }

def compute_rest_weighted_performance(stats, rest_days):
    """Weight recent performance by rest days (more rest = higher weight)."""
    if rest_days <= 0:
        return stats
    weight = min(1.0 + (rest_days / 5.0), 2.0)  # Cap at 2x weight
    return {k: v * weight for k, v in stats.items()}
