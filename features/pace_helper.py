import numpy as np

def compute_pace_adjusted(pts, pace, league_pace=100.0):
    """Convert raw points to pace-adjusted metric"""
    return (pts / pace) * league_pace if pace > 0 else pts

def compute_offensive_efficiency(pts, pace, possessions):
    """Points per 100 possessions"""
    return (pts / possessions) * 100 if possessions > 0 else 0

def compute_defensive_efficiency(opp_pts, pace, opp_possessions):
    """Opponent points per 100 possessions"""
    return (opp_pts / opp_possessions) * 100 if opp_possessions > 0 else 0

def compute_net_rating(off_rtg, def_rtg):
    """Net rating = offensive - defensive efficiency"""
    return off_rtg - def_rtg

def compute_rest_weighted_stat(stat, rest_days, decay=0.85):
    """Weight recent performance by rest days (more rest = more weight)"""
    return stat * (decay ** (max(0, 4 - rest_days)))
