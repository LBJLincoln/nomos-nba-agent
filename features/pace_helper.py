import numpy as np

def compute_pace_adjusted(pts, pace, league_pace=100.0):
    return (pts / pace) * league_pace if pace > 0 else pts

def offensive_efficiency(pts, pace):
    return (pts / pace) * 100 if pace > 0 else pts

def defensive_efficiency(opp_pts, pace):
    return (opp_pts / pace) * 100 if pace > 0 else opp_pts

def net_rating(offensive, defensive):
    return offensive - defensive
