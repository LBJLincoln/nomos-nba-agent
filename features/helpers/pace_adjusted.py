import numpy as np
def compute_pace_adjusted(pts, pace, league_pace=100.0):
    return (pts / pace) * league_pace if pace > 0 else pts
