import numpy as np

def compute_pace_adjusted_efficiency(pts, pace, league_avg_pace=100.0):
    """
    Adjust offensive/defensive efficiency based on team pace.
    League average pace normalized to 100 possessions.
    """
    if pace <= 0:
        return pts
    
    # Normalize to league average pace
    possessions_adj = (pace / league_avg_pace)
    return pts / possessions_adj

def team_pace_adjusted_stats(games, team_abbr, pace_col='pace'):
    """
    Calculate pace-adjusted offensive and defensive ratings.
    """
    off_rtg = []
    def_rtg = []
    
    for game in games:
        if game['home_team'] == team_abbr:
            team_pts = game['home_points']
            opp_pts = game['away_points']
            team_pace = game[pace_col]
        elif game['away_team'] == team_abbr:
            team_pts = game['away_points']
            opp_pts = game['home_points']
            team_pace = game[pace_col]
        else:
            continue
        
        # Adjust for pace
        off_rtg.append(compute_pace_adjusted_efficiency(team_pts, team_pace))
        def_rtg.append(compute_pace_adjusted_efficiency(opp_pts, team_pace))
    
    return {
        'off_rtg': np.mean(off_rtg) if off_rtg else 0.0,
        'def_rtg': np.mean(def_rtg) if def_rtg else 0.0
    }
