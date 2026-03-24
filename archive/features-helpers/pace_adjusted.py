import numpy as np

def compute_pace_adjusted(pts, pace, league_pace=100.0):
    """
    Compute pace-adjusted points per 100 possessions.

    Parameters:
    pts (float): Points scored
    pace (float): Team possessions per game
    league_pace (float): League average pace (default: 100)

    Returns:
    float: Pace-adjusted points per 100 possessions
    """
    return (pts / pace) * league_pace if pace > 0 else pts

def compute_pace_adjusted_ppg(team_points, opponent_points, team_pace, opponent_pace):
    """
    Compute pace-adjusted points per game for both teams.

    Parameters:
    team_points (float): Team's average points per game
    opponent_points (float): Opponent's average points per game
    team_pace (float): Team's possessions per game
    opponent_pace (float): Opponent's possessions per game

    Returns:
    tuple: (team_pace_adjusted_ppg, opponent_pace_adjusted_ppg)
    """
    league_pace = 100.0
    team_adj = (team_points / team_pace) * league_pace
    opponent_adj = (opponent_points / opponent_pace) * league_pace
    return team_adj, opponent_adj

def calculate_possession_stats(fga, fgma, fta, orb, to):
    """
    Calculate possessions and offensive rating.

    Parameters:
    fga (int): Field goals attempted
    fgma (int): Field goals made
    fta (int): Free throws attempted
    orb (int): Offensive rebounds
    to (int): Turnovers

    Returns:
    tuple: (possessions, offensive_rating)
    """
    possessions = fga - fgma + fta * 0.44 + to - orb
    offensive_rating = (100 * (fgma + 0.5 * fta * (1 - (orb / (fga - fgma + fta * 0.44 + to))))) / possessions if possessions > 0 else 0
    return possessions, offensive_rating
