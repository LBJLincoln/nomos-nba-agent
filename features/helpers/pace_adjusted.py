import numpy as np
from typing import Optional

def compute_pace_adjusted_stats(
    pts: float,
    pace: float,
    league_pace: float = 100.0,
    minutes_played: Optional[float] = None,
    league_minutes: float = 48.0
) -> float:
    """
    Compute pace-adjusted statistics for NBA performance metrics.

    Parameters:
    pts (float): Raw statistic (e.g., points, rebounds, etc.)
    pace (float): Team's actual pace (possessions per 48 minutes)
    league_pace (float): League average pace (default: 100.0)
    minutes_played (Optional[float]): Minutes played in the game (default: None)
    league_minutes (float): Standard game length in minutes (default: 48.0)

    Returns:
    float: Pace-adjusted statistic
    """
    if pace <= 0:
        return pts

    # Adjust for pace (possessions-based normalization)
    pace_adjusted = (pts / pace) * league_pace

    # If minutes played is provided, also adjust for playing time
    if minutes_played is not None and minutes_played > 0:
        per_48 = (pts / minutes_played) * league_minutes
        # Combine both adjustments with weighted average
        return (pace_adjusted + per_48) / 2

    return pace_adjusted

def compute_team_pace_adjusted_points(
    team_points: float,
    opponent_points: float,
    team_pace: float,
    opponent_pace: float,
    league_pace: float = 100.0
) -> tuple:
    """
    Compute pace-adjusted points for both teams in a game.

    Parameters:
    team_points (float): Points scored by the team
    opponent_points (float): Points scored by the opponent
    team_pace (float): Team's pace
    opponent_pace (float): Opponent's pace
    league_pace (float): League average pace

    Returns:
    tuple: (team_adjusted_points, opponent_adjusted_points)
    """
    team_adj = compute_pace_adjusted_stats(team_points, team_pace, league_pace)
    opp_adj = compute_pace_adjusted_stats(opponent_points, opponent_pace, league_pace)
    return team_adj, opp_adj

def compute_effective_field_goal_percentage(
    fgm: float,
    fgm3: float,
    fga: float
) -> float:
    """
    Compute Effective Field Goal Percentage (eFG%).

    Parameters:
    fgm (float): 2-point field goals made
    fgm3 (float): 3-point field goals made
    fga (float): Field goal attempts

    Returns:
    float: eFG% (0.0-1.0)
    """
    if fga == 0:
        return 0.0
    return (fgm + 1.25 * fgm3) / fga

def compute_player_impact_estimate(
    pts: float,
    reb: float,
    ast: float,
    stl: float,
    blk: float,
    fgm: float,
    fga: float,
    fgm3: float,
    fga3: float,
    ftm: float,
    fta: float,
    turnovers: float,
    pf: float,
    pace: float,
    league_pace: float = 100.0
) -> float:
    """
    Compute Player Impact Estimate (PIE) - a pace-adjusted performance metric.

    Parameters:
    pts (float): Points
    reb (float): Rebounds
    ast (float): Assists
    stl (float): Steals
    blk (float): Blocks
    fgm (float): Field goals made
    fga (float): Field goal attempts
    fgm3 (float): 3-point field goals made
    fga3 (float): 3-point field goal attempts
    ftm (float): Free throws made
    fta (float): Free throw attempts
    turnovers (float): Turnovers
    pf (float): Personal fouls
    pace (float): Team's pace
    league_pace (float): League average pace

    Returns:
    float: Player Impact Estimate (higher is better)
    """
    # Basic stats
    scoring = pts
    shooting_efficiency = (compute_effective_field_goal_percentage(fgm, fgm3, fga) * 100) if fga > 0 else 0
    playmaking = ast
    rebounding = reb
    defense = stl + blk
    free_throw_rate = (ftm / (fga + fta + 1e-5)) * 100

    # Negative factors
    negative_plays = turnovers + (fga - fgm) + (fta - ftm) + pf

    # Pace adjustment
    pace_factor = (pace / league_pace) if league_pace > 0 else 1.0

    # Combine with weights
    pie = (
        (scoring * 0.2) +
        (playmaking * 0.25) +
        (rebounding * 0.15) +
        (defense * 0.2) +
        (shooting_efficiency * 0.1) +
        (free_throw_rate * 0.05)
    ) - (negative_plays * 0.1)

    return pie * pace_factor

def compute_team_efficiency_rating(
    team_points: float,
    opponent_points: float,
    team_possessions: float,
    opponent_possessions: float
) -> tuple:
    """
    Compute offensive and defensive efficiency ratings.

    Parameters:
    team_points (float): Points scored by team
    opponent_points (float): Points allowed by team
    team_possessions (float): Team's estimated possessions
    opponent_possessions (float): Opponent's estimated possessions

    Returns:
    tuple: (offensive_efficiency, defensive_efficiency)
    """
    offensive_efficiency = (team_points / team_possessions) * 100 if team_possessions > 0 else 0
    defensive_efficiency = (opponent_points / opponent_possessions) * 100 if opponent_possessions > 0 else 0
    return offensive_efficiency, defensive_efficiency

def compute_possession_estimates(
    fga: float,
    oreb: float,
    tov: float,
    fta: float,
    opponent_fga: float,
    opponent_oreb: float,
    opponent_tov: float,
    opponent_fta: float
) -> tuple:
    """
    Estimate possessions for both teams using the standard NBA formula.

    Parameters:
    fga (float): Field goal attempts
    oreb (float): Offensive rebounds
    tov (float): Turnovers
    fta (float): Free throw attempts
    opponent_fga (float): Opponent's field goal attempts
    opponent_oreb (float): Opponent's offensive rebounds
    opponent_tov (float): Opponent's turnovers
    opponent_fta (float): Opponent's free throw attempts

    Returns:
    tuple: (team_possessions, opponent_possessions)
    """
    team_possessions = fga - oreb + tov + (0.4 * fta)
    opponent_possessions = opponent_fga - opponent_oreb + opponent_tov + (0.4 * opponent_fta)
    return team_possessions, opponent_possessions

def compute_advanced_stats(
    pts: float,
    fgm: float,
    fga: float,
    fgm3: float,
    fga3: float,
    ftm: float,
    fta: float,
    oreb: float,
    dreb: float,
    ast: float,
    stl: float,
    blk: float,
    tov: float,
    pf: float,
    pace: float,
    opponent_pts: float,
    opponent_fgm: float,
    opponent_fga: float,
    opponent_fgm3: float,
    opponent_fga3: float,
    opponent_ftm: float,
    opponent_fta: float,
    opponent_oreb: float,
    opponent_dreb: float,
    opponent_ast: float,
    opponent_stl: float,
    opponent_blk: float,
    opponent_tov: float,
    opponent_pf: float,
    opponent_pace: float
) -> dict:
    """
    Compute comprehensive advanced statistics for a game.

    Parameters:
    All basic box score stats for both teams and their paces

    Returns:
    dict: Dictionary of advanced statistics
    """
    stats = {}

    # Pace-adjusted points
    team_adj_pts, opp_adj_pts = compute_team_pace_adjusted_points(
        pts, opponent_pts, pace, opponent_pace
    )
    stats['pace_adjusted_points'] = team_adj_pts
    stats['opponent_pace_adjusted_points'] = opp_adj_pts

    # Efficiency ratings
    team_fga = fga
    team_fta = fta
    team_tov = tov
    opp_fga = opponent_fga
    opp_fta = opponent_fta
    opp_tov = opponent_tov

    team_possessions, opp_possessions = compute_possession_estimates(
        team_fga, oreb, team_tov, team_fta,
        opp_fga, opponent_oreb, opp_tov, opp_fta
    )

    off_eff, def_eff = compute_team_efficiency_rating(
        pts, opponent_pts, team_possessions, opp_possessions
    )
    stats['offensive_efficiency'] = off_eff
    stats['defensive_efficiency'] = def_eff

    # True Shooting Percentage
    fga_plus_fta = fga + (0.44 * fta)
    tsa = fga_plus_fta + (0.9 * fta) if (fga_plus_fta + (0.9 * fta)) > 0 else 1
    stats['true_shooting_percentage'] = pts / (2 * tsa)

    # Rebound percentages
    total_reb = oreb + dreb
    opp_total_reb = opponent_oreb + opponent_dreb
    total_missed_shots = fga - fgm + (0.44 * fta) + tov
    opp_total_missed_shots = opponent_fga - opponent_fgm + (0.44 * opponent_fta) + opponent_tov

    stats['offensive_rebound_percentage'] = (oreb / (oreb + opp_total_reb)) * 100 if (oreb + opp_total_reb) > 0 else 0
    stats['defensive_rebound_percentage'] = (dreb / (dreb + opp_total_missed_shots)) * 100 if (dreb + opp_total_missed_shots) > 0 else 0

    # Assist percentage
    stats['assist_percentage'] = (ast / (ast + 0.4 * fga + 0.44 * fta + tov)) * 100 if (ast + 0.4 * fga + 0.44 * fta + tov) > 0 else 0

    # Turnover percentage
    stats['turnover_percentage'] = (tov / (fga + 0.44 * fta + tov)) * 100 if (fga + 0.44 * fta + tov) > 0 else 0

    # Player Impact Estimate
    stats['player_impact_estimate'] = compute_player_impact_estimate(
        pts, total_reb, ast, stl, blk, fgm, fga, fgm3, fga3, ftm, fta,
        tov, pf, pace
    )

    return stats
