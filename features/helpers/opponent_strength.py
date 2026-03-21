import numpy as np
import pandas as pd
from scipy.stats import zscore

def compute_opponent_strength(df: pd.DataFrame) -> pd.Series:
    """
    Calculate opponent strength based on recent performance and schedule difficulty.

    Parameters:
    df (pd.DataFrame): Game data with columns ['team_id', 'opponent_id', 'points', 'opponent_points', 'game_date']

    Returns:
    pd.Series: Opponent strength score (0.0 to 1.0) for each game
    """
    # Calculate team performance metrics
    team_stats = df.groupby('team_id').agg({
        'points': ['mean', 'std'],
        'opponent_points': ['mean', 'std']
    }).reset_index()
    team_stats.columns = ['team_id', 'pts_mean', 'pts_std', 'opp_pts_mean', 'opp_pts_std']

    # Calculate opponent strength as win rate against similar teams
    df['result'] = (df['points'] > df['opponent_points']).astype(int)

    # Rolling win rate for each team
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')

    win_rates = df.groupby('team_id')['result'].rolling(window=10, min_periods=1).mean().reset_index()
    win_rates.columns = ['team_id', 'game_idx', 'rolling_win_rate']

    df = df.merge(win_rates, on=['team_id', 'game_idx'], how='left')

    # Strength of schedule adjustment
    df = df.merge(team_stats[['team_id', 'pts_mean']], left_on='opponent_id', right_on='team_id', how='left')
    df = df.rename(columns={'pts_mean': 'opp_team_strength'})

    # Final opponent strength score
    opponent_strength = df['rolling_win_rate'] * 0.7 + df['opp_team_strength'] * 0.3
    opponent_strength = opponent_strength.fillna(0.5)  # Default to neutral strength

    return opponent_strength

def compute_strength_of_schedule(df: pd.DataFrame) -> pd.Series:
    """
    Calculate strength of schedule for each team.

    Parameters:
    df (pd.DataFrame): Game data with opponent strength already computed

    Returns:
    pd.Series: Strength of schedule score for each game
    """
    # Calculate average opponent strength faced by each team
    sos = df.groupby('team_id')['opponent_strength'].mean().reset_index()
    sos.columns = ['team_id', 'strength_of_schedule']

    # Merge back to original data
    df = df.merge(sos, on='team_id', how='left')

    return df['strength_of_schedule']

def compute_matchup_impact(df: pd.DataFrame) -> pd.Series:
    """
    Calculate matchup impact based on team vs opponent strength differential.

    Parameters:
    df (pd.DataFrame): Game data with team and opponent strength columns

    Returns:
    pd.Series: Matchup impact score for each game
    """
    # Calculate strength differential
    df['strength_diff'] = df['team_strength'] - df['opponent_strength']

    # Normalize to get impact score
    impact = 1 / (1 + np.exp(-df['strength_diff'] / 0.5))

    return impact

# Example usage:
# df = pd.DataFrame({
#     'team_id': [1,1,2,2],
#     'opponent_id': [2,2,1,1],
#     'points': [100, 110, 95, 105],
#     'opponent_points': [90, 100, 105, 95],
#     'game_date': ['2024-01-01', '2024-01-05', '2024-01-02', '2024-01-06']
# })
#
# df['opponent_strength'] = compute_opponent_strength(df)
# df['strength_of_schedule'] = compute_strength_of_schedule(df)
# df['matchup_impact'] = compute_matchup_impact(df)
