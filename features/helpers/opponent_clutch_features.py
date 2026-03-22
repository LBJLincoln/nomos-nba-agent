import numpy as np
import pandas as pd
from datetime import datetime

def compute_opponent_strength_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose opponent strength into multiple components to capture nuanced matchups.
    
    Parameters:
    df (pd.DataFrame): Game data with columns:
        - 'team_id', 'opponent_id', 'points', 'opponent_points'
        - 'team_strength', 'opponent_strength' (pre-computed strength metrics)
    
    Returns:
    pd.DataFrame: Original dataframe with opponent strength decomposition features
    """
    df = df.copy()
    
    # Strength difference (absolute and relative)
    df['strength_diff'] = df['team_strength'] - df['opponent_strength']
    df['strength_diff_abs'] = np.abs(df['strength_diff'])
    df['strength_ratio'] = df['team_strength'] / df['opponent_strength']
    
    # Strength dominance indicators
    df['is_strength_favored'] = (df['team_strength'] > df['opponent_strength']).astype(int)
    df['strength_dominance'] = df['team_strength'] - df['opponent_strength']
    
    # Strength interaction terms
    df['strength_interaction'] = df['team_strength'] * df['opponent_strength']
    df['strength_product'] = df['team_strength'] * df['opponent_strength']
    
    # Strength category bins
    df['strength_category'] = pd.cut(df['strength_diff_abs'], 
                                     bins=[-1, 0.2, 0.5, 1.0, 2.0, np.inf],
                                     labels=['even', 'slight', 'moderate', 'strong', 'extreme'])
    
    # Strength-based win probability proxy
    df['strength_win_prob'] = 1 / (1 + np.exp(-df['strength_diff'] / 0.5))
    
    return df

def compute_clutch_time_statistics(df: pd.DataFrame, threshold_minutes: int = 5) -> pd.DataFrame:
    """
    Calculate clutch time statistics for late-game performance analysis.
    
    Parameters:
    df (pd.DataFrame): Game data with columns:
        - 'team_id', 'points', 'opponent_points', 'game_time' (in minutes)
        - 'quarter' (game quarter)
    
    Returns:
    pd.DataFrame: Original dataframe with clutch time features
    """
    df = df.copy()
    
    # Define clutch time (last 5 minutes of close games)
    df['clutch_time'] = ((df['quarter'] >= 4) & (df['game_time'] <= threshold_minutes)).astype(int)
    
    # Calculate clutch margin
    df['clutch_margin'] = df['points'] - df['opponent_points']
    df['clutch_close_game'] = (np.abs(df['clutch_margin']) <= 5).astype(int)
    
    # Clutch performance indicators
    df['clutch_opportunity'] = df['clutch_time'] & df['clutch_close_game']
    df['clutch_pressure'] = np.abs(df['clutch_margin']) * df['clutch_time']
    
    # Late-game scoring rates
    df['late_game_points'] = df['points'] * (df['quarter'] >= 4).astype(int)
    df['clutch_points_rate'] = df['late_game_points'] / df['game_time'].replace(0, 1)
    
    # Comeback indicators
    df['is_trailing'] = (df['clutch_margin'] < 0).astype(int)
    df['is_winning'] = (df['clutch_margin'] > 0).astype(int)
    
    # Clutch efficiency metrics
    df['clutch_efficiency'] = df['points'] / (df['points'] + df['opponent_points'] + 1)
    df['clutch_pressure_score'] = df['clutch_pressure'] * df['clutch_efficiency']
    
    return df
