import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List

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

def compute_back_to_back_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate back-to-back fatigue indicators and performance impact.
    
    Parameters:
    df (pd.DataFrame): Game data with columns:
        - 'team_id', 'game_date', 'points', 'opponent_points'
        - 'rest_days' (pre-computed rest days)
    
    Returns:
    pd.DataFrame: Original dataframe with back-to-back fatigue features
    """
    df = df.copy()
    
    # Basic back-to-back indicators
    df['is_back_to_back'] = (df['rest_days'] < 1).astype(int)
    df['rest_quality_score'] = np.where(df['rest_days'] < 1, 0.5,
                                      np.where(df['rest_days'] < 2, 0.7, 0.9))
    
    # Travel fatigue indicators
    df['travel_distance'] = df['travel_distance'].fillna(0)
    df['long_travel'] = (df['travel_distance'] > 1000).astype(int)
    df['medium_travel'] = ((df['travel_distance'] > 500) & (df['travel_distance'] <= 1000)).astype(int)
    
    # Combined fatigue score
    df['fatigue_score'] = 1.0
    
    # Apply penalties
    if 'is_back_to_back' in df.columns:
        df.loc[df['is_back_to_back'] == 1, 'fatigue_score'] *= 0.7
    if 'long_travel' in df.columns:
        df.loc[df['long_travel'] == 1, 'fatigue_score'] *= 0.8
    if 'medium_travel' in df.columns:
        df.loc[df['medium_travel'] == 1, 'fatigue_score'] *= 0.9
    
    # Rest-based performance adjustment
    df['rest_adjusted_points'] = df['points'] * df['rest_quality_score']
    df['rest_adjusted_defense'] = df['opponent_points'] / df['rest_quality_score']
    
    # Back-to-back win probability adjustment
    df['back_to_back_win_adj'] = np.where(df['is_back_to_back'] == 1,
                                          0.85, 1.0)
    
    return df

def compute_home_court_advantage_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive home court advantage metrics.
    
    Parameters:
    df (pd.DataFrame): Game data with columns:
        - 'team_id', 'location', 'points', 'opponent_points'
        - 'team_home_record', 'opponent_home_record' (pre-computed home records)
    
    Returns:
    pd.DataFrame: Original dataframe with home court advantage features
    """
    df = df.copy()
    
    # Basic home/away indicators
    df['is_home'] = (df['location'] == 'home').astype(int)
    df['is_away'] = (df['location'] == 'away').astype(int)
    df['is_neutral'] = ((df['location'] != 'home') & (df['location'] != 'away')).astype(int)
    
    # Home court performance metrics
    df['home_court_impact'] = np.where(df['is_home'] == 1,
                                      df['points'] - df['opponent_points'],
                                      np.where(df['is_away'] == 1,
                                              df['opponent_points'] - df['points'],
                                              0))
    
    # Home court win probability adjustment
    df['home_advantage'] = np.where(df['is_home'] == 1,
                                    0.05, np.where(df['is_away'] == 1,
                                                   -0.05, 0))
    
    # Crowd impact factor (proxy for fan attendance)
    df['crowd_impact'] = np.where(df['is_home'] == 1,
                                  1.02, np.where(df['is_away'] == 1,
                                                 0.98, 1.0))
    
    # Travel distance interaction with home court
    df['home_court_travel_interaction'] = df['crowd_impact'] * (1 - df['travel_distance'] / 2000)
    
    # Historical home court strength
    df['team_home_strength'] = df['team_home_record'] / (df['team_home_record'] + df['team_away_record'] + 1)
    df['opponent_home_strength'] = df['opponent_home_record'] / (df['opponent_home_record'] + df['opponent_away_record'] + 1)
    
    # Home court differential
    df['home_court_differential'] = df['team_home_strength'] - df['opponent_home_strength']
    
    return df

def compute_advanced_pace_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate advanced pace-adjusted statistics for performance normalization.
    
    Parameters:
    df (pd.DataFrame): Game data with pace and scoring columns
    
    Returns:
    pd.DataFrame: Original dataframe with pace-adjusted features
    """
    df = df.copy()
    
    # League average pace
    league_pace = df['pace'].mean()
    
    # Basic pace adjustment
    df['pace_adjusted_points'] = (df['points'] / df['pace']) * league_pace
    df['pace_adjusted_defense'] = (df['opponent_points'] / df['pace']) * league_pace
    
    # Efficiency metrics
    df['offensive_efficiency'] = df['points'] / (df['pace'] / 100)
    df['defensive_efficiency'] = df['opponent_points'] / (df['pace'] / 100)
    df['net_efficiency'] = df['offensive_efficiency'] - df['defensive_efficiency']
    
    # True shooting percentage
    df['true_shooting'] = df['points'] / (2 * (df['fga'] + 0.44 * df['fta']))
    
    # Pace interaction with strength
    df['pace_strength_interaction'] = df['pace'] * df['team_strength']
    df['pace_opponent_interaction'] = df['pace'] * df['opponent_strength']
    
    # Pace-based win probability adjustment
    df['pace_win_adj'] = np.where(df['pace'] > league_pace,
                                  0.02, np.where(df['pace'] < league_pace,
                                                  -0.02, 0))
    
    return df

# Example usage:
# df = pd.DataFrame({
#     'team_id': [1,1,2,2],
#     'opponent_id': [2,2,1,1],
#     'points': [100, 110, 95, 105],
#     'opponent_points': [90, 100, 105, 95],
#     'team_strength': [1.1, 1.1, 0.9, 0.9],
#     'opponent_strength': [0.9, 0.9, 1.1, 1.1],
#     'game_time': [48, 48, 48, 48],
#     'quarter': [4, 4, 4, 4],
#     'rest_days': [1, 0, 2, 1],
#     'location': ['home', 'away', 'away', 'home'],
#     'pace': [100, 105, 95, 100],
#     'fga': [80, 85, 75, 80],
#     'fta': [20, 25, 15, 20]
# })
# 
# df = compute_opponent_strength_decomposition(df)
# df = compute_clutch_time_statistics(df)
# df = compute_back_to_back_fatigue_features(df)
# df = compute_home_court_advantage_metrics(df)
# df = compute_advanced_pace_adjustments(df)
