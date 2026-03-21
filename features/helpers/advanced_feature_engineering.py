import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from typing import Dict, List

def create_interaction_features(df: pd.DataFrame, feature_columns: List[str], degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial interaction features to capture non-linear relationships.
    
    Parameters:
    df (pd.DataFrame): Input data
    feature_columns (List[str]): List of feature column names to transform
    degree (int): Degree of polynomial features (default: 2)
    
    Returns:
    pd.DataFrame: DataFrame with original features plus interaction features
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    interaction_matrix = poly.fit_transform(df[feature_columns])
    interaction_df = pd.DataFrame(interaction_matrix, columns=poly.get_feature_names_out(feature_columns))
    
    return pd.concat([df, interaction_df], axis=1)

def compute_opponent_adjusted_metrics(df: pd.DataFrame, team_id_col: str, opponent_id_col: str, metric_cols: List[str]) -> pd.DataFrame:
    """
    Compute opponent-adjusted metrics to account for strength of competition.
    
    Parameters:
    df (pd.DataFrame): Input data with team and opponent IDs
    team_id_col (str): Column name for team ID
    opponent_id_col (str): Column name for opponent ID
    metric_cols (List[str]): List of metric columns to adjust
    
    Returns:
    pd.DataFrame: DataFrame with opponent-adjusted metrics
    """
    # Calculate opponent averages
    opponent_stats = df.groupby(opponent_id_col)[metric_cols].mean().reset_index()
    opponent_stats.columns = [opponent_id_col] + [f'opponent_avg_{col}' for col in metric_cols]
    
    # Merge back to original data
    df = df.merge(opponent_stats, on=opponent_id_col, how='left')
    
    # Compute opponent-adjusted metrics
    for col in metric_cols:
        df[f'opponent_adj_{col}'] = df[col] - df[f'opponent_avg_{col}']
    
    return df

def generate_rolling_statistics(df: pd.DataFrame, group_col: str, value_col: str, window: int = 5) -> pd.DataFrame:
    """
    Generate rolling statistics for time-series analysis.
    
    Parameters:
    df (pd.DataFrame): Input data with datetime index
    group_col (str): Column to group by (e.g., team_id)
    value_col (str): Column to calculate rolling statistics for
    window (int): Window size for rolling calculations
    
    Returns:
    pd.DataFrame: DataFrame with rolling statistics
    """
    df = df.sort_index()
    
    rolling_stats = df.groupby(group_col)[value_col].rolling(window=window).agg([
        ('rolling_mean', 'mean'),
        ('rolling_std', 'std'),
        ('rolling_min', 'min'),
        ('rolling_max', 'max'),
        ('rolling_zscore', lambda x: (x - x.mean()) / x.std())
    ]).reset_index()
    
    rolling_stats.columns = [group_col, value_col, 
                           f'{value_col}_roll_mean',
                           f'{value_col}_roll_std',
                           f'{value_col}_roll_min',
                           f'{value_col}_roll_max',
                           f'{value_col}_roll_zscore']
    
    return rolling_stats

def create_team_strength_interactions(df: pd.DataFrame, team_strength_col: str, opponent_strength_col: str) -> pd.DataFrame:
    """
    Create interaction features between team and opponent strength.
    
    Parameters:
    df (pd.DataFrame): Input data
    team_strength_col (str): Column name for team strength
    opponent_strength_col (str): Column name for opponent strength
    
    Returns:
    pd.DataFrame: DataFrame with strength interaction features
    """
    df = df.copy()
    
    # Strength difference
    df['strength_diff'] = df[team_strength_col] - df[opponent_strength_col]
    
    # Strength ratio
    df['strength_ratio'] = df[team_strength_col] / df['opponent_strength_col']
    
    # Interaction term
    df['strength_interaction'] = df[team_strength_col] * df[opponent_strength_col]
    
    # Strength dominance indicator
    df['strength_dominant'] = (df[team_strength_col] > df['opponent_strength_col']).astype(int)
    
    return df

def compute_advanced_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute advanced performance metrics for model training.
    
    Parameters:
    df (pd.DataFrame): Game data with basic statistics
    
    Returns:
    pd.DataFrame: DataFrame with advanced performance metrics
    """
    df = df.copy()
    
    # Efficiency metrics
    df['offensive_efficiency'] = df['points'] / (df['pace'] / 100)
    df['defensive_efficiency'] = df['opponent_points'] / (df['pace'] / 100)
    df['net_efficiency'] = df['offensive_efficiency'] - df['defensive_efficiency']
    
    # True shooting percentage
    df['true_shooting'] = df['points'] / (2 * (df['fga'] + 0.44 * df['fta']))
    
    # Rebound percentage
    df['total_rebounds'] = df['orb'] + df['drb']
    df['rebounding_percentage'] = df['total_rebounds'] / (df['total_rebounds'] + df['opponent_total_rebounds'])
    
    # Turnover ratio
    df['turnover_ratio'] = df['tov'] / (df['fga'] + 0.44 * df['fta'] + df['tov'])
    
    return df

def create_advanced_feature_engineering_pipeline(df: pd.DataFrame, team_id_col: str, opponent_id_col: str, 
                                                 metric_cols: List[str], rest_col: str, travel_col: str) -> pd.DataFrame:
    """
    Complete feature engineering pipeline combining multiple advanced techniques.
    
    Parameters:
    df (pd.DataFrame): Input data
    team_id_col (str): Column for team ID
    opponent_id_col (str): Column for opponent ID
    metric_cols (List[str]): List of metric columns to transform
    rest_col (str): Column for rest days
    travel_col (str): Column for travel distance
    
    Returns:
    pd.DataFrame: DataFrame with all advanced features
    """
    # Create interaction features
    df = create_interaction_features(df, metric_cols, degree=2)
    
    # Opponent-adjusted metrics
    df = compute_opponent_adjusted_metrics(df, team_id_col, opponent_id_col, metric_cols)
    
    # Rolling statistics
    rolling_stats = generate_rolling_statistics(df, team_id_col, 'points', window=5)
    df = df.merge(rolling_stats, on=[team_id_col, 'date'], how='left')
    
    # Team strength interactions
    df = create_team_strength_interactions(df, 'team_strength', 'opponent_strength')
    
    # Advanced performance metrics
    df = compute_advanced_performance_metrics(df)
    
    return df
