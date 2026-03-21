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
    df['strength_ratio'] = df[team_strength_col] / df[opponent_strength_col]
    
    # Interaction term
    df['strength_interaction'] = df[team_strength_col] * df[opponent_strength_col]
    
    # Strength dominance indicator
    df['strength_dominant'] = (df[team_strength_col] > df[opponent_strength_col]).astype(int)
    
    return df

def compute_advanced_momentum_features(df: pd.DataFrame, team_id_col: str, metric_cols: List[str], n_windows: List[int]) -> pd.DataFrame:
    """
    Compute advanced momentum features with multiple time windows.
    
    Parameters:
    df (pd.DataFrame): Input data with datetime index
    team_id_col (str): Column for team ID
    metric_cols (List[str]): List of metric columns to analyze
    n_windows (List[int]): List of window sizes to compute
    
    Returns:
    pd.DataFrame: DataFrame with advanced momentum features
    """
    df = df.sort_index()
    momentum_features = []
    
    for window in n_windows:
        window_features = df.groupby(team_id_col)[metric_cols].rolling(window=window).agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('skew', lambda x: x.skew()),
            ('kurt', lambda x: x.kurtosis()),
            ('trend', lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0)
        ]).reset_index()
        
        window_features.columns = [team_id_col, 'date'] + [f'{col}_win{window}_{agg}' for col in metric_cols for agg in ['mean', 'std', 'skew', 'kurt', 'trend']]
        
        momentum_features.append(window_features)
    
    # Merge all window features
    result = momentum_features[0]
    for wf in momentum_features[1:]:
        result = result.merge(wf, on=[team_id_col, 'date'], how='left')
    
    return result

def create_rest_travel_interaction_features(df: pd.DataFrame, rest_col: str, travel_col: str) -> pd.DataFrame:
    """
    Create interaction features between rest and travel factors.
    
    Parameters:
    df (pd.DataFrame): Input data
    rest_col (str): Column name for rest days
    travel_col (str): Column name for travel distance
    
    Returns:
    pd.DataFrame: DataFrame with rest-travel interaction features
    """
    df = df.copy()
    
    # Rest-travel interaction
    df['rest_travel_interaction'] = df[rest_col] * df[travel_col]
    
    # Rest categories
    df['rest_category'] = pd.cut(df[rest_col], bins=[-1, 0, 1, 2, 3, 4, np.inf], 
                                 labels=['no_rest', 'minimal', 'short', 'medium', 'long', 'extended'])
    
    # Travel categories
    df['travel_category'] = pd.cut(df[travel_col], bins=[0, 100, 500, 1000, 2000, np.inf],
                                   labels=['local', 'regional', 'national', 'cross_country', 'international'])
    
    # Rest-travel category interaction
    df['rest_travel_category'] = df['rest_category'].astype(str) + '_' + df['travel_category'].astype(str)
    
    return df

def generate_team_performance_profiles(df: pd.DataFrame, team_id_col: str, metric_cols: List[str], n_quantiles: int = 5) -> pd.DataFrame:
    """
    Generate team performance profiles based on quantile analysis.
    
    Parameters:
    df (pd.DataFrame): Input data
    team_id_col (str): Column for team ID
    metric_cols (List[str]): List of metric columns to profile
    n_quantiles (int): Number of quantiles to create
    
    Returns:
    pd.DataFrame: DataFrame with team performance profiles
    """
    df = df.copy()
    
    for col in metric_cols:
        # Calculate quantiles
        df[f'{col}_quantile'] = df.groupby(team_id_col)[col].transform(lambda x: pd.qcut(x, q=n_quantiles, labels=False, duplicates='drop'))
        
        # Performance profile indicators
        df[f'{col}_is_top_quantile'] = (df[f'{col}_quantile'] >= n_quantiles - 1).astype(int)
        df[f'{col}_is_bottom_quantile'] = (df[f'{col}_quantile'] <= 0).astype(int)
    
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
    
    # Advanced momentum features
    momentum_features = compute_advanced_momentum_features(df, team_id_col, metric_cols, n_windows=[3, 5, 10])
    df = df.merge(momentum_features, on=[team_id_col, 'date'], how='left')
    
    # Rest-travel interactions
    df = create_rest_travel_interaction_features(df, rest_col, travel_col)
    
    # Team performance profiles
    df = generate_team_performance_profiles(df, team_id_col, metric_cols, n_quantiles=5)
    
    return df

# Example usage:
# df = pd.DataFrame({
#     'team_id': [1,1,1,2,2,2],
#     'opponent_id': [2,2,2,1,1,1],
#     'date': pd.date_range('2024-01-01', periods=6),
#     'points': [100, 110, 120, 95, 105, 115],
#     'opponent_points': [90, 100, 110, 105, 95, 85],
#     'team_strength': [1.2, 1.1, 1.3, 0.9, 1.0, 0.8],
#     'opponent_strength': [0.9, 0.8, 1.0, 1.1, 1.2, 1.0],
#     'rest_days': [2, 1, 3, 0, 2, 1],
#     'travel_distance': [0, 200, 500, 300, 0, 150]
# })
# 
# advanced_df = create_advanced_feature_engineering_pipeline(
#     df, 
#     team_id_col='team_id', 
#     opponent_id_col='opponent_id', 
#     metric_cols=['points', 'opponent_points'], 
#     rest_col='rest_days', 
#     travel_col='travel_distance'
# )
