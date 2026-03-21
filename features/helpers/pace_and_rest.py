import pandas as pd
import numpy as np

def compute_pace_adjusted_points(df: pd.DataFrame) -> pd.Series:
    """
    Compute pace-adjusted points for each team.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'points' and 'pace' columns.
    
    Returns:
    pd.Series: Pace-adjusted points for each team.
    """
    league_pace = df['pace'].mean()
    return (df['points'] / df['pace']) * league_pace

def compute_rest_weighted_performance(df: pd.DataFrame) -> pd.Series:
    """
    Compute rest-weighted performance for each team.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'points', 'rest_days' columns.
    
    Returns:
    pd.Series: Rest-weighted performance for each team.
    """
    # Assign weights based on rest days
    weights = np.where(df['rest_days'] > 2, 1.2, np.where(df['rest_days'] < 1, 0.8, 1))
    return df['points'] * weights

# Example usage:
# df = pd.DataFrame({'points': [100, 120, 110], 'pace': [100, 105, 95], 'rest_days': [3, 1, 2]})
# df['pace_adjusted_points'] = compute_pace_adjusted_points(df)
# df['rest_weighted_performance'] = compute_rest_weighted_performance(df)
