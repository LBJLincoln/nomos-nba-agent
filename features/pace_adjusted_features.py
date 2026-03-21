import numpy as np
import pandas as pd

def compute_pace_adjusted_ppp(pts, pace, league_pace=100.0):
    """
    Compute pace-adjusted points per possession.
    
    Parameters:
    pts (float): Points scored per possession.
    pace (float): Team's pace.
    league_pace (float): League average pace.
    
    Returns:
    float: Pace-adjusted points per possession.
    """
    return (pts / pace) * league_pace if pace > 0 else pts

def compute_rest_weighted_performance(pts, rest_days):
    """
    Compute rest-weighted performance.
    
    Parameters:
    pts (float): Points scored.
    rest_days (int): Number of rest days.
    
    Returns:
    float: Rest-weighted performance.
    """
    # Assuming a linear relationship between rest days and performance
    return pts * (1 + 0.05 * rest_days)

def add_pace_adjusted_features(df):
    """
    Add pace-adjusted features to the dataframe.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing team statistics.
    
    Returns:
    pd.DataFrame: Dataframe with added pace-adjusted features.
    """
    df['pace_adjusted_ppp'] = df.apply(lambda row: compute_pace_adjusted_ppp(row['pts_per_poss'], row['pace']), axis=1)
    df['rest_weighted_performance'] = df.apply(lambda row: compute_rest_weighted_performance(row['pts'], row['rest_days']), axis=1)
    return df
