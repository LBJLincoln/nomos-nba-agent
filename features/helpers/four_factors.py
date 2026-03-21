import numpy as np
import pandas as pd

def compute_four_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Dean Oliver's Four Factors for both teams in a game.
    
    Parameters:
    df (pd.DataFrame): Game data with columns:
        - 'team_id', 'opponent_id'
        - 'team_fgm', 'team_fga', 'team_fg3m', 'team_fg3a'
        - 'team_ftm', 'team_fta', 'team_orb', 'team_trb'
        - 'team_tov', 'team_pts', 'team_poss'
        - 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_fg3a'
        - 'opp_ftm', 'opp_fta', 'opp_orb', 'opp_trb'
        - 'opp_tov', 'opp_pts', 'opp_poss'
    
    Returns:
    pd.DataFrame: Original dataframe with four factors for both teams
    """
    df = df.copy()
    
    # Offensive Four Factors
    df['off_efg'] = (df['team_fgm'] + 0.5 * df['team_fg3m']) / df['team_fga']
    df['off_tov'] = df['team_tov'] / (df['team_fga'] + 0.44 * df['team_fta'] + df['team_tov'])
    df['off_orb'] = df['team_orb'] / (df['team_orb'] + (df['opp_trb'] - df['opp_orb']))
    df['off_ftr'] = df['team_fta'] / df['team_fga']
    
    # Defensive Four Factors
    df['def_efg'] = (df['opp_fgm'] + 0.5 * df['opp_fg3m']) / df['opp_fga']
    df['def_tov'] = df['opp_tov'] / (df['opp_fga'] + 0.44 * df['opp_fta'] + df['opp_tov'])
    df['def_orb'] = df['opp_orb'] / (df['opp_orb'] + (df['team_trb'] - df['team_orb']))
    df['def_ftr'] = df['opp_fta'] / df['opp_fga']
    
    # Impact metrics (difference between offensive and defensive)
    df['efg_impact'] = df['off_efg'] - df['def_efg']
    df['tov_impact'] = df['off_tov'] - df['def_tov']
    df['orb_impact'] = df['off_orb'] - df['def_orb']
    df['ftr_impact'] = df['off_ftr'] - df['def_ftr']
    
    # Four Factors rating (sum of impacts)
    df['four_factors_rating'] = df['efg_impact'] + df['tov_impact'] + df['orb_impact'] + df['ftr_impact']
    
    return df

def compute_opponent_adjusted_four_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute opponent-adjusted Four Factors to account for strength of competition.
    
    Parameters:
    df (pd.DataFrame): Game data with four factors already computed
    
    Returns:
    pd.DataFrame: Original dataframe with opponent-adjusted four factors
    """
    df = df.copy()
    
    # Calculate league averages for each factor
    league_avg = {
        'off_efg': df['off_efg'].mean(),
        'off_tov': df['off_tov'].mean(),
        'off_orb': df['off_orb'].mean(),
        'off_ftr': df['off_ftr'].mean(),
        'def_efg': df['def_efg'].mean(),
        'def_tov': df['def_tov'].mean(),
        'def_orb': df['def_orb'].mean(),
        'def_ftr': df['def_ftr'].mean()
    }
    
    # Calculate opponent averages for each factor
    opponent_stats = df.groupby('opponent_id')[['off_efg', 'off_tov', 'off_orb', 'off_ftr']].mean().reset_index()
    opponent_stats.columns = ['opponent_id', 'opp_off_efg', 'opp_off_tov', 'opp_off_orb', 'opp_off_ftr']
    
    df = df.merge(opponent_stats, on='opponent_id', how='left')
    
    # Opponent-adjusted factors
    df['off_efg_adj'] = (df['off_efg'] - df['opp_off_efg']) / league_avg['off_efg']
    df['off_tov_adj'] = (df['off_tov'] - df['opp_off_tov']) / league_avg['off_tov']
    df['off_orb_adj'] = (df['off_orb'] - df['opp_off_orb']) / league_avg['off_orb']
    df['off_ftr_adj'] = (df['off_ftr'] - df['opp_off_ftr']) / league_avg['off_ftr']
    
    # Composite opponent-adjusted rating
    df['four_factors_adj_rating'] = (df['off_efg_adj'] + df['off_tov_adj'] + df['off_orb_adj'] + df['off_ftr_adj']) / 4
    
    return df

def compute_four_factors_trends(df: pd.DataFrame, team_id_col: str = 'team_id', window: int = 5) -> pd.DataFrame:
    """
    Calculate rolling trends for Four Factors to identify momentum.
    
    Parameters:
    df (pd.DataFrame): Game data with four factors already computed
    team_id_col (str): Column name for team ID
    window (int): Rolling window size
    
    Returns:
    pd.DataFrame: Original dataframe with rolling trend features
    """
    df = df.copy()
    
    # Rolling statistics for offensive factors
    rolling_stats = df.groupby(team_id_col)[['off_efg', 'off_tov', 'off_orb', 'off_ftr']].rolling(window=window).agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('trend', lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0)
    ]).reset_index()
    
    rolling_stats.columns = [team_id_col, 'date', 
                           'off_efg_roll_mean', 'off_efg_roll_std', 'off_efg_roll_trend',
                           'off_tov_roll_mean', 'off_tov_roll_std', 'off_tov_roll_trend',
                           'off_orb_roll_mean', 'off_orb_roll_std', 'off_orb_roll_trend',
                           'off_ftr_roll_mean', 'off_ftr_roll_std', 'off_ftr_roll_trend']
    
    df = df.merge(rolling_stats, on=[team_id_col, 'date'], how='left')
    
    # Rolling statistics for defensive factors
    rolling_stats_def = df.groupby(team_id_col)[['def_efg', 'def_tov', 'def_orb', 'def_ftr']].rolling(window=window).agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('trend', lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0)
    ]).reset_index()
    
    rolling_stats_def.columns = [team_id_col, 'date', 
                               'def_efg_roll_mean', 'def_efg_roll_std', 'def_efg_roll_trend',
                               'def_tov_roll_mean', 'def_tov_roll_std', 'def_tov_roll_trend',
                               'def_orb_roll_mean', 'def_orb_roll_std', 'def_orb_roll_trend',
                               'def_ftr_roll_mean', 'def_ftr_roll_std', 'def_ftr_roll_trend']
    
    df = df.merge(rolling_stats_def, on=[team_id_col, 'date'], how='left')
    
    return df

def generate_four_factors_summary(df: pd.DataFrame, team_id: int) -> dict:
    """
    Generate comprehensive Four Factors summary for a specific team.
    
    Parameters:
    df (pd.DataFrame): Game data with four factors already computed
    team_id (int): Team ID to analyze
    
    Returns:
    dict: Summary statistics and insights
    """
    team_data = df[df['team_id'] == team_id].copy()
    
    summary = {
        'team_id': team_id,
        'games_analyzed': len(team_data),
        'four_factors_rating': team_data['four_factors_rating'].mean(),
        'four_factors_adj_rating': team_data['four_factors_adj_rating'].mean(),
        'efg_impact': team_data['efg_impact'].mean(),
        'tov_impact': team_data['tov_impact'].mean(),
        'orb_impact': team_data['orb_impact'].mean(),
        'ftr_impact': team_data['ftr_impact'].mean()
    }
    
    # Recent trends (last 10 games)
    recent_data = team_data.tail(10)
    summary['recent_trends'] = {
        'four_factors_rating': recent_data['four_factors_rating'].mean(),
        'efg_impact': recent_data['efg_impact'].mean(),
        'tov_impact': recent_data['tov_impact'].mean(),
        'orb_impact': recent_data['orb_impact'].mean(),
        'ftr_impact': recent_data['ftr_impact'].mean()
    }
    
    # Strength of schedule impact
    summary['strength_of_schedule'] = {
        'avg_opponent_strength': team_data['opponent_strength'].mean(),
        'four_factors_adj_rating': team_data['four_factors_adj_rating'].mean()
    }
    
    return summary

# Example usage:
# df = pd.DataFrame({
#     'team_id': [1,1,2,2],
#     'opponent_id': [2,2,1,1],
#     'team_fgm': [40, 45, 38, 42],
#     'team_fga': [80, 85, 75, 80],
#     'team_fg3m': [10, 12, 8, 10],
#     'team_fg3a': [25, 30, 20, 25],
#     'team_ftm': [15, 18, 12, 15],
#     'team_fta': [20, 25, 15, 20],
#     'team_orb': [10, 12, 8, 11],
#     'team_trb': [40, 45, 35, 42],
#     'team_tov': [12, 15, 10, 13],
#     'team_pts': [105, 110, 95, 100],
#     'team_poss': [90, 95, 85, 90],
#     'opp_fgm': [38, 42, 40, 45],
#     'opp_fga': [75, 80, 80, 85],
#     'opp_fg3m': [8, 10, 10, 12],
#     'opp_fg3a': [20, 25, 25, 30],
#     'opp_ftm': [12, 15, 15, 18],
#     'opp_fta': [15, 20, 20, 25],
#     'opp_orb': [8, 11, 10, 12],
#     'opp_trb': [35, 42, 45, 50],
#     'opp_tov': [10, 13, 12, 15],
#     'opp_pts': [95, 100, 105, 110],
#     'opp_poss': [85, 90, 95, 100],
#     'opponent_strength': [0.9, 1.1, 0.8, 1.2]
# })
# 
# df = compute_four_factors(df)
# df = compute_opponent_adjusted_four_factors(df)
# df = compute_four_factors_trends(df)
# summary = generate_four_factors_summary(df, team_id=1)
