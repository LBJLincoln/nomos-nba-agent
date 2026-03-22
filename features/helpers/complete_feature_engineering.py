import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss
from typing import Dict, List

def create_advanced_team_features(df: pd.DataFrame, team_id_col: str = 'team_id', 
                                 opponent_id_col: str = 'opponent_id') -> pd.DataFrame:
    """
    Create comprehensive team-specific features for performance prediction.
    
    Parameters:
    df (pd.DataFrame): Game data with team and opponent statistics
    team_id_col (str): Column name for team ID
    opponent_id_col (str): Column name for opponent ID
    
    Returns:
    pd.DataFrame: DataFrame with advanced team features
    """
    df = df.copy()
    
    # Team performance metrics
    team_stats = df.groupby(team_id_col).agg({
        'points': ['mean', 'std', 'median', 'max', 'min'],
        'opponent_points': ['mean', 'std'],
        'pace': ['mean', 'std'],
        'rest_days': ['mean', 'std']
    }).reset_index()
    team_stats.columns = [team_id_col, 'team_pts_mean', 'team_pts_std', 'team_pts_median',
                         'team_pts_max', 'team_pts_min', 'team_opp_pts_mean', 'team_opp_pts_std',
                         'team_pace_mean', 'team_pace_std', 'team_rest_mean', 'team_rest_std']
    
    # Opponent strength metrics
    opponent_strength = df.groupby(opponent_id_col)['points'].agg([
        ('opp_strength_mean', 'mean'),
        ('opp_strength_std', 'std')
    ]).reset_index()
    
    # Merge back to original data
    df = df.merge(team_stats, on=team_id_col, how='left')
    df = df.merge(opponent_strength, left_on=opponent_id_col, right_on=opponent_id_col, how='left')
    
    # Advanced interaction features
    df['strength_diff'] = df['team_pts_mean'] - df['opp_strength_mean']
    df['pace_adjusted_strength'] = df['team_pace_mean'] * df['team_pts_mean']
    df['rest_pace_interaction'] = df['team_rest_mean'] * df['team_pace_mean']
    df['volatility_ratio'] = df['team_pts_std'] / (df['team_opp_pts_mean'] + 1)
    
    return df

def compute_advanced_momentum_features(df: pd.DataFrame, window_sizes: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """
    Compute multi-window momentum features for performance trends.
    
    Parameters:
    df (pd.DataFrame): Game data with datetime index
    window_sizes (List[int]): List of window sizes for momentum calculation
    
    Returns:
    pd.DataFrame: DataFrame with momentum features
    """
    df = df.sort_values('game_date')
    
    momentum_features = []
    
    for window in window_sizes:
        # Rolling statistics
        rolling_stats = df.groupby('team_id')[['points', 'opponent_points']].rolling(window=window).agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('trend', lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0),
            ('momentum', lambda x: (x - x.shift(1)).sum())
        ]).reset_index()
        
        rolling_stats.columns = ['team_id', 'date', 'points_roll_mean', 'points_roll_std',
                               'points_roll_trend', 'points_roll_momentum',
                               'opp_points_roll_mean', 'opp_points_roll_std',
                               'opp_points_roll_trend', 'opp_points_roll_momentum']
        
        momentum_features.append(rolling_stats)
    
    # Merge all momentum features
    result = momentum_features[0]
    for mf in momentum_features[1:]:
        result = result.merge(mf, on=['team_id', 'date'], how='left', suffixes=('', f'_win{window}'))
    
    return result

def create_rest_travel_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive rest and travel interaction features.
    
    Parameters:
    df (pd.DataFrame): Game data with rest and travel information
    
    Returns:
    pd.DataFrame: DataFrame with rest-travel interaction features
    """
    df = df.copy()
    
    # Rest categories
    df['rest_category'] = pd.cut(df['rest_days'], 
                                 bins=[-1, 0, 1, 2, 3, 4, np.inf],
                                 labels=['no_rest', 'minimal', 'short', 'medium', 'long', 'extended'])
    
    # Travel categories
    df['travel_category'] = pd.cut(df['travel_distance'],
                                   bins=[0, 100, 500, 1000, 2000, np.inf],
                                   labels=['local', 'regional', 'national', 'cross_country', 'international'])
    
    # Rest-travel interaction
    df['rest_travel_interaction'] = df['rest_days'] * df['travel_distance']
    
    # Travel fatigue penalty
    df['travel_fatigue'] = np.where(df['travel_distance'] > 1000, 0.3,
                                   np.where(df['travel_distance'] > 500, 0.2, 0.1))
    
    # Rest quality score
    df['rest_quality_score'] = np.where(df['rest_days'] < 1, 0.5,
                                       np.where(df['rest_days'] < 2, 0.7,
                                               np.where(df['rest_days'] < 3, 0.9, 1.0)))
    
    # Back-to-back penalty
    df['back_to_back_penalty'] = np.where(df['rest_days'] < 1, 0.4, 0)
    
    # Combined rest-travel score
    df['rest_travel_score'] = df['rest_quality_score'] * (1 - df['travel_fatigue']) * (1 - df['back_to_back_penalty'])
    
    return df

def compute_advanced_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute advanced opponent-specific features for matchup analysis.
    
    Parameters:
    df (pd.DataFrame): Game data with opponent statistics
    
    Returns:
    pd.DataFrame: DataFrame with advanced opponent features
    """
    df = df.copy()
    
    # Opponent strength decomposition
    df['opponent_tier'] = pd.qcut(df['opponent_strength'], q=5, labels=False)
    df['strength_diff_abs'] = np.abs(df['team_strength'] - df['opponent_strength'])
    
    # Matchup difficulty indicators
    df['is_tough_matchup'] = ((df['opponent_tier'] >= 3) & (df['strength_diff_abs'] < 0.5)).astype(int)
    df['is_easy_matchup'] = ((df['opponent_tier'] <= 1) & (df['strength_diff_abs'] > 0.5)).astype(int)
    
    # Historical performance against opponent type
    opponent_type_stats = df.groupby(['opponent_tier', 'team_id'])['points'].agg([
        ('mean', 'mean'),
        ('std', 'std')
    ]).reset_index()
    
    opponent_type_stats.columns = ['opponent_tier', 'team_id', 'opp_type_pts_mean', 'opp_type_pts_std']
    
    df = df.merge(opponent_type_stats, on=['opponent_tier', 'team_id'], how='left')
    
    # Revenge game indicator (if team lost to this opponent recently)
    df['last_opponent_game'] = df.groupby('team_id')['game_date'].shift(1)
    df['is_revenge_game'] = ((df['last_opponent_game'] == df['game_date']) & 
                            (df['points'] < df['opponent_points'])).astype(int)
    
    return df

def create_polynomial_interaction_features(df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial interaction features to capture non-linear relationships.
    
    Parameters:
    df (pd.DataFrame): Input data
    degree (int): Degree of polynomial features
    
    Returns:
    pd.DataFrame: DataFrame with polynomial interaction features
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Select numeric features for polynomial expansion
    numeric_features = df.select_dtypes(include=[np.number])
    interaction_matrix = poly.fit_transform(numeric_features)
    
    # Create feature names
    feature_names = poly.get_feature_names_out(numeric_features.columns)
    
    # Create DataFrame with interaction features
    interaction_df = pd.DataFrame(interaction_matrix, columns=feature_names, index=df.index)
    
    return pd.concat([df, interaction_df], axis=1)

def generate_complete_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate complete feature set combining all advanced features.
    
    Parameters:
    df (pd.DataFrame): Raw game data
    
    Returns:
    pd.DataFrame: DataFrame with comprehensive feature engineering
    """
    # Create advanced team features
    df = create_advanced_team_features(df)
    
    # Add momentum features
    momentum_features = compute_advanced_momentum_features(df)
    df = df.merge(momentum_features, on=['team_id', 'date'], how='left')
    
    # Add rest-travel interaction features
    df = create_rest_travel_interaction_features(df)
    
    # Add advanced opponent features
    df = compute_advanced_opponent_features(df)
    
    # Create polynomial interactions
    df = create_polynomial_interaction_features(df, degree=2)
    
    # Normalize key features
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
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

def evaluate_model_performance(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Evaluate model performance using multiple metrics.
    
    Parameters:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target labels
    
    Returns:
    dict: Performance metrics
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train multiple models
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'brier_score': brier_score_loss(y_test, y_pred),
                'log_loss': log_loss(y_test, y_pred),
                'accuracy': np.mean((y_pred > 0.5) == y_test)
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results

# Example usage:
# df = pd.read_csv('data/games.csv')
# df = generate_complete_feature_set(df)
# df = compute_advanced_performance_metrics(df)
# 
# X = df.select_dtypes(include=[np.number]).drop(columns=['actual'])
# y = df['actual']
# 
# performance = evaluate_model_performance(X, y)
# print("Model Performance:")
# for model, metrics in performance.items():
#     if 'error' not in metrics:
#         print(f"{model}: Brier={metrics['brier_score']:.4f}, LogLoss={metrics['log_loss']:.4f}, Accuracy={metrics['accuracy']:.4f}")

