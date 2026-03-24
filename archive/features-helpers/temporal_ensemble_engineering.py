import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from typing import Dict, List

def create_temporal_ensemble_features(df: pd.DataFrame, target_col: str = 'actual', 
                                     temporal_window: int = 30) -> pd.DataFrame:
    """
    Create ensemble features with temporal decay weighting to capture recent performance trends.
    
    Parameters:
    df (pd.DataFrame): Input data with features and target column
    target_col (str): Name of target column
    temporal_window (int): Temporal window size in days for decay weighting
    
    Returns:
    pd.DataFrame: DataFrame with original features plus ensemble predictions with temporal weighting
    """
    df = df.copy()
    
    # Select numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove(target_col)
    
    if len(numeric_features) < 5:
        return df  # Not enough features for ensemble
    
    # Split data
    X = df[numeric_features]
    y = df[target_col]
    
    # Train multiple models with temporal awareness
    models = {
        'random_forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        'gradient_boost': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42)
    }
    
    ensemble_predictions = {}
    temporal_weights = {}
    
    # Calculate temporal decay weights
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
        current_date = df['game_date'].max()
        date_diffs = (current_date - df['game_date']).dt.days
        temporal_weights = np.exp(-date_diffs / temporal_window)
    else:
        temporal_weights = np.ones(len(df))
    
    for name, model in models.items():
        try:
            # Train-test split with temporal ordering
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, temporal_weights, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model with sample weights
            model.fit(X_train, y_train, sample_weight=w_train)
            
            # Get predictions with temporal weighting
            df[f'{name}_prob'] = model.predict_proba(X)[:, 1]
            df[f'{name}_pred'] = model.predict(X)
            
            # Apply temporal weighting to predictions
            df[f'{name}_prob_weighted'] = df[f'{name}_prob'] * (temporal_weights / temporal_weights.max())
            
            ensemble_predictions[name] = df[f'{name}_prob_weighted']
        except Exception as e:
            print(f"Error training {name}: {e}")
            df[f'{name}_prob'] = 0.5
            df[f'{name}_prob_weighted'] = 0.5
            df[f'{name}_pred'] = 0
    
    # Create ensemble features with temporal weighting
    if ensemble_predictions:
        # Weighted ensemble prediction (more recent games have higher weight)
        df['ensemble_prob'] = np.average(list(ensemble_predictions.values()), axis=0, weights=temporal_weights)
        df['ensemble_prob_weighted'] = df['ensemble_prob'] * (temporal_weights / temporal_weights.max())
        df['ensemble_pred'] = (df['ensemble_prob_weighted'] > 0.5).astype(int)
        
        # Ensemble confidence score
        df['ensemble_confidence'] = np.abs(df['ensemble_prob_weighted'] - 0.5)
        
        # Ensemble volatility (std dev of model predictions)
        df['ensemble_volatility'] = np.std(np.array(list(ensemble_predictions.values())), axis=0)
    
    return df

def compute_temporal_feature_importance(df: pd.DataFrame, target_col: str = 'actual') -> pd.DataFrame:
    """
    Compute feature importance with temporal decay weighting.
    
    Parameters:
    df (pd.DataFrame): Input data with ensemble features
    target_col (str): Name of target column
    
    Returns:
    pd.DataFrame: Feature importance dataframe
    """
    df = df.copy()
    
    # Select numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove(target_col)
    
    if len(numeric_features) < 5:
        return pd.DataFrame()
    
    # Train ensemble model for feature importance with temporal weighting
    X = df[numeric_features]
    y = df[target_col]
    
    # Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    # Apply temporal weighting if available
    if 'game_date' in df.columns:
        current_date = df['game_date'].max()
        date_diffs = (current_date - df['game_date']).dt.days
        temporal_weights = np.exp(-date_diffs / 30)
        rf.fit(X, y, sample_weight=temporal_weights)
    else:
        rf.fit(X, y)
    
    feature_importances = rf.feature_importances_
    feature_names = numeric_features
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances,
        'temporal_importance': feature_importances * (temporal_weights.mean() if 'game_date' in df.columns else 1)
    })
    
    importance_df = importance_df.sort_values('temporal_importance', ascending=False).reset_index(drop=True)
    
    return importance_df

def create_temporal_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features using ensemble predictions and temporal dynamics.
    
    Parameters:
    df (pd.DataFrame): Input data with ensemble features
    
    Returns:
    pd.DataFrame: DataFrame with interaction features
    """
    df = df.copy()
    
    # Create interaction features with ensemble predictions
    if 'ensemble_prob' in df.columns:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Interaction with ensemble predictions
        for feature in numeric_features:
            if feature not in ['ensemble_prob', 'ensemble_pred', 'ensemble_prob_weighted']:
                df[f'ensemble_x_{feature}'] = df['ensemble_prob'] * df[feature]
        
        # Temporal interaction features
        if 'game_date' in df.columns:
            df['days_since_season_start'] = (df['game_date'] - df['game_date'].min()).dt.days
            df['season_progress'] = df['days_since_season_start'] / df['days_since_season_start'].max()
            
            # Seasonal trend features
            df['seasonal_trend'] = df['season_progress'] * df['ensemble_prob']
            df['seasonal_volatility'] = df['season_progress'] * df['ensemble_volatility']
    
    return df

def compute_temporal_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute advanced temporal performance metrics.
    
    Parameters:
    df (pd.DataFrame): Game data with basic statistics
    
    Returns:
    pd.DataFrame: DataFrame with temporal performance metrics
    """
    df = df.copy()
    
    # Momentum metrics with temporal decay
    if 'game_date' in df.columns:
        df = df.sort_values('game_date')
        
        # Rolling statistics with temporal weighting
        df['weighted_points'] = df['points'] * (df['game_date'] - df['game_date'].min()).dt.days / 100
        
        # Recent performance with exponential decay
        df['recent_performance'] = df['points'] * np.exp(-(df['game_date'].max() - df['game_date']).dt.days / 7)
    
    # Advanced efficiency metrics
    if 'points' in df.columns and 'pace' in df.columns:
        df['offensive_efficiency'] = df['points'] / (df['pace'] / 100)
    
    if 'opponent_points' in df.columns and 'pace' in df.columns:
        df['defensive_efficiency'] = df['opponent_points'] / (df['pace'] / 100)
    
    if 'offensive_efficiency' in df.columns and 'defensive_efficiency' in df.columns:
        df['net_efficiency'] = df['offensive_efficiency'] - df['defensive_efficiency']
    
    return df

def create_complete_temporal_feature_set(df: pd.DataFrame, target_col: str = 'actual') -> pd.DataFrame:
    """
    Generate complete feature set with temporal ensemble engineering.
    
    Parameters:
    df (pd.DataFrame): Raw game data
    target_col (str): Target column name
    
    Returns:
    pd.DataFrame: DataFrame with comprehensive temporal feature engineering
    """
    # Create ensemble features with temporal weighting
    df = create_temporal_ensemble_features(df, target_col)
    
    # Create interaction features
    df = create_temporal_interaction_features(df)
    
    # Compute temporal performance metrics
    df = compute_temporal_performance_metrics(df)
    
    # Normalize key features
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if len(numeric_cols) > 0:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def evaluate_temporal_ensemble_performance(df: pd.DataFrame, target_col: str = 'actual') -> dict:
    """
    Evaluate temporal ensemble model performance.
    
    Parameters:
    df (pd.DataFrame): Data with ensemble predictions
    target_col (str): Target column name
    
    Returns:
    dict: Performance metrics
    """
    results = {}
    
    if 'ensemble_prob_weighted' not in df.columns:
        return {'brier_score': 1.0, 'accuracy': 0.0}
    
    probs = df['ensemble_prob_weighted'].values
    actual = df[target_col].values
    
    # Brier score with temporal weighting
    if 'game_date' in df.columns:
        current_date = df['game_date'].max()
        date_diffs = (current_date - df['game_date']).dt.days
        temporal_weights = np.exp(-date_diffs / 30)
        results['weighted_brier_score'] = np.mean(temporal_weights * (probs - actual) ** 2)
    else:
        results['weighted_brier_score'] = np.mean((probs - actual) ** 2)
    
    # Accuracy
    predicted_labels = (probs > 0.5).astype(int)
    results['accuracy'] = np.mean(predicted_labels == actual)
    
    # Log-loss
    from sklearn.metrics import log_loss
    results['log_loss'] = log_loss(actual, probs, eps=1e-15)
    
    # Temporal calibration metrics
    if 'game_date' in df.columns:
        results['temporal_calibration'] = np.corrcoef(probs, (current_date - df['game_date']).dt.days)[0, 1]
    
    return results

def generate_temporal_feature_importance_report(df: pd.DataFrame, target_col: str = 'actual') -> pd.DataFrame:
    """
    Generate feature importance report for temporal ensemble features.
    
    Parameters:
    df (pd.DataFrame): Data with temporal ensemble features
    target_col (str): Target column name
    
    Returns:
    pd.DataFrame: Feature importance dataframe
    """
    importance_df = compute_temporal_feature_importance(df, target_col)
    
    if importance_df.empty:
        return pd.DataFrame()
    
    # Add temporal importance ranking
    importance_df['temporal_rank'] = importance_df['temporal_importance'].rank(ascending=False)
    
    return importance_df

# Example usage:
# df = pd.read_csv('data/games.csv')
# df = create_complete_temporal_feature_set(df)
# performance = evaluate_temporal_ensemble_performance(df)
# print(f"Temporal Ensemble Brier Score: {performance['weighted_brier_score']:.4f}")
# print(f"Temporal Ensemble Accuracy: {performance['accuracy']:.4f}")
# importance_df = generate_temporal_feature_importance_report(df)
# print(importance_df.head(10))

