import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss
from typing import Dict, List, Tuple
from datetime import datetime

def create_temporal_ensemble_features(df: pd.DataFrame, target_col: str = 'actual', 
                                     temporal_decay: float = 0.95) -> pd.DataFrame:
    """
    Create ensemble features with temporal decay weighting for recency bias.
    
    Parameters:
    df (pd.DataFrame): Input data with features and target column
    target_col (str): Name of target column
    temporal_decay (float): Decay factor for temporal weighting (0.9-0.99)
    
    Returns:
    pd.DataFrame: DataFrame with original features plus ensemble predictions
    """
    df = df.copy()
    
    # Convert date to datetime if needed
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Select numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove(target_col)
    
    if len(numeric_features) < 5:
        return df  # Not enough features for ensemble
    
    # Split data
    X = df[numeric_features]
    y = df[target_col]
    
    # Train multiple models with temporal weighting
    models = {
        'random_forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        'gradient_boost': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42)
    }
    
    ensemble_predictions = {}
    temporal_weights = {}
    
    for name, model in models.items():
        try:
            # Apply temporal decay weighting
            if 'date' in df.columns:
                # Calculate temporal weights (more recent = higher weight)
                max_date = df['date'].max()
                df['temporal_weight'] = np.exp((df['date'] - max_date).dt.days / 365 * np.log(temporal_decay))
                temporal_weights[name] = df['temporal_weight'].values
            else:
                temporal_weights[name] = np.ones(len(df))
            
            # Train-test split with stratification
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, temporal_weights[name], test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model with sample weights
            model.fit(X_train, y_train, sample_weight=w_train)
            
            # Get predictions
            df[f'{name}_prob'] = model.predict_proba(X)[:, 1]
            df[f'{name}_pred'] = model.predict(X)
            
            ensemble_predictions[name] = df[f'{name}_prob']
        except Exception as e:
            print(f"Error training {name}: {e}")
            df[f'{name}_prob'] = 0.5
            df[f'{name}_pred'] = 0
    
    # Create ensemble features with adaptive weighting
    if ensemble_predictions:
        # Calculate model performance for weighting
        model_performances = {}
        for name, probs in ensemble_predictions.items():
            if len(probs) > 0:
                model_performances[name] = 1 - brier_score_loss(y, probs)
        
        # Normalize performances to get weights
        if model_performances:
            total_performance = sum(model_performances.values())
            model_weights = {k: v/total_performance for k, v in model_performances.items()}
        else:
            model_weights = {k: 1/len(ensemble_predictions) for k in ensemble_predictions.keys()}
        
        # Create weighted ensemble prediction
        ensemble_prob = np.zeros(len(df))
        for name, probs in ensemble_predictions.items():
            weight = model_weights.get(name, 1/len(ensemble_predictions))
            ensemble_prob += probs * weight
        
        df['ensemble_prob'] = ensemble_prob
        df['ensemble_pred'] = (df['ensemble_prob'] > 0.5).astype(int)
        
        # Add model weights as features
        for name, weight in model_weights.items():
            df[f'{name}_weight'] = weight
    
    return df

def compute_feature_interaction_importance(df: pd.DataFrame, target_col: str = 'actual') -> pd.DataFrame:
    """
    Compute feature interaction importance using ensemble methods.
    
    Parameters:
    df (pd.DataFrame): Input data with features and target column
    target_col (str): Name of target column
    
    Returns:
    pd.DataFrame: DataFrame with interaction features and importance scores
    """
    df = df.copy()
    
    # Select numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove(target_col)
    
    if len(numeric_features) < 5:
        return df
    
    # Train Random Forest for feature importance
    X = df[numeric_features]
    y = df[target_col]
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)
    
    feature_importances = rf.feature_importances_
    feature_names = numeric_features
    
    # Create interaction features for top important features
    important_features = [feature_names[i] for i in np.argsort(feature_importances)[-5:]]
    
    for i in range(len(important_features)):
        for j in range(i + 1, len(important_features)):
            feat1 = important_features[i]
            feat2 = important_features[j]
            interaction_name = f'interact_{feat1}_x_{feat2}'
            df[interaction_name] = df[feat1] * df[feat2]
            
            # Add importance score
            importance_score = feature_importances[feature_names.index(feat1)] * \
                             feature_importances[feature_names.index(feat2)]
            df[f'{interaction_name}_importance'] = importance_score
    
    return df

def create_advanced_ensemble_pipeline(df: pd.DataFrame, target_col: str = 'actual') -> pd.DataFrame:
    """
    Complete ensemble feature engineering pipeline combining multiple techniques.
    
    Parameters:
    df (pd.DataFrame): Raw game data
    target_col (str): Target column name
    
    Returns:
    pd.DataFrame: DataFrame with comprehensive ensemble features
    """
    # Create temporal ensemble features
    df = create_temporal_ensemble_features(df, target_col)
    
    # Create feature interaction importance
    df = compute_feature_interaction_importance(df, target_col)
    
    # Normalize key features
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if len(numeric_cols) > 0:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def evaluate_ensemble_performance(df: pd.DataFrame, target_col: str = 'actual') -> dict:
    """
    Evaluate ensemble model performance with temporal decay.
    
    Parameters:
    df (pd.DataFrame): Data with ensemble predictions
    target_col (str): Target column name
    
    Returns:
    dict: Performance metrics
    """
    results = {}
    
    if 'ensemble_prob' not in df.columns:
        return {'brier_score': 1.0, 'accuracy': 0.0}
    
    probs = df['ensemble_prob'].values
    actual = df[target_col].values
    
    # Brier score with temporal weighting
    if 'date' in df.columns:
        max_date = df['date'].max()
        temporal_weights = np.exp((df['date'] - max_date).dt.days / 365 * np.log(0.95))
        results['temporal_brier'] = np.average((probs - actual) ** 2, weights=temporal_weights)
    else:
        results['temporal_brier'] = np.mean((probs - actual) ** 2)
    
    # Standard metrics
    results['brier_score'] = np.mean((probs - actual) ** 2)
    results['accuracy'] = np.mean((probs > 0.5) == actual)
    results['log_loss'] = log_loss(actual, probs, eps=1e-15)
    
    # Model contribution analysis
    model_contributions = {}
    for model_name in ['random_forest', 'gradient_boost', 'logistic_regression']:
        if f'{model_name}_prob' in df.columns:
            model_contributions[model_name] = 1 - brier_score_loss(actual, df[f'{model_name}_prob'])
    
    results['model_contributions'] = model_contributions
    
    return results

# Example usage:
# df = pd.read_csv('data/games.csv')
# df = create_advanced_ensemble_pipeline(df)
# performance = evaluate_ensemble_performance(df)
# print(f"Ensemble Brier Score: {performance['brier_score']:.4f}")
# print(f"Temporal Brier Score: {performance['temporal_brier']:.4f}")
# print(f"Model Contributions: {performance['model_contributions']}")

