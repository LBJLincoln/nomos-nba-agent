import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List

def create_ensemble_features(df: pd.DataFrame, target_col: str = 'actual') -> pd.DataFrame:
    """
    Create ensemble features by training multiple models and using their predictions as features.
    
    Parameters:
    df (pd.DataFrame): Input data with features and target column
    target_col (str): Name of target column
    
    Returns:
    pd.DataFrame: DataFrame with original features plus ensemble predictions
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
    
    # Train multiple models
    models = {
        'random_forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        'gradient_boost': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42)
    }
    
    ensemble_predictions = {}
    
    for name, model in models.items():
        try:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions
            df[f'{name}_prob'] = model.predict_proba(X)[:, 1]
            df[f'{name}_pred'] = model.predict(X)
            
            ensemble_predictions[name] = df[f'{name}_prob']
        except Exception as e:
            print(f"Error training {name}: {e}")
            df[f'{name}_prob'] = 0.5
            df[f'{name}_pred'] = 0
    
    # Create ensemble features
    if ensemble_predictions:
        df['ensemble_prob'] = np.mean(list(ensemble_predictions.values()), axis=0)
        df['ensemble_pred'] = (df['ensemble_prob'] > 0.5).astype(int)
    
    return df

def compute_feature_importance_ensemble(df: pd.DataFrame, target_col: str = 'actual') -> pd.DataFrame:
    """
    Compute feature importance using ensemble methods and create importance-weighted features.
    
    Parameters:
    df (pd.DataFrame): Input data with features and target column
    target_col (str): Name of target column
    
    Returns:
    pd.DataFrame: DataFrame with importance-weighted features
    """
    df = df.copy()
    
    # Select numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove(target_col)
    
    if len(numeric_features) < 5:
        return df
    
    # Train ensemble model for feature importance
    X = df[numeric_features]
    y = df[target_col]
    
    # Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)
    
    feature_importances = rf.feature_importances_
    feature_names = numeric_features
    
    # Create importance-weighted features
    for i, feature in enumerate(feature_names):
        if feature_importances[i] > 0.05:  # Only for important features
            importance = feature_importances[i]
            df[f'{feature}_importance_weighted'] = df[feature] * importance
    
    return df

def create_interaction_features_with_ensembles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features using ensemble predictions and original features.
    
    Parameters:
    df (pd.DataFrame): Input data
    
    Returns:
    pd.DataFrame: DataFrame with interaction features
    """
    df = df.copy()
    
    # Create ensemble features first
    df = create_ensemble_features(df)
    df = compute_feature_importance_ensemble(df)
    
    # Create interaction features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Interaction with ensemble predictions
    if 'ensemble_prob' in df.columns:
        for feature in numeric_features:
            if feature not in ['ensemble_prob', 'ensemble_pred']:
                df[f'ensemble_x_{feature}'] = df['ensemble_prob'] * df[feature]
    
    # Interaction between important features
    important_features = [col for col in numeric_features if '_importance_weighted' in col]
    
    for i in range(len(important_features)):
        for j in range(i + 1, len(important_features)):
            feat1 = important_features[i]
            feat2 = important_features[j]
            df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
    
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
    if 'points' in df.columns and 'pace' in df.columns:
        df['offensive_efficiency'] = df['points'] / (df['pace'] / 100)
    
    if 'opponent_points' in df.columns and 'pace' in df.columns:
        df['defensive_efficiency'] = df['opponent_points'] / (df['pace'] / 100)
    
    if 'offensive_efficiency' in df.columns and 'defensive_efficiency' in df.columns:
        df['net_efficiency'] = df['offensive_efficiency'] - df['defensive_efficiency']
    
    # True shooting percentage
    if 'points' in df.columns and 'fga' in df.columns and 'fta' in df.columns:
        df['true_shooting'] = df['points'] / (2 * (df['fga'] + 0.44 * df['fta']))
    
    # Rebound percentage
    if 'orb' in df.columns and 'drb' in df.columns and 'opponent_orb' in df.columns and 'opponent_drb' in df.columns:
        df['total_rebounds'] = df['orb'] + df['drb']
        df['rebounding_percentage'] = df['total_rebounds'] / (df['total_rebounds'] + df['opponent_orb'] + df['opponent_drb'])
    
    # Turnover ratio
    if 'tov' in df.columns and 'fga' in df.columns and 'fta' in df.columns:
        df['turnover_ratio'] = df['tov'] / (df['fga'] + 0.44 * df['fta'] + df['tov'])
    
    return df

def create_complete_feature_set(df: pd.DataFrame, target_col: str = 'actual') -> pd.DataFrame:
    """
    Generate complete feature set combining all advanced techniques.
    
    Parameters:
    df (pd.DataFrame): Raw game data
    target_col (str): Target column name
    
    Returns:
    pd.DataFrame: DataFrame with comprehensive feature engineering
    """
    # Create ensemble features
    df = create_ensemble_features(df, target_col)
    
    # Create interaction features
    df = create_interaction_features_with_ensembles(df)
    
    # Compute advanced performance metrics
    df = compute_advanced_performance_metrics(df)
    
    # Normalize key features
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if len(numeric_cols) > 0:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def evaluate_ensemble_performance(df: pd.DataFrame, target_col: str = 'actual') -> dict:
    """
    Evaluate ensemble model performance.
    
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
    
    # Brier score
    results['brier_score'] = np.mean((probs - actual) ** 2)
    
    # Accuracy
    predicted_labels = (probs > 0.5).astype(int)
    results['accuracy'] = np.mean(predicted_labels == actual)
    
    # Log-loss
    from sklearn.metrics import log_loss
    results['log_loss'] = log_loss(actual, probs, eps=1e-15)
    
    return results

# Example usage:
# df = pd.read_csv('data/games.csv')
# df = create_complete_feature_set(df)
# performance = evaluate_ensemble_performance(df)
# print(f"Ensemble Brier Score: {performance['brier_score']:.4f}")
# print(f"Ensemble Accuracy: {performance['accuracy']:.4f}")
