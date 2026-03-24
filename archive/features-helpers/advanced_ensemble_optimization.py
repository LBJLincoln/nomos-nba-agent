import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss
from typing import List, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def compute_advanced_features(games: pd.DataFrame, team_id: int) -> Dict[str, float]:
    """
    Compute advanced features combining multiple signal sources.
    
    Parameters:
    games (pd.DataFrame): Game data with historical performance
    team_id (int): Team ID to analyze
    
    Returns:
    Dict[str, float]: Advanced feature dictionary
    """
    features = {}
    
    # Momentum features
    team_games = games[games['team_id'] == team_id].copy()
    recent_games = team_games.tail(5)
    
    features['weighted_win_streak'] = (recent_games['result'] * 
                                      (1 + recent_games['opponent_strength'])).sum()
    
    features['margin_trend'] = np.polyfit(np.arange(len(recent_games)), 
                                          recent_games['point_diff'], 1)[0]
    
    features['volatility'] = recent_games['point_diff'].std()
    
    # Rest and travel impact
    current_date = pd.to_datetime(team_games['game_date'].iloc[-1])
    last_game = team_games[team_games['game_date'] < current_date].iloc[-1]
    
    features['rest_days'] = (current_date - pd.to_datetime(last_game['game_date'])).days
    features['travel_distance'] = last_game['travel_distance']
    
    # Strength interaction
    features['strength_diff'] = team_games['team_strength'].iloc[-1] - \
                                team_games['opponent_strength'].iloc[-1]
    
    features['strength_interaction'] = (team_games['team_strength'].iloc[-1] * 
                                        team_games['opponent_strength'].iloc[-1])
    
    return features

def create_ensemble_models(X_train: np.ndarray, y_train: np.ndarray) -> List:
    """
    Create diverse ensemble of models with different strengths.
    
    Parameters:
    X_train (np.ndarray): Training features
    y_train (np.ndarray): Training labels
    
    Returns:
    List: List of trained models
    """
    models = []
    
    # Random Forest - good for non-linear relationships
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    models.append(('rf', rf))
    
    # Gradient Boosting - captures complex interactions
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    models.append(('gb', gb))
    
    # Logistic Regression - provides baseline and calibration
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models.append(('lr', lr))
    
    # Neural Network - captures non-linear patterns
    from sklearn.neural_network import MLPClassifier
    nn = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    nn.fit(X_train, y_train)
    models.append(('nn', nn))
    
    return models

def optimize_ensemble_weights(models: List, X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """
    Optimize ensemble weights using validation data.
    
    Parameters:
    models (List): List of trained models
    X_val (np.ndarray): Validation features
    y_val (np.ndarray): Validation labels
    
    Returns:
    np.ndarray: Optimized weights for each model
    """
    # Get predictions from all models
    predictions = np.column_stack([model.predict_proba(X_val)[:, 1] for _, model in models])
    
    # Define objective function (weighted Brier score)
    def objective(weights):
        # Normalize weights
        weights = weights / weights.sum()
        
        # Ensemble prediction
        ensemble_pred = np.dot(predictions, weights)
        
        # Calculate weighted Brier score
        brier = brier_score_loss(y_val, ensemble_pred)
        
        # Add regularization to prevent extreme weights
        weight_penalty = 0.1 * np.sum((weights - 0.5) ** 2)
        
        return brier + weight_penalty
    
    # Initial weights (equal)
    initial_weights = np.ones(len(models)) / len(models)
    
    # Bounds for weights (0 to 1)
    bounds = [(0.01, 0.99) for _ in range(len(models))]
    
    # Constraints: weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Perform optimization
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints,
                     method='SLSQP', options={'maxiter': 1000})
    
    # Return optimized weights
    optimized_weights = result.x / result.x.sum()
    
    return optimized_weights

def ensemble_predict(models: List, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Make ensemble predictions using optimized weights.
    
    Parameters:
    models (List): List of trained models
    weights (np.ndarray): Optimized weights
    X (np.ndarray): Features for prediction
    
    Returns:
    np.ndarray: Ensemble predictions (probabilities)
    """
    # Get predictions from all models
    predictions = np.column_stack([model.predict_proba(X)[:, 1] for _, model in models])
    
    # Compute weighted ensemble prediction
    ensemble_pred = np.dot(predictions, weights)
    
    return ensemble_pred

def evaluate_ensemble_performance(models: List, weights: np.ndarray, 
                                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate ensemble performance on test data.
    
    Parameters:
    models (List): List of trained models
    weights (np.ndarray): Optimized weights
    X_test (np.ndarray): Test features
    y_test (np.ndarray): Test labels
    
    Returns:
    Dict[str, float]: Performance metrics
    """
    results = {}
    
    # Ensemble prediction
    ensemble_pred = ensemble_predict(models, weights, X_test)
    
    # Brier score
    results['brier_score'] = brier_score_loss(y_test, ensemble_pred)
    
    # Log loss
    results['log_loss'] = log_loss(y_test, ensemble_pred)
    
    # Accuracy
    results['accuracy'] = np.mean((ensemble_pred > 0.5) == y_test)
    
    # Per-model performance
    results['per_model_performance'] = {}
    for i, (_, model) in enumerate(models):
        model_pred = model.predict_proba(X_test)[:, 1]
        results['per_model_performance'][f'model_{i}'] = {
            'brier_score': brier_score_loss(y_test, model_pred),
            'log_loss': log_loss(y_test, model_pred),
            'accuracy': np.mean((model_pred > 0.5) == y_test)
        }
    
    return results

def generate_ensemble_report(models: List, weights: np.ndarray, 
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Generate comprehensive ensemble report.
    
    Parameters:
    models (List): List of trained models
    weights (np.ndarray): Optimized weights
    X_test (np.ndarray): Test features
    y_test (np.ndarray): Test labels
    
    Returns:
    Dict: Comprehensive report with metrics and analysis
    """
    report = {}
    
    # Performance evaluation
    report['performance'] = evaluate_ensemble_performance(models, weights, X_test, y_test)
    
    # Weight analysis
    report['weights'] = {
        'optimized_weights': weights.tolist(),
        'weight_ranks': np.argsort(weights)[::-1].tolist(),
        'top_3_models': np.argsort(weights)[-3:][::-1].tolist()
    }
    
    # Improvement metrics
    baseline_brier = np.mean([report['performance']['per_model_performance'][f'model_{i}']['brier_score'] 
                             for i in range(len(models))])
    ensemble_brier = report['performance']['brier_score']
    
    report['improvement'] = {
        'brier_improvement': baseline_brier - ensemble_brier,
        'percentage_improvement': ((baseline_brier - ensemble_brier) / baseline_brier * 100) if baseline_brier > 0 else 0
    }
    
    # Model diversity analysis
    predictions = np.column_stack([model.predict_proba(X_test)[:, 1] for _, model in models])
    correlation_matrix = np.corrcoef(predictions.T)
    report['model_diversity'] = {
        'average_correlation': np.mean(correlation_matrix[np.triu_indices(len(models), 1)]),
        'correlation_matrix': correlation_matrix.tolist()
    }
    
    return report

def create_feature_engineering_pipeline(games: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive feature engineering pipeline.
    
    Parameters:
    games (pd.DataFrame): Game data with historical performance
    
    Returns:
    pd.DataFrame: DataFrame with engineered features
    """
    # Calculate team-specific features
    team_features = []
    for team_id in games['team_id'].unique():
        team_games = games[games['team_id'] == team_id].copy()
        
        # Advanced features
        features = compute_advanced_features(games, team_id)
        
        # Rolling statistics
        team_games['point_diff'] = team_games['points'] - team_games['opponent_points']
        
        rolling_stats = team_games['point_diff'].rolling(window=5).agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('trend', lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0)
        ]).fillna(0)
        
        features.update({
            'rolling_mean': rolling_stats['mean'].iloc[-1],
            'rolling_std': rolling_stats['std'].iloc[-1],
            'rolling_trend': rolling_stats['trend'].iloc[-1]
        })
        
        # Rest and travel
        features['rest_impact'] = features['rest_days'] * 0.1
        features['travel_impact'] = min(features['travel_distance'] / 1000, 1.0)
        
        team_features.append({
            'team_id': team_id,
            'game_id': team_games['game_id'].iloc[-1],
            **features
        })
    
    return pd.DataFrame(team_features)

# Example usage:
# games = pd.DataFrame({...})  # Your game data
# 
# # Create features
# features_df = create_feature_engineering_pipeline(games)
# 
# # Prepare training data
# X = features_df.drop(columns=['team_id', 'game_id'])
# y = (games['points'] > games['opponent_points']).astype(int).values
# 
# # Split data
# from sklearn.model_selection import train_test_split
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# 
# # Create and train models
# models = create_ensemble_models(X_train.values, y_train)
# 
# # Optimize weights
# optimized_weights = optimize_ensemble_weights(models, X_val.values, y_val)
# 
# # Evaluate performance
# report = generate_ensemble_report(models, optimized_weights, X_test.values, y_test)
# 
# print(f"Optimized weights: {optimized_weights}")
# print(f"Brier score improvement: {report['improvement']['percentage_improvement']:.2f}%")
# print(f"Model diversity (avg correlation): {report['model_diversity']['average_correlation']:.3f}")

