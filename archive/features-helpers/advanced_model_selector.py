import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss
from typing import Dict, List

def load_game_data(data_path: str = 'data/games.csv') -> pd.DataFrame:
    """
    Load game data from CSV file.

    Parameters:
    data_path (str): Path to game data CSV file

    Returns:
    pd.DataFrame: Loaded game data
    """
    data_file = pd.read_csv(data_path)
    return data_file

def create_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive feature set for model training.

    Parameters:
    df (pd.DataFrame): Game data

    Returns:
    pd.DataFrame: Feature set
    """
    # Basic features
    df['margin'] = df['points'] - df['opponent_points']
    df['pace_adjusted_points'] = df['points'] / (df['pace'] / 100)
    df['home_advantage'] = np.where(df['location'] == 'home', 1, 0)

    # Advanced features
    df['rest_quality_score'] = np.where(df['rest_days'] < 1, 0.5, np.where(df['rest_days'] < 2, 0.7, 0.9))
    df['travel_distance'] = df['travel_distance'].fillna(0)
    df['timezone_adjustment'] = np.where(df['timezone'] == df['opponent_timezone'], 0, 1)

    # Feature engineering
    df['weighted_win_streak'] = df['win_streak'] * df['rest_quality_score']
    df['margin_trend'] = df['margin'].rolling(window=5).mean()

    return df

def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest') -> Dict[str, any]:
    """
    Train a model based on the given type.

    Parameters:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target labels
    model_type (str): Type of model to train (random_forest, gradient_boost, logistic_regression)

    Returns:
    Dict[str, any]: Trained model and its performance metrics
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boost':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42)
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)

    return {
        'model': model,
        'brier_score': brier,
        'log_loss': logloss
    }

def select_best_model(df: pd.DataFrame) -> Dict[str, any]:
    """
    Select the best model based on Brier score.

    Parameters:
    df (pd.DataFrame): Game data

    Returns:
    Dict[str, any]: Best model and its performance metrics
    """
    feature_set = create_feature_set(df)
    X = feature_set.drop(['actual'], axis=1)
    y = feature_set['actual']

    models = {
        'random_forest': train_model(X, y, model_type='random_forest'),
        'gradient_boost': train_model(X, y, model_type='gradient_boost'),
        'logistic_regression': train_model(X, y, model_type='logistic_regression')
    }

    best_model = min(models, key=lambda x: models[x]['brier_score'])

    return models[best_model]

def main():
    # Load data
    games = load_game_data()

    if games.empty:
        print("No game data available. Exiting.")
        return

    # Select best model
    best_model = select_best_model(games)

    print(f"Best model: {list(best_model.keys())[0]}")
    print(f"Brier score: {best_model['brier_score']:.4f}")
    print(f"Log loss: {best_model['log_loss']:.4f}")

if __name__ == "__main__":
    main()

