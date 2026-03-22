import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

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

def compute_feature_importances(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute feature importances using Random Forest.

    Parameters:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target labels

    Returns:
    pd.DataFrame: Feature importance dataframe
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    })

    return feature_importance_df

def optimize_feature_set(X: pd.DataFrame, y: pd.Series, threshold: float = 0.01) -> pd.DataFrame:
    """
    Optimize feature set by removing low-importance features.

    Parameters:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target labels
    threshold (float): Importance threshold for feature removal

    Returns:
    pd.DataFrame: Optimized feature matrix
    """
    feature_importance_df = compute_feature_importances(X, y)
    important_features = feature_importance_df[feature_importance_df['importance'] > threshold]['feature']

    optimized_X = X[important_features]

    return optimized_X

def evaluate_model_performance(X: pd.DataFrame, y: pd.Series) -> float:
    """
    Evaluate model performance using Brier score.

    Parameters:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target labels

    Returns:
    float: Brier score
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_pred)

    return brier

def main():
    # Load data
    games = load_game_data()

    if games.empty:
        print("No game data available. Exiting.")
        return

    # Define feature and target columns
    feature_columns = ['weighted_win_streak', 'margin_trend', 'avg_margin', 'margin_volatility',
                        'rest_quality_score', 'travel_distance', 'timezone_adjustment',
                        'opponent_strength', 'pace_adjusted_points', 'home_advantage',
                        'back_to_back_penalty', 'recent_performance', 'season_performance']
    target_column = 'actual'

    X = games[feature_columns]
    y = games[target_column]

    # Optimize feature set
    optimized_X = optimize_feature_set(X, y)

    # Evaluate model performance
    brier = evaluate_model_performance(optimized_X, y)

    print(f"Optimized Brier score: {brier:.4f}")

if __name__ == "__main__":
    main()
