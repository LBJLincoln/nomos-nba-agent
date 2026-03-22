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

def train_quick_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a quick model for improvement.

    Parameters:
    df (pd.DataFrame): Feature set

    Returns:
    RandomForestClassifier: Trained model
    """
    # Define feature and target columns
    feature_columns = ['weighted_win_streak', 'margin_trend', 'avg_margin', 'margin_volatility',
                        'rest_quality_score', 'travel_distance', 'timezone_adjustment',
                        'opponent_strength', 'pace_adjusted_points', 'home_advantage',
                        'back_to_back_penalty', 'recent_performance', 'season_performance']
    target_column = 'actual'

    X = df[feature_columns]
    y = df[target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def evaluate_model_performance(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Evaluate model performance using Brier score.

    Parameters:
    model (RandomForestClassifier): Trained model
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target labels

    Returns:
    float: Brier score
    """
    y_pred = model.predict_proba(X)[:, 1]
    brier = brier_score_loss(y, y_pred)

    return brier

def main():
    # Load data
    games = load_game_data()

    if games.empty:
        print("No game data available. Exiting.")
        return

    # Create feature set
    feature_set = create_feature_set(games)

    # Train quick model
    model = train_quick_model(feature_set)

    # Evaluate model performance
    brier = evaluate_model_performance(model, feature_set.drop(['actual'], axis=1), feature_set['actual'])

    print(f"Quick model Brier score: {brier:.4f}")

if __name__ == "__main__":
    main()

