import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
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

def select_important_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Select important features using Random Forest.

    Parameters:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target labels

    Returns:
    pd.DataFrame: Selected feature subset
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    feature_importances = rf.feature_importances_
    feature_names = X.columns

    # Select top features
    threshold = 0.05
    important_features = [feature for feature, importance in zip(feature_names, feature_importances) if importance > threshold]

    return X[important_features]

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

    # Create feature set
    feature_set = create_feature_set(games)

    # Select important features
    X = feature_set.drop(['actual'], axis=1)
    y = feature_set['actual']
    selected_features = select_important_features(X, y)

    # Evaluate model performance
    brier = evaluate_model_performance(selected_features, y)

    print(f"Selected features: {selected_features.columns.tolist()}")
    print(f"Brier score: {brier:.4f}")

if __name__ == "__main__":
    main()

