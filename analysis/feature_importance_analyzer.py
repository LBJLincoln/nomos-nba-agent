import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_game_data(data_path: str = 'data/games.csv') -> pd.DataFrame:
    """
    Load game data from CSV file.

    Parameters:
    data_path (str): Path to game data CSV file

    Returns:
    pd.DataFrame: Loaded game data
    """
    data_file = Path(data_path)

    if not data_file.exists():
        print(f"Error: Data file not found at {data_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} games from {data_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def load_predictions(predictions_path: str = 'predictions.csv') -> pd.DataFrame:
    """
    Load prediction results from CSV file.

    Parameters:
    predictions_path (str): Path to predictions CSV file

    Returns:
    pd.DataFrame: Loaded prediction data
    """
    predictions_file = Path(predictions_path)

    if not predictions_file.exists():
        print(f"Error: Predictions file not found at {predictions_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(predictions_path)
        print(f"Loaded {len(df)} predictions from {predictions_path}")
        return df
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return pd.DataFrame()

def analyze_feature_importance(X: pd.DataFrame, y: pd.Series, feature_names: list, 
                               n_estimators: int = 100, random_state: int = 42) -> dict:
    """
    Analyze feature importance using Random Forest and permutation importance.
    
    Parameters:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target labels
    feature_names (list): List of feature names
    n_estimators (int): Number of trees in Random Forest
    random_state (int): Random seed for reproducibility
    
    Returns:
    dict: Dictionary containing feature importance metrics
    """
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Calculate permutation importance
    result = permutation_importance(
        rf, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
    )
    permutation_importances = result.importances_mean
    
    # Combine both metrics
    combined_importance = importances + permutation_importances
    
    # Create feature importance report
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'random_forest_importance': importances,
        'permutation_importance': permutation_importances,
        'combined_importance': combined_importance
    })
    
    # Sort by combined importance
    feature_importance_df = feature_importance_df.sort_values('combined_importance', ascending=False).reset_index(drop=True)
    
    # Calculate performance metrics
    brier_score = np.mean((rf.predict_proba(X_test)[:, 1] - y_test) ** 2)
    accuracy = rf.score(X_test, y_test)
    
    return {
        'feature_importance_df': feature_importance_df,
        'model': rf,
        'test_metrics': {
            'brier_score': brier_score,
            'accuracy': accuracy
        },
        'permutation_importances': result
    }

def visualize_feature_importance(importance_data: dict, output_path: str = 'feature_importance.png'):
    """
    Create comprehensive feature importance visualization.
    
    Parameters:
    importance_data (dict): Feature importance analysis results
    output_path (str): Path to save visualization
    """
    feature_importance_df = importance_data['feature_importance_df']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Combined importance bar plot
    top_features = feature_importance_df.head(15)
    axes[0, 0].barh(top_features['feature'], top_features['combined_importance'])
    axes[0, 0].set_title('Top 15 Features by Combined Importance')
    axes[0, 0].set_xlabel('Importance Score')
    axes[0, 0].set_ylabel('Feature')
    axes[0, 0].invert_yaxis()
    
    # Random Forest importance
    axes[0, 1].barh(top_features['feature'], top_features['random_forest_importance'])
    axes[0, 1].set_title('Random Forest Feature Importance')
    axes[0, 1].set_xlabel('Importance Score')
    axes[0, 1].invert_yaxis()
    
    # Permutation importance
    axes[1, 0].barh(top_features['feature'], top_features['permutation_importance'])
    axes[1, 0].set_title('Permutation Importance')
    axes[1, 0].set_xlabel('Importance Score')
    axes[1, 0].invert_yaxis()
    
    # Feature importance correlation
    corr_matrix = feature_importance_df[['random_forest_importance', 'permutation_importance']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('Feature Importance Correlation')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance visualization saved to {output_path}")

def generate_importance_report(importance_data: dict, output_path: str = 'feature_importance_report.txt'):
    """
    Generate text report of feature importance analysis.
    
    Parameters:
    importance_data (dict): Feature importance analysis results
    output_path (str): Path to save report
    """
    feature_importance_df = importance_data['feature_importance_df']
    test_metrics = importance_data['test_metrics']
    
    with open(output_path, 'w') as f:
        f.write("FEATURE IMPORTANCE ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Test Set Performance:\n")
        f.write(f"  Brier Score: {test_metrics['brier_score']:.4f}\n")
        f.write(f"  Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Total Features Analyzed: {len(feature_importance_df)}\n\n")
        
        f.write("TOP 20 FEATURES BY COMBINED IMPORTANCE:\n")
        f.write("-"*40 + "\n")
        
        for idx, row in feature_importance_df.head(20).iterrows():
            f.write(f"{row['feature']:40} | RF: {row['random_forest_importance']:6.4f} | "
                    f"Perm: {row['permutation_importance']:6.4f} | Combined: {row['combined_importance']:6.4f}\n")
        
        f.write("\n\nFEATURE CATEGORIES:\n")
        f.write("-"*40 + "\n")
        
        # Categorize features (example categories - adjust based on actual features)
        momentum_features = [col for col in feature_importance_df['feature'] if 'momentum' in col or 'streak' in col]
        rest_features = [col for col in feature_importance_df['feature'] if 'rest' in col or 'travel' in col]
        strength_features = [col for col in feature_importance_df['feature'] if 'strength' in col or 'opponent' in col]
        pace_features = [col for col in feature_importance_df['feature'] if 'pace' in col or 'points' in col]
        
        if momentum_features:
            f.write(f"\nMomentum Features ({len(momentum_features)}):\n")
            for feat in momentum_features[:5]:
                row = feature_importance_df[feature_importance_df['feature'] == feat].iloc[0]
                f.write(f"  {feat:30} - {row['combined_importance']:6.4f}\n")
        
        if rest_features:
            f.write(f"\nRest/Travel Features ({len(rest_features)}):\n")
            for feat in rest_features[:5]:
                row = feature_importance_df[feature_importance_df['feature'] == feat].iloc[0]
                f.write(f"  {feat:30} - {row['combined_importance']:6.4f}\n")
        
        if strength_features:
            f.write(f"\nOpponent Strength Features ({len(strength_features)}):\n")
            for feat in strength_features[:5]:
                row = feature_importance_df[feature_importance_df['feature'] == feat].iloc[0]
                f.write(f"  {feat:30} - {row['combined_importance']:6.4f}\n")
        
        if pace_features:
            f.write(f"\nPace/Scoring Features ({len(pace_features)}):\n")
            for feat in pace_features[:5]:
                row = feature_importance_df[feature_importance_df['feature'] == feat].iloc[0]
                f.write(f"  {feat:30} - {row['combined_importance']:6.4f}\n")
        
        f.write("\n\nRECOMMENDATIONS:\n")
        f.write("-"*40 + "\n")
        f.write("1. Focus feature engineering on top 10-15 features\n")
        f.write("2. Consider removing features with very low importance (<0.01)\n")
        f.write("3. Investigate interactions between top features\n")
        f.write("4. Validate importance stability across different time periods\n")
        f.write("5. Monitor for feature drift over season\n")

def main():
    print("NBA Feature Importance Analysis")
    print("="*40)
    
    # Load data
    games = load_game_data()
    predictions = load_predictions()
    
    if games.empty or predictions.empty:
        print("Error: Could not load required data files")
        return
    
    # Merge data to get target labels
    merged = pd.merge(predictions, games, on=['game_id', 'team_id'], how='left')
    merged['target'] = (merged['actual'] > 0.5).astype(int)
    
    # Select features (adjust based on actual available features)
    feature_columns = [
        'weighted_win_streak', 'margin_trend', 'avg_margin', 'margin_volatility',
        'rest_quality_score', 'travel_distance', 'timezone_adjustment',
        'opponent_strength', 'pace_adjusted_points', 'home_advantage',
        'back_to_back_penalty', 'recent_performance', 'season_performance'
    ]
    
    # Filter to available features
    available_features = [col for col in feature_columns if col in merged.columns]
    print(f"Analyzing {len(available_features)} features")
    
    if len(available_features) < 5:
        print("Not enough features for analysis")
        return
    
    X = merged[available_features]
    y = merged['target']
    
    # Analyze feature importance
    print("Analyzing feature importance...")
    importance_data = analyze_feature_importance(X, y, available_features)
    
    # Generate visualizations
    print("Generating visualizations...")
    visualize_feature_importance(importance_data)
    
    # Generate report
    print("Generating report...")
    generate_importance_report(importance_data)
    
    print("\nFeature importance analysis complete!")
    print(f"Visualization: feature_importance.png")
    print(f"Report: feature_importance_report.txt")

if __name__ == "__main__":
    main()
