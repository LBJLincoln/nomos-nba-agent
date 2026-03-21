import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

def compute_feature_correlations(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix for selected features.
    
    Parameters:
    df (pd.DataFrame): Game data
    feature_columns (list): List of feature column names
    
    Returns:
    pd.DataFrame: Correlation matrix
    """
    if len(feature_columns) == 0:
        print("No features to analyze")
        return pd.DataFrame()
    
    # Select only numeric features
    numeric_features = df[feature_columns].select_dtypes(include=[np.number])
    
    # Compute correlation matrix
    corr_matrix = numeric_features.corr()
    
    return corr_matrix

def identify_redundant_features(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> dict:
    """
    Identify redundant features based on correlation threshold.
    
    Parameters:
    corr_matrix (pd.DataFrame): Correlation matrix
    threshold (float): Correlation threshold for redundancy
    
    Returns:
    dict: Dictionary with redundant features grouped by correlation clusters
    """
    # Create absolute correlation matrix
    abs_corr = corr_matrix.abs()
    
    # Find highly correlated pairs
    highly_corr = (abs_corr > threshold) & (abs_corr < 1.0)
    
    # Group correlated features
    feature_groups = {}
    processed = set()
    
    for feature in corr_matrix.columns:
        if feature in processed:
            continue
        
        # Find all features correlated with this one
        correlated = highly_corr[feature][highly_corr[feature]].index.tolist()
        correlated.append(feature)
        
        # Add to groups
        group_key = tuple(sorted(correlated))
        feature_groups[group_key] = correlated
        
        # Mark as processed
        processed.update(correlated)
    
    return feature_groups

def visualize_correlation_matrix(corr_matrix: pd.DataFrame, output_path: str = 'correlation_matrix.png'):
    """
    Visualize correlation matrix as heatmap.
    
    Parameters:
    corr_matrix (pd.DataFrame): Correlation matrix
    output_path (str): Path to save heatmap image
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation matrix visualization saved to {output_path}")

def generate_redundancy_report(feature_groups: dict, corr_matrix: pd.DataFrame, output_path: str = 'redundant_features.txt'):
    """
    Generate text report of redundant features.
    
    Parameters:
    feature_groups (dict): Dictionary of correlated feature groups
    corr_matrix (pd.DataFrame): Correlation matrix
    output_path (str): Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("REDUNDANT FEATURE ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        if not feature_groups:
            f.write("No redundant features found (all correlations below threshold)\n")
            return
        
        f.write(f"Threshold: |correlation| > 0.8\n\n")
        f.write("FEATURE GROUPS (HIGHLY CORRELATED):\n")
        f.write("-"*40 + "\n")
        
        for group_idx, (group_key, features) in enumerate(feature_groups.items(), 1):
            f.write(f"\nGroup {group_idx}:\n")
            f.write(f"  Features: {', '.join(features)}\n")
            
            # Show correlation values within group
            for i, feat1 in enumerate(features):
                for j, feat2 in enumerate(features):
                    if i < j:
                        corr_val = corr_matrix.loc[feat1, feat2]
                        f.write(f"    {feat1} ↔ {feat2}: {corr_val:.3f}\n")
            
            # Suggest which feature to keep (highest variance)
            variances = {feat: corr_matrix.loc[feat, feat] for feat in features}
            best_feature = max(variances.items(), key=lambda x: x[1])[0]
            f.write(f"  Recommended to keep: {best_feature}\n")
        
        f.write("\n\nRECOMMENDATIONS:\n")
        f.write("-"*40 + "\n")
        f.write("Consider removing all but one feature from each group to reduce redundancy\n")
        f.write("Features with highest variance within each group are typically most informative\n")
        f.write("After removal, retrain models to verify performance is maintained or improved\n")

def main():
    print("NBA Feature Correlation Analysis")
    print("="*40)
    
    # Load data
    df = load_game_data()
    
    if df.empty:
        print("No data available. Exiting.")
        return
    
    # Define feature columns (example - adjust based on actual features)
    feature_columns = [
        'weighted_win_streak', 'margin_trend', 'avg_margin', 'margin_volatility',
        'rest_quality_score', 'travel_distance', 'timezone_adjustment',
        'opponent_strength', 'pace_adjusted_points', 'home_advantage',
        'back_to_back_penalty', 'recent_performance', 'season_performance'
    ]
    
    # Filter to available features
    available_features = [col for col in feature_columns if col in df.columns]
    
    if not available_features:
        print("No matching features found in data")
        return
    
    print(f"Analyzing {len(available_features)} features:")
    print(f"  {', '.join(available_features)}")
    
    # Compute correlations
    print("\nComputing correlation matrix...")
    corr_matrix = compute_feature_correlations(df, available_features)
    
    if corr_matrix.empty:
        print("No correlations to compute")
        return
    
    # Identify redundant features
    print("Identifying redundant features...")
    redundant_groups = identify_redundant_features(corr_matrix, threshold=0.8)
    
    # Visualize
    print("Generating correlation visualization...")
    visualize_correlation_matrix(corr_matrix)
    
    # Generate report
    print("Generating redundancy report...")
    generate_redundancy_report(redundant_groups, corr_matrix)
    
    print("\nAnalysis complete!")
    print(f"Correlation matrix: correlation_matrix.png")
    print(f"Redundancy report: redundant_features.txt")

if __name__ == "__main__":
    main()
