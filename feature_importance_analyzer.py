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

def prepare_features(df: pd.DataFrame, target_col: str = 'actual') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target vector.
    """
    # Select numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col != target_col]
    
    # Handle missing values
    X = df[numeric_features].fillna(df[numeric_features].mean())
    y = df[target_col]
    
    return X, y

def analyze_feature_importance(X: pd.DataFrame, y: pd.Series, feature_names: list) -> dict:
    """
    Analyze feature importance using Random Forest and permutation importance.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    
    # Permutation importance
    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    
    # Feature importance
    importances = rf.feature_importances_
    
    # Combine both metrics
    combined_importance = importances + result.importances_mean
    
    # Sort features
    sorted_idx = np.argsort(combined_importance)[::-1]
    
    report = {
        'feature_names': [feature_names[i] for i in sorted_idx],
        'importances': importances[sorted_idx],
        'permutation_importances': result.importances_mean[sorted_idx],
        'combined_importances': combined_importance[sorted_idx],
        'model_score': rf.score(X_test, y_test)
    }
    
    return report

def visualize_feature_importance(report: dict, output_path: str = 'feature_importance.png'):
    """
    Visualize feature importance as bar plot.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    features = report['feature_names'][:20]
    importances = report['combined_importances'][:20]
    
    ax.barh(features, importances, color='steelblue')
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 20 Feature Importances')
    ax.set_xlim(0, max(importances) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance visualization saved to {output_path}")

def generate_importance_report(report: dict, output_path: str = 'feature_importance_report.txt'):
    """
    Generate text report of feature importance analysis.
    """
    with open(output_path, 'w') as f:
        f.write("FEATURE IMPORTANCE ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Model Test Accuracy: {report['model_score']:.4f}\n\n")
        
        f.write("TOP 20 FEATURES BY IMPORTANCE:\n")
        f.write("-"*40 + "\n")
        for i, feature in enumerate(report['feature_names'][:20]):
            importance = report['combined_importances'][i]
            perm_importance = report['permutation_importances'][i]
            f.write(f"{i+1:2d}. {feature:30} | Importance: {importance:6.4f} | Perm: {perm_importance:6.4f}\n")
        
        f.write("\n\nRECOMMENDATIONS:\n")
        f.write("-"*40 + "\n")
        f.write("1. Top 5 features are most predictive - consider focusing feature engineering here\n")
        f.write("2. Features with high permutation importance but low RF importance may be redundant\n")
        f.write("3. Consider removing bottom 20% of features to reduce model complexity\n")
        f.write("4. Retrain model with top features to verify performance is maintained\n")

def main():
    print("NBA Feature Importance Analyzer")
    print("="*40)
    
    # Load data
    df = load_game_data()
    
    if df.empty:
        print("No data available. Exiting.")
        return
    
    # Prepare features
    X, y = prepare_features(df)
    
    if X.empty or y.empty:
        print("No valid features found. Exiting.")
        return
    
    print(f"Analyzing {X.shape[1]} features...")
    
    # Analyze importance
    print("Computing feature importance...")
    report = analyze_feature_importance(X, y, X.columns.tolist())
    
    # Visualize
    print("Generating visualization...")
    visualize_feature_importance(report)
    
    # Generate report
    print("Generating text report...")
    generate_importance_report(report)
    
    print("\nAnalysis complete!")
    print(f"Visualization: feature_importance.png")
    print(f"Report: feature_importance_report.txt")

if __name__ == "__main__":
    main()
