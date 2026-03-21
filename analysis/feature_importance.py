import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

def analyze_feature_importance(X, y, feature_names):
    """
    Analyze feature importance using Random Forest and permutation importance
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Permutation importance
    result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
    
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
        'combined_importances': combined_importance[sorted_idx]
    }
    
    return report

def print_importance_report(report):
    print("\nFEATURE IMPORTANCE REPORT")
    print("="*60)
    for i, feature in enumerate(report['feature_names']):
        print(f"{feature:30} | Importance: {report['combined_importances'][i]:.4f} | Perm: {report['permutation_importances'][i]:.4f}")
    
    print("\nTOP 5 FEATURES:")
    for i in range(5):
        feature = report['feature_names'][i]
        importance = report['combined_importances'][i]
        print(f"  {i+1}. {feature:30} - {importance:.4f}")
