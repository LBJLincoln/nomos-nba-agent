import numpy as np
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier

def analyze_feature_importance(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Analyze feature importance using Random Forest"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    return {feature_names[i]: importances[i] for i in range(len(feature_names))}

def rank_features_by_importance(feature_importance: Dict[str, float]) -> List[Tuple[str, float]]:
    """Rank features by importance"""
    return sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

def get_top_features(feature_importance: Dict[str, float], n: int = 10) -> List[str]:
    """Get top N features by importance"""
    ranked = rank_features_by_importance(feature_importance)
    return [feature[0] for feature in ranked[:n]]
