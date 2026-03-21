import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_selection import SelectKBest, f_classif

def select_best_features(X: np.ndarray, y: np.ndarray, k: int = 20) -> Tuple[np.ndarray, List[int]]:
    """Select best K features using ANOVA F-test"""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    return X_new, selected_indices

def filter_features_by_variance(X: np.ndarray, threshold: float = 0.01) -> Tuple[np.ndarray, List[int]]:
    """Filter features by variance threshold"""
    variances = np.var(X, axis=0)
    selected_indices = np.where(variances > threshold)[0]
    return X[:, selected_indices], selected_indices.tolist()
