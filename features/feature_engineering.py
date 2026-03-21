import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

def select_top_features(X, y, k=20):
    """Select top k features based on ANOVA F-test"""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    return X_new, selected_indices

def create_interaction_features(X, degree=2):
    """Create polynomial interaction features"""
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_interactions = poly.fit_transform(X)
    return X_interactions

def normalize_features(X):
    """Standardize features to zero mean and unit variance"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def engineer_features(X, y, k=20, degree=2):
    """Complete feature engineering pipeline"""
    # Select top features
    X_selected, selected_indices = select_top_features(X, y, k=k)
    
    # Create interactions
    X_interactions = create_interaction_features(X_selected, degree=degree)
    
    # Normalize
    X_engineered = normalize_features(X_interactions)
    
    return X_engineered, selected_indices
