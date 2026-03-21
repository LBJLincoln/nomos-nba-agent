import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class DynamicFeatureWeighting:
    """
    Learns optimal feature weights based on historical prediction performance
    to improve model calibration and reduce Brier score.
    """
    
    def __init__(self, alpha=1.0):
        self.scaler = StandardScaler()
        self.weight_model = Ridge(alpha=alpha, fit_intercept=False)
        self.feature_importance = None
        
    def fit(self, X, y, predictions):
        """
        Fit feature weights based on prediction errors
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: True outcomes (0/1 for home win)
            predictions: Model predictions (probabilities)
        """
        # Calculate prediction errors
        errors = y - predictions
        
        # Scale features for stability
        X_scaled = self.scaler.fit_transform(X)
        
        # Learn weights that minimize squared error of predictions
        self.weight_model.fit(X_scaled, errors)
        
        # Store feature importance
        self.feature_importance = np.abs(self.weight_model.coef_)
        self.feature_importance /= np.sum(self.feature_importance)
        
    def transform(self, X):
        """
        Apply learned weights to features
        
        Args:
            X: Feature matrix
            
        Returns:
            Weighted feature matrix
        """
        if self.feature_importance is None:
            return X
        
        X_scaled = self.scaler.transform(X)
        return X_scaled * self.weight_model.coef_
    
    def get_top_features(self, n=10):
        """
        Get top weighted features
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature_index, importance) tuples
        """
        if self.feature_importance is None:
            return []
        
        indices = np.argsort(self.feature_importance)[::-1]
        return [(i, self.feature_importance[i]) for i in indices[:n]]
    
    def reset(self):
        """Reset the weighting model"""
        self.feature_importance = None
        self.weight_model = Ridge(alpha=self.weight_model.alpha, fit_intercept=False)
        self.scaler = StandardScaler()
