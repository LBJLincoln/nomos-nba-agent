import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Tuple, List, Generator

class TemporalCrossValidator:
    """
    Cross-validation splitter with temporal awareness for time-series data.
    Prevents data leakage by ensuring future games never appear in training set.
    """
    
    def __init__(self, n_splits: int = 5, min_train_size: int = 100):
        """
        Initialize temporal cross-validator.
        
        Parameters:
        n_splits (int): Number of cross-validation folds
        min_train_size (int): Minimum number of training samples required
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.kf = KFold(n_splits=n_splits, shuffle=False)
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate temporal splits.
        
        Parameters:
        X (pd.DataFrame): Feature matrix with datetime index
        y (pd.Series): Target labels (optional)
        groups (np.ndarray): Group labels (optional)
        
        Returns:
        Generator: Yields (train_idx, test_idx) tuples
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a datetime index for temporal splitting")
        
        n_samples = len(X)
        min_train_required = self.min_train_size + (n_samples - self.min_train_size) // self.n_splits
        
        if n_samples < self.min_train_size * 2:
            raise ValueError(f"Need at least {self.min_train_size * 2} samples for temporal CV")
        
        # Generate splits
        for train_idx, test_idx in self.kf.split(X):
            # Ensure temporal order is preserved
            if train_idx[-1] >= test_idx[0]:
                raise ValueError("Temporal split violation: training data contains future samples")
            
            # Check minimum training size
            if len(train_idx) < self.min_train_size:
                continue
            
            yield train_idx, test_idx
    
    def temporal_train_test_split(self, X: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a single temporal train-test split.
        
        Parameters:
        X (pd.DataFrame): Feature matrix with datetime index
        test_size (float): Proportion of data to use for testing (0.0-1.0)
        
        Returns:
        Tuple: (train_data, test_data)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a datetime index for temporal splitting")
        
        n_samples = len(X)
        test_samples = int(n_samples * test_size)
        
        if test_samples < 10:
            raise ValueError("Test set too small - increase test_size or provide more data")
        
        train_end_idx = n_samples - test_samples
        train_data = X.iloc[:train_end_idx]
        test_data = X.iloc[train_end_idx:]
        
        return train_data, test_data
    
    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series, model, metric) -> dict:
        """
        Perform walk-forward validation with temporal splits.
        
        Parameters:
        X (pd.DataFrame): Feature matrix with datetime index
        y (pd.Series): Target labels
        model (object): Model with fit() and predict() methods
        metric (function): Metric function(y_true, y_pred) -> float
        
        Returns:
        dict: Validation results with metrics and predictions
        """
        results = {
            'train_sizes': [],
            'test_sizes': [],
            'metrics': [],
            'predictions': [],
            'true_labels': []
        }
        
        # Create expanding window
        n_samples = len(X)
        train_size = max(self.min_train_size, int(n_samples * 0.5))
        
        for i in range(train_size, n_samples, max(1, (n_samples - train_size) // 10)):
            X_train, X_test = X.iloc[:i], X.iloc[i:i+1]
            y_train, y_test = y.iloc[:i], y.iloc[i:i+1]
            
            if len(X_test) == 0:
                continue
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metric
            metric_value = metric(y_test, y_pred)
            
            # Store results
            results['train_sizes'].append(len(X_train))
            results['test_sizes'].append(len(X_test))
            results['metrics'].append(metric_value)
            results['predictions'].append(y_pred[0] if hasattr(y_pred, '__len__') and len(y_pred) == 1 else y_pred)
            results['true_labels'].append(y_test.values[0] if hasattr(y_test, '__len__') and len(y_test) == 1 else y_test.values)
        
        return results
    
    def create_validation_schedule(self, X: pd.DataFrame, validation_frequency: str = 'monthly') -> List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """
        Create a validation schedule for periodic model evaluation.
        
        Parameters:
        X (pd.DataFrame): Feature matrix with datetime index
        validation_frequency (str): Frequency for validation ('monthly', 'weekly', 'quarterly')
        
        Returns:
        List: List of (X_train, y_train, X_val, y_val) tuples
        """
        schedule = []
        X = X.copy()
        
        if validation_frequency == 'monthly':
            periods = pd.date_range(start=X.index.min(), end=X.index.max(), freq='M')
        elif validation_frequency == 'weekly':
            periods = pd.date_range(start=X.index.min(), end=X.index.max(), freq='W')
        elif validation_frequency == 'quarterly':
            periods = pd.date_range(start=X.index.min(), end=X.index.max(), freq='Q')
        else:
            raise ValueError("Frequency must be 'monthly', 'weekly', or 'quarterly'")
        
        for period_end in periods:
            # Training data: all data before this period
            X_train = X[X.index < period_end]
            
            # Validation data: data in this period
            X_val = X[(X.index >= period_end) & (X.index < period_end + pd.DateOffset(months=1))]
            
            if len(X_train) < self.min_train_size or len(X_val) < 10:
                continue
            
            # Assuming y is available in X as a column
            y_train = X_train.pop('target')
            y_val = X_val.pop('target')
            
            schedule.append((X_train, y_train, X_val, y_val))
        
        return schedule

def temporal_correlation_check(X: pd.DataFrame, y: pd.Series, max_lag: int = 10) -> pd.DataFrame:
    """
    Check for temporal correlations that might indicate data leakage.
    
    Parameters:
    X (pd.DataFrame): Feature matrix with datetime index
    y (pd.Series): Target labels
    max_lag (int): Maximum lag to check
    
    Returns:
    pd.DataFrame: Correlation values at different lags
    """
    correlations = []
    
    for lag in range(1, max_lag + 1):
        X_shifted = X.shift(lag).dropna()
        y_shifted = y[X_shifted.index]
        
        if len(X_shifted) < 10:
            continue
        
        corr_matrix = X_shifted.corrwith(y_shifted)
        correlations.append({
            'lag': lag,
            'correlation': corr_matrix.abs().max(),
            'features': corr_matrix.abs().idxmax()
        })
    
    return pd.DataFrame(correlations)

# Example usage:
# X = pd.DataFrame({
#     'feature1': np.random.randn(1000),
#     'feature2': np.random.randn(1000)
# }, index=pd.date_range('2020-01-01', periods=1000, freq='D'))
# 
# y = (np.random.randn(1000) > 0).astype(int)
# 
# tcv = TemporalCrossValidator(n_splits=5, min_train_size=200)
# 
# # Generate splits
# for train_idx, test_idx in tcv.split(X):
#     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
# 
# # Walk-forward validation
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# results = tcv.walk_forward_validation(X, y, model, metric=brier_score_loss)
# 
# # Create validation schedule
# schedule = tcv.create_validation_schedule(X, validation_frequency='monthly')
