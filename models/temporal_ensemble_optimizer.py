import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, brier_score_loss
from typing import List, Tuple, Callable
from datetime import datetime

class TemporalEnsembleOptimizer:
    """
    Advanced ensemble optimizer with temporal awareness and custom loss functions.
    Combines ensemble weight optimization with temporal validation to prevent data leakage.
    """
    
    def __init__(self, n_models: int, temporal_window: int = 30, 
                 initial_weights: List[float] = None):
        """
        Initialize temporal ensemble optimizer.
        
        Parameters:
        n_models (int): Number of models in ensemble
        temporal_window (int): Temporal window size for validation (in days)
        initial_weights (List[float]): Initial weights for models
        """
        self.n_models = n_models
        self.temporal_window = temporal_window
        
        if initial_weights is None:
            self.weights = np.ones(n_models) / n_models
        else:
            self.weights = np.array(initial_weights) / np.sum(initial_weights)
    
    def custom_temporal_loss(self, predictions: np.ndarray, true_labels: np.ndarray,
                            dates: np.ndarray, weights: np.ndarray) -> float:
        """
        Custom loss function that combines temporal awareness with ensemble weighting.
        
        Parameters:
        predictions (np.ndarray): Model predictions (n_samples, n_models)
        true_labels (np.ndarray): True binary labels
        dates (np.ndarray): Dates for temporal ordering
        weights (np.ndarray): Ensemble weights to optimize
        
        Returns:
        float: Combined loss value
        """
        # Ensemble prediction
        ensemble_pred = np.dot(predictions, weights)
        
        # Temporal decay factor (more recent games weighted higher)
        date_diffs = np.array([(datetime.now() - datetime.strptime(d, '%Y-%m-%d')).days 
                              for d in dates])
        temporal_weights = np.exp(-date_diffs / self.temporal_window)
        
        # Weighted Brier score
        brier = np.mean(temporal_weights * (ensemble_pred - true_labels) ** 2)
        
        # Add regularization to prevent extreme weights
        weight_penalty = 0.1 * np.sum((weights - 0.5) ** 2)
        
        # Add calibration penalty (penalize overconfident wrong predictions)
        confidence = np.abs(ensemble_pred - 0.5)
        calibration_penalty = 0.05 * np.mean((ensemble_pred - true_labels) ** 2 * (1 - confidence))
        
        return brier + weight_penalty + calibration_penalty
    
    def optimize_weights(self, predictions: np.ndarray, true_labels: np.ndarray,
                        dates: np.ndarray, max_iter: int = 1000) -> np.ndarray:
        """
        Optimize ensemble weights using temporal-aware custom loss.
        
        Parameters:
        predictions (np.ndarray): Model predictions (n_samples, n_models)
        true_labels (np.ndarray): True binary labels
        dates (np.ndarray): Dates for temporal ordering
        max_iter (int): Maximum optimization iterations
        
        Returns:
        np.ndarray: Optimized weights
        """
        # Define objective function
        def objective(weights):
            return self.custom_temporal_loss(predictions, true_labels, dates, weights)
        
        # Bounds for weights (0 to 1)
        bounds = [(0.01, 0.99) for _ in range(self.n_models)]
        
        # Constraints: weights must sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Initial weights
        initial_weights = self.weights.copy()
        
        # Perform optimization
        result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints,
                         method='SLSQP', options={'maxiter': max_iter, 'ftol': 1e-6})
        
        # Return optimized weights
        optimized_weights = result.x / result.x.sum()
        
        return optimized_weights
    
    def temporal_cross_validation(self, predictions: np.ndarray, true_labels: np.ndarray,
                                 dates: np.ndarray, n_splits: int = 5) -> Tuple[np.ndarray, float]:
        """
        Perform temporal cross-validation for robust weight optimization.
        
        Parameters:
        predictions (np.ndarray): Model predictions (n_samples, n_models)
        true_labels (np.ndarray): True binary labels
        dates (np.ndarray): Dates for temporal ordering
        n_splits (int): Number of cross-validation folds
        
        Returns:
        Tuple: (optimized_weights, mean_loss)
        """
        from sklearn.model_selection import KFold
        
        # Sort by date
        sorted_idx = np.argsort([datetime.strptime(d, '%Y-%m-%d') for d in dates])
        predictions = predictions[sorted_idx]
        true_labels = true_labels[sorted_idx]
        dates = dates[sorted_idx]
        
        kf = KFold(n_splits=n_splits, shuffle=False)
        fold_weights = []
        fold_losses = []
        
        for train_idx, test_idx in kf.split(predictions):
            X_train, X_test = predictions[train_idx], predictions[test_idx]
            y_train, y_test = true_labels[train_idx], true_labels[test_idx]
            dates_train, dates_test = dates[train_idx], dates[test_idx]
            
            # Optimize weights on training fold
            fold_weights.append(self.optimize_weights(X_train, y_train, dates_train))
            
            # Calculate loss on test fold
            fold_loss = self.custom_temporal_loss(X_test, y_test, dates_test, fold_weights[-1])
            fold_losses.append(fold_loss)
        
        # Average weights across folds
        optimized_weights = np.mean(fold_weights, axis=0)
        mean_loss = np.mean(fold_losses)
        
        return optimized_weights, mean_loss
    
    def ensemble_predict(self, predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions using optimized weights.
        
        Parameters:
        predictions (np.ndarray): Model predictions (n_samples, n_models)
        weights (np.ndarray): Ensemble weights
        
        Returns:
        np.ndarray: Ensemble predictions
        """
        # Normalize weights
        weights = weights / weights.sum()
        
        # Compute weighted ensemble prediction
        ensemble_pred = np.dot(predictions, weights)
        
        return ensemble_pred
    
    def evaluate_ensemble(self, predictions: np.ndarray, true_labels: np.ndarray,
                         dates: np.ndarray, weights: np.ndarray) -> dict:
        """
        Evaluate ensemble performance with temporal metrics.
        
        Parameters:
        predictions (np.ndarray): Model predictions (n_samples, n_models)
        true_labels (np.ndarray): True binary labels
        dates (np.ndarray): Dates for temporal ordering
        weights (np.ndarray): Ensemble weights
        
        Returns:
        dict: Performance metrics
        """
        results = {}
        
        # Ensemble prediction
        ensemble_pred = self.ensemble_predict(predictions, weights)
        
        # Overall metrics
        results['brier_score'] = brier_score_loss(true_labels, ensemble_pred)
        results['log_loss'] = log_loss(true_labels, ensemble_pred)
        
        # Temporal metrics
        date_diffs = np.array([(datetime.now() - datetime.strptime(d, '%Y-%m-%d')).days 
                              for d in dates])
        results['temporal_brier'] = np.mean((date_diffs / 30) * (ensemble_pred - true_labels) ** 2)
        
        # Weight analysis
        results['weights'] = weights.tolist()
        results['weight_ranks'] = np.argsort(weights)[::-1].tolist()
        
        return results
    
    def generate_optimization_report(self, predictions: np.ndarray, true_labels: np.ndarray,
                                    dates: np.ndarray, optimized_weights: np.ndarray) -> dict:
        """
        Generate comprehensive optimization report.
        
        Parameters:
        predictions (np.ndarray): Model predictions (n_samples, n_models)
        true_labels (np.ndarray): True binary labels
        dates (np.ndarray): Dates for temporal ordering
        optimized_weights (np.ndarray): Optimized model weights
        
        Returns:
        dict: Comprehensive report
        """
        report = {}
        
        # Performance evaluation
        report['performance'] = self.evaluate_ensemble(predictions, true_labels, dates, optimized_weights)
        
        # Weight analysis
        report['weights'] = {
            'optimized_weights': optimized_weights.tolist(),
            'weight_ranks': np.argsort(optimized_weights)[::-1].tolist(),
            'top_3_models': np.argsort(optimized_weights)[-3:][::-1].tolist()
        }
        
        # Improvement metrics
        baseline_brier = np.mean([brier_score_loss(true_labels, predictions[:, i]) 
                                 for i in range(predictions.shape[1])])
        ensemble_brier = report['performance']['brier_score']
        
        report['improvement'] = {
            'brier_improvement': baseline_brier - ensemble_brier,
            'percentage_improvement': ((baseline_brier - ensemble_brier) / baseline_brier * 100) if baseline_brier > 0 else 0
        }
        
        # Temporal analysis
        report['temporal_analysis'] = self.temporal_cross_validation(predictions, true_labels, dates)
        
        return report

# Example usage:
# predictions = np.array([
#     [0.7, 0.6, 0.8],  # Model 1 predictions
#     [0.4, 0.5, 0.3],  # Model 2 predictions
#     [0.9, 0.85, 0.95] # Model 3 predictions
# ]).T
# 
# true_labels = np.array([1, 0, 1])
# dates = np.array(['2024-01-01', '2024-01-05', '2024-01-10'])
# 
# optimizer = TemporalEnsembleOptimizer(n_models=3, temporal_window=30)
# optimized_weights = optimizer.optimize_weights(predictions, true_labels, dates)
# print(f"Optimized weights: {optimized_weights}")
# 
# report = optimizer.generate_optimization_report(predictions, true_labels, dates, optimized_weights)
# print(f"Brier improvement: {report['improvement']['percentage_improvement']:.2f}%")
