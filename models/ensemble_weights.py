import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from typing import List, Tuple

def optimize_ensemble_weights(predictions: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """
    Optimize ensemble weights using cross-validated log-loss minimization.
    
    Parameters:
    predictions (np.ndarray): 2D array of shape (n_samples, n_models) with model predictions
    true_labels (np.ndarray): 1D array of true binary labels
    
    Returns:
    np.ndarray: Optimized weights for each model
    """
    n_models = predictions.shape[1]
    
    # Define objective function (log-loss)
    def objective(weights):
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        
        # Compute weighted ensemble prediction
        ensemble_pred = np.dot(predictions, weights)
        
        # Calculate log-loss
        loss = log_loss(true_labels, ensemble_pred)
        
        return loss
    
    # Initial weights (equal weighting)
    initial_weights = np.ones(n_models) / n_models
    
    # Bounds for weights (0 to 1)
    bounds = [(0, 1) for _ in range(n_models)]
    
    # Constraints: weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Perform optimization
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints, method='SLSQP')
    
    # Return optimized weights (normalized)
    optimized_weights = result.x / result.x.sum()
    
    return optimized_weights

def cross_validate_ensemble_weights(predictions: np.ndarray, true_labels: np.ndarray, n_splits: int = 5) -> Tuple[np.ndarray, float]:
    """
    Perform cross-validated ensemble weight optimization.
    
    Parameters:
    predictions (np.ndarray): 2D array of shape (n_samples, n_models) with model predictions
    true_labels (np.ndarray): 1D array of true binary labels
    n_splits (int): Number of cross-validation folds
    
    Returns:
    Tuple: (optimized_weights, mean_log_loss)
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_weights = []
    fold_losses = []
    
    for train_idx, test_idx in kf.split(predictions):
        X_train, X_test = predictions[train_idx], predictions[test_idx]
        y_train, y_test = true_labels[train_idx], true_labels[test_idx]
        
        # Optimize weights on training fold
        fold_weights.append(optimize_ensemble_weights(X_train, y_train))
        
        # Calculate log-loss on test fold
        ensemble_pred = np.dot(X_test, fold_weights[-1])
        fold_loss = log_loss(y_test, ensemble_pred)
        fold_losses.append(fold_loss)
    
    # Average weights across folds
    optimized_weights = np.mean(fold_weights, axis=0)
    mean_log_loss = np.mean(fold_losses)
    
    return optimized_weights, mean_log_loss

def ensemble_predict(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Make predictions using weighted ensemble.
    
    Parameters:
    predictions (np.ndarray): 2D array of shape (n_samples, n_models) with model predictions
    weights (np.ndarray): 1D array of model weights
    
    Returns:
    np.ndarray: Ensemble predictions
    """
    # Normalize weights
    weights = weights / weights.sum()
    
    # Compute weighted ensemble prediction
    ensemble_pred = np.dot(predictions, weights)
    
    return ensemble_pred

def evaluate_ensemble_performance(predictions: np.ndarray, true_labels: np.ndarray, weights: np.ndarray) -> dict:
    """
    Evaluate ensemble performance metrics.
    
    Parameters:
    predictions (np.ndarray): 2D array of shape (n_samples, n_models) with model predictions
    true_labels (np.ndarray): 1D array of true binary labels
    weights (np.ndarray): 1D array of model weights
    
    Returns:
    dict: Performance metrics
    """
    results = {}
    
    # Ensemble prediction
    ensemble_pred = ensemble_predict(predictions, weights)
    
    # Log-loss
    results['log_loss'] = log_loss(true_labels, ensemble_pred)
    
    # Brier score
    results['brier_score'] = np.mean((ensemble_pred - true_labels) ** 2)
    
    # Accuracy
    ensemble_label = (ensemble_pred > 0.5).astype(int)
    results['accuracy'] = np.mean(ensemble_label == true_labels)
    
    # Per-model performance
    results['per_model_performance'] = {}
    for i in range(predictions.shape[1]):
        model_pred = predictions[:, i]
        results['per_model_performance'][f'model_{i}'] = {
            'log_loss': log_loss(true_labels, model_pred),
            'brier_score': np.mean((model_pred - true_labels) ** 2),
            'accuracy': np.mean((model_pred > 0.5) == true_labels)
        }
    
    return results

def generate_ensemble_weights_report(predictions: np.ndarray, true_labels: np.ndarray, optimized_weights: np.ndarray) -> dict:
    """
    Generate comprehensive ensemble weights report.
    
    Parameters:
    predictions (np.ndarray): 2D array of shape (n_samples, n_models) with model predictions
    true_labels (np.ndarray): 1D array of true binary labels
    optimized_weights (np.ndarray): Optimized model weights
    
    Returns:
    dict: Comprehensive report
    """
    report = {}
    
    # Performance evaluation
    report['performance'] = evaluate_ensemble_performance(predictions, true_labels, optimized_weights)
    
    # Weight analysis
    report['weights'] = {
        'optimized_weights': optimized_weights.tolist(),
        'weight_ranks': np.argsort(optimized_weights)[::-1].tolist(),
        'top_3_models': np.argsort(optimized_weights)[-3:][::-1].tolist()
    }
    
    # Improvement metrics
    baseline_log_loss = np.mean([log_loss(true_labels, predictions[:, i]) for i in range(predictions.shape[1])])
    ensemble_log_loss = report['performance']['log_loss']
    
    report['improvement'] = {
        'log_loss_improvement': baseline_log_loss - ensemble_log_loss,
        'percentage_improvement': ((baseline_log_loss - ensemble_log_loss) / baseline_log_loss * 100) if baseline_log_loss > 0 else 0
    }
    
    return report

# Example usage:
# predictions = np.array([
#     [0.7, 0.6, 0.8],  # Model 1 predictions
#     [0.4, 0.5, 0.3],  # Model 2 predictions
#     [0.9, 0.85, 0.95] # Model 3 predictions
# ])
# 
# true_labels = np.array([1, 0, 1])
# 
# # Optimize weights
# optimized_weights = optimize_ensemble_weights(predictions, true_labels)
# print(f"Optimized weights: {optimized_weights}")
# 
# # Cross-validated optimization
# cv_weights, cv_log_loss = cross_validate_ensemble_weights(predictions, true_labels, n_splits=5)
# print(f"CV-optimized weights: {cv_weights}, Mean log-loss: {cv_log_loss:.4f}")
# 
# # Generate report
# report = generate_ensemble_weights_report(predictions, true_labels, cv_weights)
# print(f"Log-loss improvement: {report['improvement']['percentage_improvement']:.2f}%")

