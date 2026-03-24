import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from scipy.stats import beta
import pandas as pd

class AdvancedCalibrator:
    """
    Advanced probability calibration for NBA predictions.
    Uses isotonic regression with beta calibration for improved Brier scores.
    """
    
    def __init__(self, n_bins: int = 10, smoothing_alpha: float = 1.0):
        """
        Initialize calibrator.
        
        Parameters:
        n_bins (int): Number of bins for initial calibration
        smoothing_alpha (float): Beta distribution smoothing parameter
        """
        self.n_bins = n_bins
        self.smoothing_alpha = smoothing_alpha
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        self.beta_params = None
        self.bin_edges = None
        self.bin_means = None
        
    def fit(self, probabilities: np.ndarray, true_labels: np.ndarray):
        """
        Fit the calibrator to training data.
        
        Parameters:
        probabilities (np.ndarray): Raw model probabilities
        true_labels (np.ndarray): True binary labels (0 or 1)
        """
        if len(probabilities) < 10:
            raise ValueError("Need at least 10 samples for calibration")
        
        # Step 1: Beta calibration for initial smoothing
        self._fit_beta_calibration(probabilities, true_labels)
        
        # Step 2: Isotonic regression on beta-calibrated probabilities
        beta_calibrated = self._apply_beta_calibration(probabilities)
        self._fit_isotonic_regression(beta_calibrated, true_labels)
        
        # Step 3: Bin analysis for confidence adjustment
        self._compute_bin_statistics(probabilities, true_labels)
        
    def _fit_beta_calibration(self, probabilities: np.ndarray, true_labels: np.ndarray):
        """
        Fit beta calibration model.
        """
        # Beta calibration parameters: P(y=1|x) = (1 + expit(a + bx)) / (1 + c * expit(a + bx))
        # Simplified version using moment matching
        mean_prob = np.mean(probabilities)
        mean_label = np.mean(true_labels)
        
        # Estimate beta parameters
        a = np.log(mean_label / (1 - mean_label)) - np.log(mean_prob / (1 - mean_prob))
        b = 1.0
        c = 1.0 + self.smoothing_alpha * (mean_label - mean_prob)
        
        self.beta_params = (a, b, c)
        
    def _apply_beta_calibration(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply beta calibration to probabilities.
        """
        if self.beta_params is None:
            return probabilities
        
        a, b, c = self.beta_params
        expit = 1 / (1 + np.exp(-(a + b * np.log(probabilities / (1 - probabilities)))))
        calibrated = (1 + expit) / (1 + c * expit)
        return np.clip(calibrated, 0.01, 0.99)
    
    def _fit_isotonic_regression(self, probabilities: np.ndarray, true_labels: np.ndarray):
        """
        Fit isotonic regression on beta-calibrated probabilities.
        """
        self.isotonic_regressor.fit(probabilities, true_labels)
    
    def _compute_bin_statistics(self, probabilities: np.ndarray, true_labels: np.ndarray):
        """
        Compute bin statistics for confidence adjustment.
        """
        # Create bins
        self.bin_edges = np.histogram_bin_edges(probabilities, self.n_bins)
        bin_indices = np.digitize(probabilities, self.bin_edges)
        
        # Calculate bin means
        self.bin_means = []
        for i in range(1, self.n_bins + 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_mean = np.mean(probabilities[mask])
                true_mean = np.mean(true_labels[mask])
                self.bin_means.append((bin_mean, true_mean, np.sum(mask)))
            else:
                self.bin_means.append((None, None, 0))
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calibrate new probabilities.
        
        Parameters:
        probabilities (np.ndarray): Raw model probabilities
        
        Returns:
        np.ndarray: Calibrated probabilities
        """
        if len(probabilities) == 0:
            return np.array([])
        
        # Apply beta calibration
        beta_calibrated = self._apply_beta_calibration(probabilities)
        
        # Apply isotonic regression
        calibrated = self.isotonic_regressor.predict(beta_calibrated)
        
        # Apply bin-based confidence adjustment
        calibrated = self._adjust_confidence(calibrated)
        
        return np.clip(calibrated, 0.01, 0.99)
    
    def _adjust_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Adjust confidence based on bin statistics.
        """
        if self.bin_means is None or len(self.bin_means) == 0:
            return probabilities
        
        # Find bin for each probability
        bin_indices = np.digitize(probabilities, self.bin_edges)
        
        # Adjust based on bin statistics
        adjusted = []
        for i, bin_idx in enumerate(bin_indices):
            if bin_idx < 1 or bin_idx > len(self.bin_means):
                adjusted.append(probabilities[i])
                continue
            
            bin_mean, true_mean, count = self.bin_means[bin_idx - 1]
            
            if bin_mean is None or count < 5:
                adjusted.append(probabilities[i])
            else:
                # Confidence adjustment: if model is overconfident, reduce confidence
                confidence_ratio = abs(probabilities[i] - 0.5) / 0.5
                adjustment = (true_mean - bin_mean) * confidence_ratio
                adjusted_value = np.clip(probabilities[i] + adjustment, 0.01, 0.99)
                adjusted.append(adjusted_value)
        
        return np.array(adjusted)
    
    def get_calibration_metrics(self, probabilities: np.ndarray, true_labels: np.ndarray) -> dict:
        """
        Calculate calibration metrics.
        
        Parameters:
        probabilities (np.ndarray): Calibrated probabilities
        true_labels (np.ndarray): True binary labels
        
        Returns:
        dict: Calibration metrics
        """
        metrics = {}
        
        # Brier score
        metrics['brier_score'] = np.mean((probabilities - true_labels) ** 2)
        
        # Log loss
        eps = 1e-15
        log_probs = np.clip(probabilities, eps, 1 - eps)
        metrics['log_loss'] = -np.mean(true_labels * np.log(log_probs) + (1 - true_labels) * np.log(1 - log_probs))
        
        # Calibration error
        prob_true, prob_pred = calibration_curve(true_labels, probabilities, n_bins=self.n_bins)
        metrics['calibration_error'] = np.mean(np.abs(prob_true - prob_pred))
        
        # Reliability diagram data
        metrics['reliability_data'] = {
            'predicted': prob_pred,
            'actual': prob_true
        }
        
        # Confidence distribution
        metrics['confidence_stats'] = {
            'mean': np.mean(probabilities),
            'std': np.std(probabilities),
            'overconfident': np.mean(probabilities > 0.7),
            'underconfident': np.mean(probabilities < 0.3)
        }
        
        return metrics
    
    def plot_calibration(self, probabilities: np.ndarray, true_labels: np.ndarray, output_path: str = 'calibration_plot.png'):
        """
        Plot calibration curve.
        
        Parameters:
        probabilities (np.ndarray): Calibrated probabilities
        true_labels (np.ndarray): True binary labels
        output_path (str): Path to save plot
        """
        import matplotlib.pyplot as plt
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(true_labels, probabilities, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model Calibration')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Probability Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Calibration plot saved to {output_path}")

# Example usage:
# calibrator = AdvancedCalibrator(n_bins=10, smoothing_alpha=1.0)
# calibrator.fit(raw_probabilities, true_labels)
# calibrated_probs = calibrator.calibrate(new_raw_probabilities)
# metrics = calibrator.get_calibration_metrics(calibrated_probs, true_labels)
# calibrator.plot_calibration(calibrated_probs, true_labels)
