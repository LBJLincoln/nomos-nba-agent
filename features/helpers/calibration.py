import numpy as np
from sklearn.calibration import CalibratedClassifierCV, PlattScaling
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target

class PlattScalingCalibrator(BaseEstimator, ClassifierMixin):
    """
    Platt scaling calibrator for binary classification models.

    Parameters:
    base_estimator (object): The base estimator to calibrate.
    method (str, default='sigmoid'): The calibration method.
    cv (int, default=3): The number of folds for cross-validation.

    Attributes:
    calibrated_estimator_ (object): The calibrated estimator.
    """

    def __init__(self, base_estimator, method='sigmoid', cv=3):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv

    def fit(self, X, y):
        """
        Fit the calibrator.

        Parameters:
        X (array-like): The training data.
        y (array-like): The target values.

        Returns:
        self: The calibrated estimator.
        """
        X, y = check_X_y(X, y)
        self.calibrated_estimator_ = CalibratedClassifierCV(
            base_estimator=self.base_estimator,
            method=self.method,
            cv=self.cv
        )
        self.calibrated_estimator_.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predict probabilities.

        Parameters:
        X (array-like): The data to predict.

        Returns:
        array-like: The predicted probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.calibrated_estimator_.predict_proba(X)

class IsotonicRegressionCalibrator(BaseEstimator, ClassifierMixin):
    """
    Isotonic regression calibrator for binary classification models.

    Parameters:
    base_estimator (object): The base estimator to calibrate.

    Attributes:
    calibrated_estimator_ (object): The calibrated estimator.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        """
        Fit the calibrator.

        Parameters:
        X (array-like): The training data.
        y (array-like): The target values.

        Returns:
        self: The calibrated estimator.
        """
        X, y = check_X_y(X, y)
        self.calibrated_estimator_ = CalibratedClassifierCV(
            base_estimator=self.base_estimator,
            method='isotonic',
            cv=3
        )
        self.calibrated_estimator_.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predict probabilities.

        Parameters:
        X (array-like): The data to predict.

        Returns:
        array-like: The predicted probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.calibrated_estimator_.predict_proba(X)

# Example usage:
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a random classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=3, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a base estimator
base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a Platt scaling calibrator
platt_calibrator = PlattScalingCalibrator(base_estimator)

# Fit the calibrator
platt_calibrator.fit(X_train, y_train)

# Predict probabilities
platt_pred_proba = platt_calibrator.predict_proba(X_test)

# Create an isotonic regression calibrator
isotonic_calibrator = IsotonicRegressionCalibrator(base_estimator)

# Fit the calibrator
isotonic_calibrator.fit(X_train, y_train)

# Predict probabilities
isotonic_pred_proba = isotonic_calibrator.predict_proba(X_test)
