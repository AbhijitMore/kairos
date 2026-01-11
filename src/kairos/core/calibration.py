from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from typing import Any
import numpy as np

def calibrate_model(model: Any, X_val: np.ndarray, y_val: np.ndarray) -> Any:
    """
    Fits a post-hoc calibrator (Isotonic Regression) using validation data.
    """
    if hasattr(model, 'predict_proba'):
        val_probs = model.predict_proba(X_val)[:, 1]
    else:
        val_probs = model.predict(X_val)
        
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_probs.reshape(-1, 1), y_val)
    return calibrator

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)

    nonzero = bin_total > 0
    prob_pred_bins = bin_sums[nonzero] / bin_total[nonzero]
    prob_true_bins = bin_true[nonzero] / bin_total[nonzero]
    
    ece = np.sum(np.abs(prob_true_bins - prob_pred_bins) * (bin_total[nonzero] / len(y_true)))
    return float(ece)
