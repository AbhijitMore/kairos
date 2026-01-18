import pytest

pytestmark = pytest.mark.unit


import numpy as np
import pytest

pytestmark = pytest.mark.unit


from sklearn.linear_model import LogisticRegression
from kairos.core.calibration import calibrate_model, compute_ece


def test_compute_ece():
    y_true = np.array([0, 0, 1, 1])
    # Perfect calibration:
    # Bin [0, 0.5): probs 0.1, 0.1 -> Mean prob 0.1, Mean true 0.0 -> Abs diff 0.1
    # Bin [0.5, 1]: probs 0.9, 0.9 -> Mean prob 0.9, Mean true 1.0 -> Abs diff 0.1
    # ECE = (2/4 * 0.1) + (2/4 * 0.1) = 0.1
    y_prob = np.array([0.1, 0.1, 0.9, 0.9])

    ece = compute_ece(y_true, y_prob, n_bins=2)
    assert pytest.approx(ece, 0.01) == 0.1


def test_calibrate_model():
    X_val = np.random.rand(10, 2)
    y_val = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Mock model with predict_proba
    model = LogisticRegression()
    model.fit(X_val, y_val)

    calibrator = calibrate_model(model, X_val, y_val)
    assert hasattr(calibrator, "predict")

    # Test on some sample data
    probs = model.predict_proba(X_val)[:, 1]
    cal_probs = calibrator.predict(probs.reshape(-1, 1))
    assert cal_probs.shape == probs.shape
    assert np.all(cal_probs >= 0) and np.all(cal_probs <= 1)
