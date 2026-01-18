import pytest

pytestmark = pytest.mark.unit


import numpy as np
import pytest

pytestmark = pytest.mark.unit


from src.kairos.tracking.metrics import evaluate_probabilistic_model, evaluate_policy


def test_evaluate_probabilistic_model():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = evaluate_probabilistic_model(y_true, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["log_loss"] < 0.5


def test_evaluate_policy():
    y_true = np.array([0, 1, 0, 1])
    probs = np.array([0.1, 0.9, 0.5, 0.2])
    # tau_low=0.2, tau_high=0.8
    # Decisions: 0.1->REJECT, 0.9->ACCEPT, 0.5->ABSTAIN, 0.2->REJECT
    # Covered: [0.1, 0.9, 0.2]
    # Coverage: 3/4 = 0.75

    metrics = evaluate_policy(y_true, probs, tau_low=0.2, tau_high=0.8)
    assert metrics["coverage"] == 0.75
    assert metrics["abstention_rate"] == 0.25
    # Covered items: (0, REJECT), (1, ACCEPT), (1, REJECT)
    # (1, REJECT) is an error.
    # Accuracy covered: 2/3 = 0.666...
    assert pytest.approx(metrics["accuracy_covered"], 0.01) == 0.666
