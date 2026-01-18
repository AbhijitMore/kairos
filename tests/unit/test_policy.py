import pytest

pytestmark = pytest.mark.unit


import numpy as np
from src.kairos.core.policy import KairosPolicy, Action, compute_cost


def test_kairos_policy_decisions():
    policy = KairosPolicy(tau_low=0.2, tau_high=0.8)

    # Test individual decisions
    assert policy.decide(0.1) == Action.REJECT.value
    assert policy.decide(0.5) == Action.ABSTAIN.value
    assert policy.decide(0.9) == Action.ACCEPT.value
    assert policy.decide(0.2) == Action.REJECT.value  # Boundary
    assert policy.decide(0.8) == Action.ACCEPT.value  # Boundary


def test_kairos_policy_predict_vectorized():
    policy = KairosPolicy(tau_low=0.2, tau_high=0.8)
    probs = np.array([0.1, 0.4, 0.9, 0.2, 0.8])
    expected = np.array(
        [
            Action.REJECT.value,
            Action.ABSTAIN.value,
            Action.ACCEPT.value,
            Action.REJECT.value,
            Action.ACCEPT.value,
        ],
        dtype=object,
    )

    np.testing.assert_array_equal(policy.predict_with_policy(probs), expected)


def test_compute_cost_default():
    y_true = np.array([0, 1, 0, 1])
    # Decisions:
    # 0 -> ACCEPT (FP)
    # 1 -> ABSTAIN (Abstain)
    # 0 -> REJECT (Correct)
    # 1 -> REJECT (FN)
    y_pred = np.array(
        [
            Action.ACCEPT.value,
            Action.ABSTAIN.value,
            Action.REJECT.value,
            Action.REJECT.value,
        ]
    )

    # Costs: fp=10, abstain=2, fn=5
    # Total = 10 + 2 + 0 + 5 = 17
    total, avg = compute_cost(y_true, y_pred)
    assert total == 17.0
    assert avg == 17.0 / 4


def test_compute_cost_custom():
    y_true = np.array([0, 1])
    y_pred = np.array([Action.ACCEPT.value, Action.REJECT.value])
    custom_costs = {"fp": 100.0, "fn": 50.0, "abstain": 1.0}

    total, avg = compute_cost(y_true, y_pred, costs=custom_costs)
    assert total == 150.0
    assert avg == 75.0
