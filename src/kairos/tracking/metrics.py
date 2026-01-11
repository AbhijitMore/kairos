import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from typing import Dict, Any
from src.kairos.core.policy import Action, KairosPolicy, compute_cost


def evaluate_probabilistic_model(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Standard binary classification evaluation.
    """
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }


def evaluate_policy(
    y_true: np.ndarray, probs: np.ndarray, tau_low: float, tau_high: float
) -> Dict[str, Any]:
    """
    Evaluates system performance under the risk-aware policy.
    """
    policy = KairosPolicy(tau_low, tau_high)
    preds = policy.predict_with_policy(probs)
    abstain_mask = preds == Action.ABSTAIN.value
    covered_mask = ~abstain_mask

    coverage = float(np.mean(covered_mask))
    total_cost, avg_cost = compute_cost(y_true, preds)

    accuracy_covered = 0.0
    if np.sum(covered_mask) > 0:
        # Map decisions to 0/1 for accuracy calculation
        # REJECT -> 0, ACCEPT -> 1
        # We only care about covered items here
        covered_preds = preds[covered_mask]
        covered_labels = np.zeros_like(covered_preds, dtype=int)
        covered_labels[covered_preds == Action.ACCEPT.value] = 1
        covered_labels[covered_preds == Action.REJECT.value] = 0

        accuracy_covered = float(accuracy_score(y_true[covered_mask], covered_labels))

    return {
        "coverage": coverage,
        "abstention_rate": 1.0 - coverage,
        "avg_cost": avg_cost,
        "accuracy_covered": accuracy_covered,
        "error_covered": 1.0 - accuracy_covered,
    }


def plot_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, title: str = "Calibration Curve"
):
    """
    Utility to visualize probability reliability.
    """
    from .calibration import plot_reliability_diagram

    return plot_reliability_diagram(y_true, y_prob, title=title)


def plot_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, tau_low: float, tau_high: float
):
    """
    Visualizes the 3x2 decision matrix.
    """
    policy = KairosPolicy(tau_low, tau_high)
    preds = policy.predict_with_policy(y_prob)

    # Rows: Reject, Accept, Abstain
    # Cols: 0, 1
    cm = np.zeros((3, 2), dtype=int)
    for p, t in zip(preds, y_true):
        col = int(t)
        if p == Action.REJECT.value:
            row = 0
        elif p == Action.ACCEPT.value:
            row = 1
        else:
            row = 2
        cm[row, col] += 1

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low Income", "High Income"],
        yticklabels=["Reject", "Accept", "Abstain"],
    )
    plt.title(f"Policy Confusion Matrix (Low={tau_low}, High={tau_high})")
    plt.xlabel("Actual")
    plt.ylabel("Decision")
    plt.show()
    return cm
