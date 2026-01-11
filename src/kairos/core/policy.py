from enum import Enum
import numpy as np

class Action(Enum):
    REJECT = "REJECT"
    ACCEPT = "ACCEPT"
    ABSTAIN = "ABSTAIN"

class KairosPolicy:
    """
    Decouples business logic from model output.
    """
    def __init__(self, tau_low: float, tau_high: float):
        self.tau_low = tau_low
        self.tau_high = tau_high

    def decide(self, probability: float) -> str:
        if probability <= self.tau_low:
            return Action.REJECT.value
        elif probability >= self.tau_high:
            return Action.ACCEPT.value
        return Action.ABSTAIN.value

    def predict_with_policy(self, probabilities: np.ndarray) -> np.ndarray:
        """Vectorized decision making."""
        decisions = np.full(probabilities.shape, Action.ABSTAIN.value, dtype=object)
        decisions[probabilities <= self.tau_low] = Action.REJECT.value
        decisions[probabilities >= self.tau_high] = Action.ACCEPT.value
        return decisions

def compute_cost(y_true: np.ndarray, y_pred: np.ndarray, costs: dict = None) -> tuple:
    """
    Computes policy cost based on decisions.
    """
    if costs is None:
        costs = {'fp': 10.0, 'fn': 5.0, 'abstain': 2.0}
        
    total_cost = 0.0
    for y_t, y_p in zip(y_true, y_pred):
        if y_p == Action.ABSTAIN.value:
            total_cost += costs['abstain']
        elif y_t == 0 and y_p == Action.ACCEPT.value: # False Positive
            total_cost += costs['fp']
        elif y_t == 1 and y_p == Action.REJECT.value: # False Negative
            total_cost += costs['fn']
            
    return total_cost, total_cost / len(y_true)
