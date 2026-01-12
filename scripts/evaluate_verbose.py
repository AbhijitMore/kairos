import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from src.kairos.data.loader import load_adult_data
from src.kairos.core.pipeline import KairosInferenceEngine
from src.kairos.core.policy import KairosPolicy

# 1. Load Data
df = load_adult_data()
X = df.drop(columns=["target", "income", "fnlwgt", "education"])
y = df["target"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Load Engine
engine = KairosInferenceEngine.load("outputs/kairos_model")
probs = engine.predict_calibrated(X_test)
raw_preds = (probs >= 0.5).astype(int)

# 3. Policy decisions
policy = KairosPolicy(0.2, 0.8)
policy_preds = policy.predict_with_policy(probs)
covered_mask = policy_preds != "ABSTAIN"
y_true_covered = y_test.values[covered_mask]
preds_mapped = np.where(policy_preds[covered_mask] == "ACCEPT", 1, 0)

# 4. Calculate
print("--- RAW MODEL METRICS (Threshold 0.5) ---")
print(f"Precision: {precision_score(y_test, raw_preds):.4f}")
print(f"Recall:    {recall_score(y_test, raw_preds):.4f}")
print(f"F1:        {f1_score(y_test, raw_preds):.4f}")

print("\n--- POLICY-GATED METRICS (Automation) ---")
print(f"Automation Rate: {np.mean(covered_mask):.4f}")
print(f"Gated Precision: {precision_score(y_true_covered, preds_mapped):.4f}")
print(f"Gated Recall:    {recall_score(y_true_covered, preds_mapped):.4f}")
