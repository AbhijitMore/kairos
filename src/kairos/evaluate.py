import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

from src.kairos.data.loader import load_adult_data
from src.kairos.core.pipeline import KairosInferenceEngine
from src.kairos.core.policy import KairosPolicy
from src.kairos.core.calibration import compute_ece
from src.kairos.tracking.metrics import evaluate_probabilistic_model

def evaluate(model_path: str = "outputs/kairos_model", regression_gate: bool = True):
    print(f"🦅 KAIROS Evaluation Protocol")
    print(f"Loading model from: {model_path}")
    
    # 1. Load and Reconstruct Test Set (Strict Parity with Training)
    # Note: We use the exact same seed as main.py to reproduce the hold-out set
    df = load_adult_data()
    X = df.drop(columns=['target', 'income', 'fnlwgt', 'education'])
    y = df['target']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Test Set Size: {len(X_test)} samples")
    
    # 2. Load Engine
    try:
        engine = KairosInferenceEngine.load(model_path)
    except FileNotFoundError:
        print(f"❌ Model artifact not found at {model_path}. Run 'python main.py' first.")
        sys.exit(1)
        
    print("Running Inference...")
    probs = engine.predict_calibrated(X_test)
    
    # 3. Policy Execution (Abstain Logic)
    # Standard KAIROS thresholds
    tau_low, tau_high = 0.2, 0.8
    policy = KairosPolicy(tau_low, tau_high)
    preds_policy = policy.predict_with_policy(probs)
    
    # 4. Calculate Claims
    # Coverage: % of cases where we did NOT abstain
    covered_mask = (preds_policy != 'ABSTAIN')
    automation_rate = np.mean(covered_mask)
    
    # Precision on Covered Cases
    y_true_covered = y_test.values[covered_mask]
    # Map ACCEPT/REJECT to 1/0
    preds_mapped = np.where(preds_policy[covered_mask] == 'ACCEPT', 1, 0)
    precision = precision_score(y_true_covered, preds_mapped)
    
    # ECE on Full Set
    ece = compute_ece(y_test.values, probs)
    
    print("\n" + "="*40)
    print("📊 VERIFIED PERFORMANCE METRICS")
    print("="*40)
    print(f"Precision (Claim: >96%):      {precision:.4f} {'✅' if precision >= 0.96 else '⚠️'}")
    print(f"Automation Rate (Claim: ~70%): {automation_rate:.4f}")
    print(f"Calibration ECE (Claim: <0.02): {ece:.5f}   {'✅' if ece < 0.02 else '⚠️'}")
    print("="*40)
    
    # 5. Regression Gate
    if regression_gate:
        if precision < 0.95:
            print(f"\n❌ REGRESSION FAILURE: Precision {precision:.4f} is below 0.95 threshold.")
            sys.exit(1)
        if ece > 0.02:
            print(f"\n❌ REGRESSION FAILURE: ECE {ece:.5f} exceeds 0.02 threshold.")
            sys.exit(1)
        print("\n✅ System Passed All Regression Checks.")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAIROS Canonical Evaluation")
    parser.add_argument("--model-path", default="outputs/kairos_model", help="Path to serialized artifacts")
    parser.add_argument("--no-gate", action="store_true", help="Disable regression failure exit codes")
    args = parser.parse_args()
    
    evaluate(model_path=args.model_path, regression_gate=not args.no_gate)
