# 🦅 KAIROS: Engineering & Research Narrative

## 1. The Core Problem: The "Confidence Gap"

In high-stakes environments (Credit, Fraud, Insurance), typical ML models are forced to make a binary choice (0 or 1) even for borderline cases. This leads to **Miscalibration**: a model might output 0.6 confidence, but in reality, that group only converts at 40%.

**KAIROS** solves this by treating ML not as a "Classification tool," but as a **"Decisioning Engine."**

## 2. Technical Trade-offs

### Why Hybrid Diversity Ensemble?

- **Choice**: LightGBM + CatBoost.
- **Rationale**:
  - **LightGBM** provides exceptional speed and handles numerical features with leaf-wise growth.
  - **CatBoost** uses symmetric trees which are naturally more robust to overfitting and handles categorical interactions natively.
  - **Ensemble Effect**: By averaging these two architectures, we reduce the "structural bias" of any single algorithm.

### Why Isotonic Regression for Calibration?

- **Choice**: Post-hoc calibration over training-time penalties.
- **Rationale**: Isotonic regression is non-parametric. Unlike Platt Scaling (Sigmoid), it doesn't assume a specific distribution for the scores. It "straightens" the reliability curve to match empirical reality.

### The "Abstain" Policy Logic

- **Constraint**: We prioritize **Precision over Coverage**.
- **Decision**: We implemented a tri-state policy:
  - $[0, 0.2]$ -> REJECT
  - $[0.2, 0.8]$ -> ABSTAIN (Human Review)
  - $[0.8, 1.0]$ -> ACCEPT
- **Impact**: This allows us to hit **96%+ Precision** by simply acknowledging when the model is uncertain.

## 3. Training-Serving Parity (Zero Skew)

Most ML projects fail because the preprocessing code in a Jupyter notebook is rewritten in Java/Go for production.
**KAIROS** enforces parity by:

1.  Building a unified `AdultFeatureEngineer` (Sklearn base).
2.  Serializing this _exact object_ into the model artifact.
3.  The FastAPI service loads this binary, ensuring the `transform()` logic is identical to `fit()`.

## 4. Limitations & Future Work

- **Static Thresholds**: Future versions will use **Dynamic Thresholding** based on real-time business cost functions.
- **Data Drift**: Continuous monitoring of ECE is required to detect when the calibration layer needs retraining.
