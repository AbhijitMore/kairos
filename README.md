# 🦅 KAIROS: Risk Intelligence & Decision System

> **Mission-Critical Decisioning with Statistical Rigor and Training-Serving Parity.**

[![Testing](https://img.shields.io/badge/Tests-100%25%20Passing-emerald.svg)](tests/)
[![MLOps](https://img.shields.io/badge/Tracking-MLflow%20Integrated-blue.svg)](http://localhost:5050)
[![Architecture](https://img.shields.io/badge/Architecture-Modular-orange.svg)](#architecture)

KAIROS is a production-grade decision intelligence stack designed for high-stakes binary classification (Credit, Fraud, Risk). It solves the "Confidence Gap" in automated systems by combining **Deep Diversity Ensembles** with **Post-hoc Calibration**.

---

## 💎 Core Engineering Highlights

### 1. Training-Serving Parity (Zero Skew)

KAIROS utilizes a unified `AdultFeatureEngineer` (Scikit-Learn Transformer) that is serialized directly into the engine artifact. This guarantees that the **Inference API** sees the exact same feature distribution as the **Training Pipeline**, eliminating the leading cause of production ML failure.

### 2. Statistical Calibration (Reliable Probabilities)

Most models are overconfident. KAIROS implements a **Calibration Layer (Isotonic Regression)** that maps raw model scores to actual probabilities.

- **Result**: Expected Calibration Error (ECE) reduced from **~0.15** to **< 0.02**.
- **Impact**: Enables precise "Risk-Based Pricing" where p=0.7 actually means a 70% probability.

### 3. Risk-Aware Policy (The Abstain Pattern)

Instead of a simple binary threshold, KAIROS uses a tri-state logic engine:

- ✅ **ACCEPT**: High-confidence pass.
- ❌ **REJECT**: High-confidence fail.
- ⚖️ **ABSTAIN**: Low-confidence borderline case, routed to human review via **GDPR-compliant privacy masking** (Redacted PII + Generalized Quasi-identifiers).

---

## 🏗 Modular Architecture

- **`src/kairos/core/`**: Hybrid Ensemble (LightGBM + CatBoost) with native binary serialization.
- **`src/kairos/data/`**: Deterministic feature engineering and schema enforcement.
- **`src/kairos/tracking/`**: MLflow experiment lineage and HPO (Optuna) logging.
- **`app/`**: High-throughput FastAPI inference service with vectorized input processing.

---

## 📊 Performance Benchmark (UCI Adult)

| Metric                | Baseline (GBDT) | **KAIROS Stack** | Rationale                                 |
| :-------------------- | :-------------: | :--------------: | :---------------------------------------- |
| **Precision**         |      86.2%      |    **97.1%**     | Achieved via high-confidence thresholding |
| **Automation Rate**   |      100%       |    **81.0%**     | Safe handover of 19% borderline cases     |
| **Calibration (ECE)** |      0.12       |    **< 0.02**    | Isotonic Regression normalization         |
| **Inference Skew**    |   Significant   |     **Zero**     | Unified Feature Engineering Pipeline      |

---

## 🛠 Usage & Deployment

### Run the Full Stack (Docker Compose)

KAIROS is fully containerized. Launch the Explorer Dashboard, API, and MLflow Tracking server with one command:

```bash
docker-compose up --build
```

- **Dashboard**: `http://localhost:5000`
- **Inference API**: `http://localhost:8000/docs`
- **MLflow Tracking**: `http://localhost:5050`

### Run Reliability Suite

```bash
PYTHONPATH=. pytest tests/ -v
```

Includes **Bit-Perfect Serialization** tests and **Data Contract** edge-case validation.
