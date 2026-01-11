import pytest
import pandas as pd
import numpy as np
import os
from src.kairos.core.pipeline import KairosInferenceEngine

# Define path relative to project root
MODEL_PATH = "outputs/kairos_model"


@pytest.fixture(scope="module")
def real_engine():
    """
    Load the ACTUAL unified pipeline (not a mock).
    """
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model artifact not found at {MODEL_PATH}. Skipping smoke tests.")
    return KairosInferenceEngine.load(MODEL_PATH)


def test_engine_load_sanity(real_engine):
    """Confirm engine loads and has correct components."""
    assert real_engine.pipeline is not None
    assert real_engine.calibrator is not None


def test_prediction_schema_and_range(real_engine):
    """
    SMOKE TEST: Run inference on a generated dummy sample.
    Verifies:
    1. Pipeline accepts dictionary-like dataframe
    2. Output is [0, 1] probability
    3. No NaNs
    """
    # Create a dummy row that matches the Adult schema
    dummy_data = pd.DataFrame(
        [
            {
                "age": 30,
                "workclass": "Private",
                "fnlwgt": 123456,
                "education": "Bachelors",  # Can be dropped by loader but pipeline might handle raw
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
                "income": "<=50K",  # target, ignored
            }
        ]
    )

    # Preprocessing expects simplified columns, but let's see if pipeline handles raw.
    # The pipeline starts with `AdultFeatureEngineer`.
    # `AdultFeatureEngineer.transform` expects the dataframe to have columns.

    probs = real_engine.predict_calibrated(dummy_data)

    # Architecture Test: Output shape
    assert len(probs) == 1

    # Range Test
    prob = probs[0]
    assert isinstance(prob, (float, np.floating))
    assert 0.0 <= prob <= 1.0, f"Probability {prob} out of bounds [0,1]"

    # Reliability Test
    assert not np.isnan(prob), "Model returned NaN probability"


def test_batch_inference(real_engine):
    """Ensure the system can handle a batch of rows without crashing."""
    batch_size = 5
    # Create batch
    batch = pd.DataFrame(
        [
            {
                "age": 25 + i,
                "workclass": "Private",
                "education_num": 10,
                "marital_status": "Never-married",
                "occupation": "Sales",
                "relationship": "Own-child",
                "race": "Black",
                "sex": "Female",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
            for i in range(batch_size)
        ]
    )

    probs = real_engine.predict_calibrated(batch)

    assert len(probs) == batch_size
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
