import pytest

pytestmark = pytest.mark.unit


import pandas as pd
import numpy as np
import pytest

pytestmark = pytest.mark.unit


from kairos.data.transformers import AdultFeatureEngineer
from kairos.core.pipeline import create_kairos_pipeline
from kairos.utils.config_loader import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def pipeline(config):
    # We need a fitted pipeline to test things like unknown categories
    # For speed, we'll create a tiny dummy fit
    pipe = create_kairos_pipeline(config)
    df_train = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
            "workclass": ["Private"] * 10,
            "education_num": [10, 10, 11, 11, 12, 12, 13, 13, 14, 14],
            "marital_status": ["Never-married"] * 10,
            "occupation": ["Tech-support"] * 10,
            "relationship": ["Own-child"] * 10,
            "race": ["White"] * 10,
            "sex": ["Male"] * 10,
            "capital_gain": [0, 0, 100, 100, 200, 200, 300, 300, 5000, 5000],
            "capital_loss": [0] * 10,
            "hours_per_week": [40] * 10,
            "native_country": ["United-States"] * 10,
        }
    )
    y_train = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    pipe.fit(df_train, y_train)
    return pipe


def test_contract_unknown_categorical_values(pipeline):
    """
    Ensures the pipeline doesn't crash when encountering a categorical level
    it hasn't seen during training (e.g., a new country).
    """
    df_new = pd.DataFrame(
        {
            "age": [35],
            "workclass": ["Unknown-Entity"],  # Unknown
            "education_num": [12],
            "marital_status": ["Married-civ-spouse"],
            "occupation": ["Astronaut"],  # Unknown
            "relationship": ["Unmarried"],
            "race": ["Asian-Pac-Islander"],
            "sex": ["Male"],
            "capital_gain": [0],
            "capital_loss": [0],
            "hours_per_week": [40],
            "native_country": ["Mars"],  # Unknown
        }
    )

    # Should not raise ValueError
    try:
        probs = pipeline.predict_proba(df_new)
        assert probs.shape == (1, 2)
    except Exception as e:
        pytest.fail(f"Pipeline crashed on unknown categorical value: {e}")


def test_contract_out_of_bounds_numerical(pipeline):
    """
    Ensures numerical scaling handles extreme outliers without producing NaNs.
    """
    df_outlier = pd.DataFrame(
        {
            "age": [1000],  # Extreme
            "workclass": ["Private"],
            "education_num": [-50],  # Extreme
            "marital_status": ["Never-married"],
            "occupation": ["Tech-support"],
            "relationship": ["Own-child"],
            "race": ["White"],
            "sex": ["Male"],
            "capital_gain": [1_000_000_000],  # Extreme
            "capital_loss": [0],
            "hours_per_week": [168],  # Max hours
            "native_country": ["United-States"],
        }
    )

    probs = pipeline.predict_proba(df_outlier)
    assert not np.any(
        np.isnan(probs)
    ), "NaN detected in probability output for extreme outliers"


def test_contract_missing_non_critical_columns():
    """
    Tests if the feature engineer can handle a completely empty/null row
    by filling defaults rather than crashing.
    """
    engineer = AdultFeatureEngineer()
    df_empty = pd.DataFrame([{}])  # Simulate missing columns

    # AdultFeatureEngineer is expected to be robust
    transformed = engineer.transform(df_empty)
    # It should at least produce the engineered columns with default 0s
    expected_cols = ["capital_net", "age_bin", "hours_per_edu"]
    for col in expected_cols:
        assert col in transformed.columns
