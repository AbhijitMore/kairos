import pandas as pd
import numpy as np
import os
import pytest
from kairos.core.pipeline import create_kairos_pipeline, KairosInferenceEngine
from kairos.utils.config_loader import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def trained_engine(config):
    # Setup a small trained pipeline
    pipe = create_kairos_pipeline(config)
    df_train = pd.DataFrame(
        {
            "age": [25, 45, 30, 60, 40, 50, 22, 55, 33, 44],
            "workclass": ["Private"] * 10,
            "education_num": [10, 15, 12, 16, 11, 14, 9, 13, 12, 11],
            "marital_status": ["Never-married"] * 10,
            "occupation": ["Tech-support"] * 10,
            "relationship": ["Own-child"] * 10,
            "race": ["White"] * 10,
            "sex": ["Male"] * 10,
            "capital_gain": [0, 5000, 100, 10000, 500, 2000, 0, 1000, 200, 300],
            "capital_loss": [0] * 10,
            "hours_per_week": [40] * 10,
            "native_country": ["United-States"] * 10,
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    pipe.fit(df_train, y_train)
    return KairosInferenceEngine(pipe)


def test_pipeline_serialization_reproducibility(trained_engine):
    """
    Critical Test: Verifies that saving and loading the model leads to
    EXACTLY the same results. Any bit-drift here can cause production incidents.
    """
    test_df = pd.DataFrame(
        {
            "age": [35],
            "workclass": ["Private"],
            "education_num": [13],
            "marital_status": ["Married-civ-spouse"],
            "occupation": ["Exec-managerial"],
            "relationship": ["Husband"],
            "race": ["White"],
            "sex": ["Male"],
            "capital_gain": [2000],
            "capital_loss": [0],
            "hours_per_week": [40],
            "native_country": ["United-States"],
        }
    )

    # 1. Get original prediction
    original_probs = trained_engine.predict_calibrated(test_df)

    # 2. Serialize and Deserialize (Robust way)
    dump_path = "tests/test_model_dir"
    trained_engine.save(dump_path)

    reloaded_engine = KairosInferenceEngine.load(dump_path)

    # 3. Get reloaded prediction
    reloaded_probs = reloaded_engine.predict_calibrated(test_df)

    # Cleanup
    import shutil

    if os.path.exists(dump_path):
        shutil.rmtree(dump_path)

    # 4. Assert exact equality
    np.testing.assert_allclose(
        original_probs,
        reloaded_probs,
        rtol=1e-7,
        err_msg="Serialized model output deviated from original!",
    )


def test_policy_logic():
    """Tests the decoupled policy engine."""
    from kairos.core.policy import KairosPolicy

    policy = KairosPolicy(tau_low=0.2, tau_high=0.8)

    assert policy.decide(0.1) == "REJECT"
    assert policy.decide(0.9) == "ACCEPT"
    assert policy.decide(0.5) == "ABSTAIN"


def test_monotonicity_logic(trained_engine):
    """
    Business Logic Test: Ensures that the model follows basic directional logic.
    Increasing 'education_num' should generally increase or maintain the probability
    of >50K income, assuming all other factors are constant.
    """
    base_instance = {
        "age": 35,
        "workclass": "Private",
        "education_num": 8,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

    df_low_edu = pd.DataFrame([base_instance])

    high_edu_instance = base_instance.copy()
    high_edu_instance["education_num"] = 16
    df_high_edu = pd.DataFrame([high_edu_instance])

    prob_low = trained_engine.predict_calibrated(df_low_edu)[0]
    prob_high = trained_engine.predict_calibrated(df_high_edu)[0]

    # In a trained model this should hold. If it doesn't, we have a 'logic bug' or serious data issue.
    # Note: On a tiny 4-row train set it might be noisy, but it tests the harness.
    assert (
        prob_high >= prob_low
    ), f"Monotonicity violation: Education {prob_high} < {prob_low}"
