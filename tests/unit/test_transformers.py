import pytest

pytestmark = pytest.mark.unit


import pandas as pd
from src.kairos.data.transformers import AdultFeatureEngineer


def test_adult_feature_engineer_logic():
    # Arrange
    df = pd.DataFrame(
        {
            "age": [30],
            "capital_gain": [1000],
            "capital_loss": [200],
            "education_num": [10],
            "hours_per_week": [40],
        }
    )

    engineer = AdultFeatureEngineer()

    # Act
    transformed = engineer.transform(df)

    # Assert
    assert "capital_net" in transformed.columns
    assert transformed.loc[0, "capital_net"] == 800
    assert "age_bin" in transformed.columns
    assert transformed.loc[0, "age_bin"] == 1  # 30 is in (20, 30] bin which is index 1
    assert "hours_per_edu" in transformed.columns
    assert transformed.loc[0, "hours_per_edu"] == 40 / 11


def test_adult_feature_engineer_missing_cols():
    # Should handle missing columns gracefully or at least not crash
    df = pd.DataFrame({"age": [25]})
    engineer = AdultFeatureEngineer()
    transformed = engineer.transform(df)
    assert "age_bin" in transformed.columns
