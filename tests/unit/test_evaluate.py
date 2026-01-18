import pytest

pytestmark = pytest.mark.unit


import pytest

pytestmark = pytest.mark.unit


from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from kairos.evaluate import evaluate


@patch("kairos.evaluate.load_adult_data")
@patch("kairos.evaluate.KairosInferenceEngine.load")
@patch("kairos.evaluate.sys.exit")
@patch("kairos.evaluate.KairosPolicy")
def test_evaluate_success(mock_policy_cls, mock_exit, mock_engine_load, mock_load_data):
    mock_exit.side_effect = SystemExit

    # 100 samples
    df = pd.DataFrame(
        {
            "target": [1] * 100,  # All targets are 1
            "income": ["<=50K"] * 100,
            "fnlwgt": [0] * 100,
            "education": ["Some"] * 100,
            "age": [30] * 100,
            "workclass": ["Private"] * 100,
        }
    )
    mock_load_data.return_value = df

    mock_engine = MagicMock()
    # 20 samples for test set
    mock_engine.predict_calibrated.return_value = np.array([1.0] * 20)
    mock_engine_load.return_value = mock_engine

    mock_policy = MagicMock()
    # All are "ACCEPT"
    mock_policy.predict_with_policy.return_value = np.array(["ACCEPT"] * 20)
    mock_policy_cls.return_value = mock_policy

    # This should yield Precision=1.0 and ECE=0.0
    with pytest.raises(SystemExit):
        evaluate(model_path="fake_path", regression_gate=True)
    mock_exit.assert_called_with(0)


@patch("kairos.evaluate.load_adult_data")
@patch("kairos.evaluate.KairosInferenceEngine.load")
@patch("kairos.evaluate.sys.exit")
def test_evaluate_model_not_found(mock_exit, mock_engine_load, mock_load_data):
    mock_exit.side_effect = SystemExit

    df = pd.DataFrame(
        {
            "target": [0, 1] * 5,
            "income": [0] * 10,
            "fnlwgt": [0] * 10,
            "education": [0] * 10,
            "age": [30] * 10,
            "workclass": [0] * 10,
        }
    )
    mock_load_data.return_value = df
    mock_engine_load.side_effect = FileNotFoundError()

    with pytest.raises(SystemExit):
        evaluate(model_path="missing_path")
    mock_exit.assert_called_with(1)


@patch("kairos.evaluate.load_adult_data")
@patch("kairos.evaluate.KairosInferenceEngine.load")
@patch("kairos.evaluate.sys.exit")
@patch("kairos.evaluate.KairosPolicy")
@patch("kairos.evaluate.precision_score")
@patch("kairos.evaluate.compute_ece")
def test_evaluate_regression_failure_precision(
    mock_ece,
    mock_precision,
    mock_policy_cls,
    mock_exit,
    mock_engine_load,
    mock_load_data,
):
    mock_exit.side_effect = SystemExit

    df = pd.DataFrame(
        {
            "target": [1, 0] * 50,
            "income": [0] * 100,
            "fnlwgt": [0] * 100,
            "education": [0] * 100,
            "age": [30] * 100,
            "workclass": [0] * 100,
        }
    )
    mock_load_data.return_value = df

    mock_engine = MagicMock()
    mock_engine.predict_calibrated.return_value = np.array([0.9] * 20)
    mock_engine_load.return_value = mock_engine

    mock_policy = MagicMock()
    mock_policy.predict_with_policy.return_value = np.array(["ACCEPT"] * 20)
    mock_policy_cls.return_value = mock_policy

    mock_precision.return_value = 0.80
    mock_ece.return_value = 0.01

    with pytest.raises(SystemExit):
        evaluate(regression_gate=True)
    mock_exit.assert_called_with(1)


@patch("kairos.evaluate.load_adult_data")
@patch("kairos.evaluate.KairosInferenceEngine.load")
@patch("kairos.evaluate.sys.exit")
@patch("kairos.evaluate.KairosPolicy")
@patch("kairos.evaluate.precision_score")
@patch("kairos.evaluate.compute_ece")
def test_evaluate_regression_failure_ece(
    mock_ece,
    mock_precision,
    mock_policy_cls,
    mock_exit,
    mock_engine_load,
    mock_load_data,
):
    mock_exit.side_effect = SystemExit

    df = pd.DataFrame(
        {
            "target": [1, 0] * 50,
            "income": [0] * 100,
            "fnlwgt": [0] * 100,
            "education": [0] * 100,
            "age": [30] * 100,
            "workclass": [0] * 100,
        }
    )
    mock_load_data.return_value = df

    mock_engine = MagicMock()
    mock_engine.predict_calibrated.return_value = np.array([0.9] * 20)
    mock_engine_load.return_value = mock_engine

    mock_policy = MagicMock()
    mock_policy.predict_with_policy.return_value = np.array(["ACCEPT"] * 20)
    mock_policy_cls.return_value = mock_policy

    mock_precision.return_value = 0.98
    mock_ece.return_value = 0.05

    with pytest.raises(SystemExit):
        evaluate(regression_gate=True)
    mock_exit.assert_called_with(1)
