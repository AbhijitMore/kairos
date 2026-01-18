import pytest

pytestmark = pytest.mark.unit


import pandas as pd
from unittest.mock import patch, MagicMock
import os
from kairos.data.loader import download_uci_adult, load_adult_data, COLUMN_NAMES


@patch("kairos.data.loader.requests.get")
@patch("kairos.data.loader.os.path.exists")
@patch("kairos.data.loader.open")
def test_download_uci_adult(mock_open, mock_exists, mock_requests):
    mock_exists.return_value = False
    mock_response = MagicMock()
    mock_response.content = b"fake data"
    mock_requests.return_value = mock_response

    output_dir = "fake_data_dir"
    train_path, test_path = download_uci_adult(output_dir)

    assert train_path == os.path.join(output_dir, "adult.data")
    assert test_path == os.path.join(output_dir, "adult.test")
    assert mock_requests.call_count == 2
    mock_open.assert_called()


@patch("kairos.data.loader.download_uci_adult")
@patch("kairos.data.loader.pd.read_csv")
def test_load_adult_data(mock_read_csv, mock_download):
    mock_download.return_value = ("train.csv", "test.csv")

    # Create fake dataframes
    df_train = pd.DataFrame({"age": [39, 50], "income": ["<=50K", "<=50K"]})
    df_test = pd.DataFrame({"age": [38], "income": ["<=50K."]})

    # Match the 25 columns expected in load_adult_data
    for col in COLUMN_NAMES:
        if col not in df_train:
            df_train[col] = "missing"
        if col not in df_test:
            df_test[col] = "missing"

    # Mock read_csv to return these dataframes
    mock_read_csv.side_effect = [df_train, df_test]

    df = load_adult_data("fake_dir")

    assert len(df) == 3
    assert "target" in df.columns
    # Check income cleanup in test set (trailing dot removed)
    assert (df["income"] == "<=50K").all()
    assert (df["target"] == 0).all()
