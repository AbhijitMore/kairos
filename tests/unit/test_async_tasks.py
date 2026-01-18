import pytest

pytestmark = pytest.mark.unit


from unittest.mock import patch, MagicMock
import numpy as np
from app.tasks import predict_batch_task


@patch("app.tasks.get_engine")
def test_predict_batch_task(mock_get_engine):
    # Setup mock engine
    mock_engine = MagicMock()
    mock_engine.predict_calibrated.return_value = np.array([0.1, 0.9])
    mock_get_engine.return_value = mock_engine

    # Test data
    records = [{"age": 30, "workclass": "Private"}, {"age": 40, "workclass": "Public"}]

    # Execute task
    result = predict_batch_task(records)

    # Verify
    assert result["count"] == 2
    assert result["predictions"] == [0.1, 0.9]
    assert "processed_at" in result
    mock_engine.predict_calibrated.assert_called_once()
