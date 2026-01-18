import pytest

pytestmark = pytest.mark.unit


from kairos.utils.privacy import (
    generalize_age,
    generalize_marital_status,
    categorize_occupation,
    mask_for_review,
    create_review_payload,
)


def test_generalize_age():
    assert generalize_age(39) == "30-39"
    assert generalize_age(20) == "20-29"
    assert generalize_age(5) == "0-9"
    assert generalize_age("invalid") == "Unknown"
    assert generalize_age(None) == "Unknown"


def test_generalize_marital_status():
    assert generalize_marital_status("Never-married") == "Single"
    assert generalize_marital_status("Married-civ-spouse") == "Partnered"
    assert generalize_marital_status("Divorced") == "Single"
    assert generalize_marital_status("Widowed") == "Single"
    assert generalize_marital_status("Married-spouse-absent") == "Partnered"


def test_categorize_occupation():
    assert categorize_occupation("Tech-support") == "Technical"
    assert categorize_occupation("Sales") == "Sales & Marketing"
    assert categorize_occupation("Unknown-job") == "Other"


def test_mask_for_review():
    sample_case = {
        "age": 39,
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States",
        "relationship": "Husband",
    }

    masked = mask_for_review(sample_case)

    assert masked["age"] == "30-39"
    assert masked["education_num"] == 13
    assert masked["race"] == "[PROTECTED]"
    assert masked["sex"] == "[PROTECTED]"
    assert masked["native_country"] == "[PROTECTED]"
    assert masked["relationship"] == "[PROTECTED]"
    assert masked["marital_status"] == "Single"
    assert masked["occupation"] == "Administrative"


def test_create_review_payload():
    raw_features = {"age": 39, "sex": "Male"}
    payload = create_review_payload("case-123", raw_features, 0.4567, 0.1234)

    assert payload["case_id"] == "case-123"
    assert payload["features"]["age"] == "30-39"
    assert payload["features"]["sex"] == "[PROTECTED]"
    assert payload["model_assessment"]["probability"] == 0.457  # Rounded
    assert payload["model_assessment"]["uncertainty"] == 0.123  # Rounded
    assert "ABSTAIN" in payload["model_assessment"]["reason"]
