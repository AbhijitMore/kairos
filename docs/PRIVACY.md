# Privacy & Compliance Documentation

## Overview

KAIROS implements privacy-preserving data masking for all cases displayed in the frontend interface. This ensures compliance with GDPR, CCPA, and anti-discrimination regulations while maintaining decision transparency.

## Masking Strategy

### 1. Protected Attributes (Complete Redaction)

These fields are **completely masked** to prevent discrimination and bias:

- **`race`** - Racial/ethnic background
- **`sex`** - Gender identity
- **`native_country`** - National origin

**Display**: `🔒 Protected` (shown in amber/italic)

### 2. Quasi-Identifiers (Generalization)

These fields are **generalized** to reduce re-identification risk while preserving context:

- **`age`**: Converted to 10-year ranges
  - Example: `39` → `"30-39"`
- **`marital_status`**: Simplified to binary categories
  - Example: `"Married-civ-spouse"` → `"Partnered"`
  - Example: `"Never-married"` → `"Single"`
- **`occupation`**: Grouped into high-level categories
  - Example: `"Tech-support"` → `"Technical"`
  - Example: `"Exec-managerial"` → `"Professional"`

### 3. Decision-Relevant Features (Preserved)

These fields are **kept as-is** because they have low PII risk and are essential for decision validation:

- `education_num` - Years of education
- `hours_per_week` - Work hours
- `capital_gain` - Investment income
- `capital_loss` - Investment losses
- `workclass` - Employment type

## Implementation

### Backend (`src/privacy.py`)

```python
from src.privacy import mask_for_review

raw_features = {
    "age": 39,
    "race": "White",
    "sex": "Male",
    "education_num": 13
}

masked = mask_for_review(raw_features)
# Output: {
#     "age": "30-39",
#     "race": "[PROTECTED]",
#     "sex": "[PROTECTED]",
#     "education_num": 13
# }
```

### Frontend (`frontend/app.py`)

The `/random_profile` endpoint automatically applies masking:

```python
features_masked = mask_for_review(features_display)
return jsonify({
    "vector": vector,              # Preprocessed for model
    "features_masked": features_masked  # Masked for UI display
})
```

### UI Display (`frontend/templates/index.html`)

Protected fields are visually distinguished:

- **Color**: Amber (`#f59e0b`)
- **Icon**: 🔒 Lock symbol
- **Style**: Italic text

## Compliance Mapping

| Regulation                       | Requirement            | KAIROS Implementation                 |
| -------------------------------- | ---------------------- | ------------------------------------- |
| **GDPR Article 5(1)(c)**         | Data minimization      | Only decision-relevant features shown |
| **GDPR Article 25**              | Privacy by design      | Masking applied by default            |
| **CCPA § 1798.100**              | Right to know          | Audit trail maintained via case IDs   |
| **FCRA § 615**                   | Adverse action notices | Full context preserved for appeals    |
| **Equal Credit Opportunity Act** | Anti-discrimination    | Protected attributes redacted         |

## Audit Trail

While the frontend displays masked data, the backend maintains:

1. **Original Features**: Stored in database with case ID
2. **Masked Features**: Shown to reviewers
3. **Decision Metadata**: Model probability, uncertainty, policy thresholds
4. **Timestamp**: When the case was generated/reviewed

This allows for:

- **Compliance audits**: Verify masking was applied correctly
- **Model debugging**: Investigate false positives/negatives
- **Appeal handling**: Re-examine original data if needed

## Production Deployment

### For ABSTAIN Cases Only

In production, you may want to apply masking **only** for cases sent to human review:

```python
# In app/main.py
if action == Action.ABSTAIN:
    review_payload = create_review_payload(
        case_id=generate_case_id(),
        raw_features=original_features,
        model_probability=cal_prob,
        model_uncertainty=uncertainty
    )
    send_to_review_queue(review_payload)
```

### For All Cases (Current Implementation)

The current frontend applies masking to **all** displayed cases for demonstration purposes. This is the most conservative approach and ensures no PII leakage.

## Testing

Run the privacy module directly to see masking in action:

```bash
# From project root
python src/kairos/utils/privacy.py
```

Expected output:

```
Original Features:
{'age': 39, 'race': 'White', 'sex': 'Male', ...}

Masked for Review:
{'age': '30-39', 'race': '[PROTECTED]', 'sex': '[PROTECTED]', ...}
```

## Future Enhancements

1. **Differential Privacy**: Add noise to numerical features
2. **K-Anonymity**: Ensure each combination appears ≥k times
3. **Consent Management**: Allow users to opt-in to sharing specific attributes
4. **Regional Compliance**: Adjust masking rules based on jurisdiction
