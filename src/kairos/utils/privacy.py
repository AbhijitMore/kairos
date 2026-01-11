"""
Privacy-Preserving Data Masking for Human Review
==================================================

This module implements GDPR/CCPA-compliant data masking for cases that require
human review (ABSTAIN decisions). It uses a generalization approach to balance
privacy protection with decision context preservation.

Masking Strategy:
-----------------
1. **Protected Attributes**: Completely redacted to prevent discrimination
   - race, sex, native_country

2. **Quasi-Identifiers**: Generalized to reduce re-identification risk
   - age → age ranges (e.g., 39 → "30-39")
   - marital_status → binary categories
   - occupation → high-level categories

3. **Decision-Relevant Features**: Preserved as-is (low PII risk)
   - education_num, hours_per_week, capital_gain, capital_loss, workclass

Compliance:
-----------
- GDPR Article 5(1)(c): Data minimization
- GDPR Article 25: Privacy by design
- CCPA Section 1798.100: Right to know
- Fair Credit Reporting Act (FCRA): Adverse action notices

Usage:
------
    from src.privacy import mask_for_review
    
    raw_features = {"age": 39, "race": "White", "sex": "Male", ...}
    masked = mask_for_review(raw_features)
    # Output: {"age": "30-39", "race": "[PROTECTED]", "sex": "[PROTECTED]", ...}
"""

from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

# Occupation categorization mapping
OCCUPATION_CATEGORIES = {
    'Tech-support': 'Technical',
    'Craft-repair': 'Skilled Labor',
    'Other-service': 'Service',
    'Sales': 'Sales & Marketing',
    'Exec-managerial': 'Professional',
    'Prof-specialty': 'Professional',
    'Handlers-cleaners': 'Service',
    'Machine-op-inspct': 'Skilled Labor',
    'Adm-clerical': 'Administrative',
    'Farming-fishing': 'Agriculture',
    'Transport-moving': 'Transportation',
    'Priv-house-serv': 'Service',
    'Protective-serv': 'Public Service',
    'Armed-Forces': 'Public Service',
}


def categorize_occupation(occupation: str) -> str:
    """
    Generalizes occupation to high-level category.
    
    Args:
        occupation: Raw occupation string from dataset
        
    Returns:
        High-level occupation category
    """
    return OCCUPATION_CATEGORIES.get(occupation, 'Other')


def generalize_age(age: Union[int, float]) -> str:
    """
    Generalizes age to 10-year ranges.
    
    Args:
        age: Exact age value
        
    Returns:
        Age range string (e.g., "30-39")
    """
    try:
        age_int = int(age)
        lower = (age_int // 10) * 10
        upper = lower + 9
        return f"{lower}-{upper}"
    except (ValueError, TypeError):
        return "Unknown"


def generalize_marital_status(status: str) -> str:
    """
    Generalizes marital status to binary categories.
    
    Args:
        status: Raw marital status string
        
    Returns:
        "Partnered" or "Single"
    """
    status_lower = str(status).lower()
    
    if 'never' in status_lower:
        return 'Single'
        
    partnered_keywords = ['married', 'spouse']
    for keyword in partnered_keywords:
        if keyword in status_lower:
            return 'Partnered'
    
    return 'Single'


def mask_for_review(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies privacy-preserving masking to feature dictionary.
    
    This function implements a generalization-based masking strategy that:
    1. Redacts protected attributes completely
    2. Generalizes quasi-identifiers to reduce re-identification risk
    3. Preserves decision-relevant features
    
    Args:
        features: Dictionary of raw feature values
        
    Returns:
        Dictionary with masked/generalized values
        
    Example:
        >>> raw = {"age": 39, "race": "White", "sex": "Male", "education_num": 13}
        >>> masked = mask_for_review(raw)
        >>> masked
        {"age": "30-39", "race": "[PROTECTED]", "sex": "[PROTECTED]", "education_num": 13}
    """
    masked = {}
    
    for feature_name, value in features.items():
        # Protected attributes - Complete redaction (including proxies like relationship)
        if feature_name in ['race', 'sex', 'native_country', 'relationship']:
            masked[feature_name] = '[PROTECTED]'
            logger.debug(f"Redacted protected attribute: {feature_name}")
        
        # Quasi-identifiers - Generalization
        elif feature_name == 'age':
            masked[feature_name] = generalize_age(value)
        
        elif feature_name == 'marital_status':
            masked[feature_name] = generalize_marital_status(value)
        
        elif feature_name == 'occupation':
            masked[feature_name] = categorize_occupation(value)
        
        # Decision-relevant features - Preserve as-is
        else:
            masked[feature_name] = value
    
    logger.info(f"Masked {len([k for k, v in masked.items() if v == '[PROTECTED]'])} protected attributes")
    return masked


def create_review_payload(
    case_id: str,
    raw_features: Dict[str, Any],
    model_probability: float,
    model_uncertainty: float
) -> Dict[str, Any]:
    """
    Creates a privacy-compliant payload for human review.
    
    Args:
        case_id: Unique identifier for audit trail
        raw_features: Original feature dictionary
        model_probability: Calibrated probability from ensemble
        model_uncertainty: Uncertainty score
        
    Returns:
        Review payload with masked features and metadata
    """
    return {
        'case_id': case_id,
        'features': mask_for_review(raw_features),
        'model_assessment': {
            'probability': round(model_probability, 3),
            'uncertainty': round(model_uncertainty, 3),
            'reason': 'ABSTAIN - Insufficient confidence for automated decision'
        },
        'review_instructions': (
            'Please review the following case. Protected attributes have been '
            'redacted to ensure unbiased decision-making. Focus on education, '
            'work history, and financial indicators.'
        )
    }


if __name__ == "__main__":
    # Example usage
    sample_case = {
        'age': 39,
        'workclass': 'State-gov',
        'education_num': 13,
        'marital_status': 'Never-married',
        'occupation': 'Adm-clerical',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States'
    }
    
    print("Original Features:")
    print(sample_case)
    print("\nMasked for Review:")
    print(mask_for_review(sample_case))
