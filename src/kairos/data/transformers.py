import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AdultFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for the UCI Adult dataset to ensure consistent feature engineering
    between training and inference.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Initialize engineered columns so schema remains stable
        for col in [
            "capital_net",
            "age_bin",
            "hours_per_edu",
            "hrs_edu",
            "age_edu",
            "cap_gain_tax",
        ]:
            if col not in X.columns:
                X[col] = 0.0

        # 1. Basic cleaning (Ensure consistent types for categorical)
        cat_cols = [
            "workclass",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country",
        ]
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")

        # 2. Socioeconomic Feature Engineering
        if "capital_gain" in X.columns and "capital_loss" in X.columns:
            X["capital_net"] = X["capital_gain"] - X["capital_loss"]

        if "age" in X.columns:
            X["age_bin"] = (
                pd.cut(
                    X["age"],
                    bins=[0, 20, 30, 40, 50, 60, 70, 80, 100],
                    labels=False,
                    include_lowest=True,
                )
                .fillna(0)
                .astype(int)
            )

        if "hours_per_week" in X.columns and "education_num" in X.columns:
            X["hours_per_edu"] = X["hours_per_week"] / (X["education_num"] + 1)
            X["hrs_edu"] = X["hours_per_week"] * X["education_num"]

        if "age" in X.columns and "education_num" in X.columns:
            X["age_edu"] = X["age"] * X["education_num"]
            X["cap_gain_tax"] = (
                X["capital_gain"] * (X["age"] / 100.0)
                if "capital_gain" in X.columns
                else 0
            )

        # Replace any potential NaNs or infs generated
        # Handle numeric and categorical columns separately
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns

        # For numeric columns: replace inf and fillna with 0
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)

        # For categorical columns: only fillna with a string placeholder
        for col in categorical_cols:
            if X[col].isna().any():
                # Add 'Unknown' to categories if not present, then fill
                if isinstance(X[col].dtype, pd.CategoricalDtype):
                    if "Unknown" not in X[col].cat.categories:
                        X[col] = X[col].cat.add_categories(["Unknown"])
                    X[col] = X[col].fillna("Unknown")
                else:
                    X[col] = X[col].fillna("Unknown")

        return X


class HomeCreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for the Home Credit dataset focusing on banking risk metrics.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Financial Ratios
        if "AMT_INCOME_TOTAL" in X.columns:
            if "AMT_CREDIT" in X.columns:
                X["CREDIT_INCOME_PERCENT"] = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"]
            if "AMT_ANNUITY" in X.columns:
                X["ANNUITY_INCOME_PERCENT"] = X["AMT_ANNUITY"] / X["AMT_INCOME_TOTAL"]
            if "AMT_GOODS_PRICE" in X.columns:
                X["GOODS_PRICE_PERCENT"] = X["AMT_GOODS_PRICE"] / X["AMT_INCOME_TOTAL"]

        # 2. Timing Features (Days to Years)
        if "DAYS_BIRTH" in X.columns:
            X["AGE_YEARS"] = X["DAYS_BIRTH"] / -365.0
        if "DAYS_EMPLOYED" in X.columns:
            X["EMPLOYMENT_YEARS"] = X["DAYS_EMPLOYED"] / -365.0

        # 3. Aggregated Scores
        score_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        available_scores = [c for c in score_cols if c in X.columns]
        if available_scores:
            X["EXT_SOURCES_MEAN"] = X[available_scores].mean(axis=1)
            X["EXT_SOURCES_PROD"] = X[available_scores].prod(axis=1)

        # 4. Handle categorical
        cat_cols = X.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            X[col] = X[col].astype("category")

        # Replace any potential NaNs or infs
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)

        # Categorical fillna
        categorical_cols = X.select_dtypes(include=["category"]).columns
        for col in categorical_cols:
            if X[col].isna().any():
                if "Unknown" not in X[col].cat.categories:
                    X[col] = X[col].cat.add_categories(["Unknown"])
                X[col] = X[col].fillna("Unknown")

        return X
