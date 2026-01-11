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
        X = X.replace([np.inf, -np.inf], 0).fillna(0)

        return X
