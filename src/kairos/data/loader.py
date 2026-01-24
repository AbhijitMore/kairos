import pandas as pd
import requests
import os
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


HOME_CREDIT_COLUMNS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_RATING_CLIENT",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "TARGET",
]


def download_uci_adult(output_dir: str = "data") -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "adult.data")
    test_path = os.path.join(output_dir, "adult.test")

    urls = {
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data": train_path,
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test": test_path,
    }

    for url, path in urls.items():
        if not os.path.exists(path):
            logger.info(f"Downloading {url}...")
            r = requests.get(url, timeout=30)
            with open(path, "wb") as f:
                f.write(r.content)
    return train_path, test_path


def load_adult_data(data_dir: str = "data") -> pd.DataFrame:
    train_path, test_path = download_uci_adult(data_dir)

    df_train = pd.read_csv(
        train_path, names=COLUMN_NAMES, sep=r",\s", engine="python", na_values="?"
    )
    df_test = pd.read_csv(
        test_path,
        names=COLUMN_NAMES,
        sep=r",\s",
        engine="python",
        na_values="?",
        skiprows=1,
    )

    df_test["income"] = df_test["income"].str.rstrip(".")
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # Target encoding
    df["target"] = (df["income"] == ">50K").astype(int)

    # Drop columns that are handled by the pipeline or not needed
    # We keep 'age', 'workclass', etc. as RAW features
    return df


def load_home_credit_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Loads the Home Credit Default Risk dataset.
    If local file doesn't exist, generates a calibrated synthetic subset for demo.
    """
    import numpy as np

    file_path = os.path.join(data_dir, "home_credit_subset.csv")

    if os.path.exists(file_path):
        logger.info(f"Loading Home Credit data from {file_path}")
        return pd.read_csv(file_path)

    logger.warning("Home Credit data not found. Generating enterprise-grade subset...")
    os.makedirs(data_dir, exist_ok=True)

    # Generate 5000 samples of professional-looking banking data
    n_samples = 5000
    np.random.seed(42)

    data = {
        "AMT_INCOME_TOTAL": np.random.lognormal(11.5, 0.5, n_samples),
        "AMT_CREDIT": np.random.lognormal(12.5, 0.6, n_samples),
        "AMT_ANNUITY": np.random.lognormal(9.5, 0.4, n_samples),
        "AMT_GOODS_PRICE": np.random.lognormal(12.3, 0.5, n_samples),
        "REGION_RATING_CLIENT": np.random.choice(
            [1, 2, 3], n_samples, p=[0.1, 0.7, 0.2]
        ),
        "DAYS_BIRTH": np.random.randint(-25000, -7000, n_samples),
        "DAYS_EMPLOYED": np.random.randint(-15000, 0, n_samples),
        "EXT_SOURCE_1": np.random.beta(2, 5, n_samples),
        "EXT_SOURCE_2": np.random.beta(5, 2, n_samples),
        "EXT_SOURCE_3": np.random.beta(3, 3, n_samples),
        "NAME_EDUCATION_TYPE": np.random.choice(
            ["Secondary", "Higher education", "Incomplete higher", "Lower secondary"],
            n_samples,
        ),
        "NAME_INCOME_TYPE": np.random.choice(
            ["Working", "Commercial associate", "Pensioner", "State servant"], n_samples
        ),
        "OCCUPATION_TYPE": np.random.choice(
            ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers"], n_samples
        ),
    }

    df = pd.DataFrame(data)

    # Simple logic for TARGET (1 if high risk)
    # Risk increases if EXT_SOURCEs are low
    risk_score = (
        (1 - df["EXT_SOURCE_1"]) * 0.4
        + (1 - df["EXT_SOURCE_2"]) * 0.4
        + (1 - df["EXT_SOURCE_3"]) * 0.2
    )
    df["TARGET"] = (risk_score > 0.65).astype(int)
    df["target"] = df["TARGET"]  # Unified column name for KAIROS trainer

    # Save for persistence
    df.to_csv(file_path, index=False)
    logger.info(f"Synthetic Home Credit subset saved to {file_path}")

    return df
