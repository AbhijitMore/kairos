import pandas as pd
import requests
import os
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]

def download_uci_adult(output_dir: str = "data") -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "adult.data")
    test_path = os.path.join(output_dir, "adult.test")

    urls = {
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data": train_path,
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test": test_path
    }

    for url, path in urls.items():
        if not os.path.exists(path):
            logger.info(f"Downloading {url}...")
            r = requests.get(url)
            with open(path, "wb") as f:
                f.write(r.content)
    return train_path, test_path

def load_adult_data(data_dir: str = "data") -> pd.DataFrame:
    train_path, test_path = download_uci_adult(data_dir)
    
    df_train = pd.read_csv(train_path, names=COLUMN_NAMES, sep=r',\s', engine='python', na_values="?")
    df_test = pd.read_csv(test_path, names=COLUMN_NAMES, sep=r',\s', engine='python', na_values="?", skiprows=1)
    
    df_test['income'] = df_test['income'].str.rstrip('.')
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    
    # Target encoding
    df['target'] = (df['income'] == '>50K').astype(int)
    
    # Drop columns that are handled by the pipeline or not needed
    # We keep 'age', 'workclass', etc. as RAW features
    return df
