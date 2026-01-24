from flask import Flask, render_template, request, jsonify
import requests
import os
import sys
import pandas as pd
import numpy as np
import threading
import logging
import random

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from kairos.data.loader import load_adult_data, load_home_credit_data  # noqa: E402
from kairos.utils.privacy import mask_for_review  # noqa: E402

app = Flask(__name__)

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/predict")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KAIROS-FRONTEND")

# Dataset storage for demo
DATASETS = {"adult": None, "home_credit": None}
CURRENT_DATASET = "adult"


def load_datasets():
    """Loads datasets into memory for random sampling in the demo."""
    global DATASETS
    logger.info("KAIROS Frontend: Warming up data cache...")
    try:
        DATASETS["adult"] = load_adult_data()
        DATASETS["home_credit"] = load_home_credit_data()
        logger.info("KAIROS Frontend: Cache Warmup Complete.")
    except Exception as e:
        logger.error(f"KAIROS Frontend: Cache Warmup Failed: {e}")


@app.route("/")
def index():
    if DATASETS["adult"] is None:
        threading.Thread(target=load_datasets).start()
    return render_template("index.html")


@app.route("/switch_dataset", methods=["POST"])
def switch_dataset():
    global CURRENT_DATASET
    ds = request.json.get("dataset")
    if ds in DATASETS:
        CURRENT_DATASET = ds
        return jsonify({"status": "success", "current": ds})
    return jsonify({"error": "Invalid dataset"}), 400


@app.route("/random_profile", methods=["GET"])
def random_profile():
    global CURRENT_DATASET
    df = DATASETS.get(CURRENT_DATASET)

    if df is None:
        return jsonify({"error": "Data engine warming up..."}), 503

    try:
        idx = random.choice(df.index)  # nosec B311
        row = df.loc[idx]
        raw_features = row.to_dict()

        # Clean up columns not needed for the prediction API
        prediction_payload = {
            k: v
            for k, v in raw_features.items()
            if k not in ["target", "income", "fnlwgt", "education", "split", "TARGET"]
        }

        # --- DYNAMIC SCHEMA-AWARE SANITIZATION ---
        def sanitize(key, val):
            if pd.isna(val):
                # Banking numeric fields should default to 0
                if key.startswith("AMT_") or key.startswith("EXT_") or "DAYS" in key:
                    return 0.0
                # Adult dataset numeric fields
                if key in [
                    "age",
                    "education_num",
                    "capital_gain",
                    "capital_loss",
                    "hours_per_week",
                ]:
                    return 0
                return "Unknown"

            if isinstance(val, (np.integer, int)):
                return int(val)
            if isinstance(val, (np.floating, float)):
                if not np.isfinite(val):
                    return 0.0
                return float(val)
            return str(val)

        sanitized_payload = {k: sanitize(k, v) for k, v in prediction_payload.items()}

        # Format for display
        display_features = {}
        for k, v in sanitized_payload.items():
            if isinstance(v, float):
                display_features[k] = round(v, 2)
            else:
                display_features[k] = v

        # Apply privacy masking
        masked_features = mask_for_review(display_features)

        return jsonify(
            {
                "dataset": CURRENT_DATASET,
                "raw_features": sanitized_payload,
                "features_display": display_features,
                "features_masked": masked_features,
                "ground_truth": int(raw_features.get("target", 0)),
            }
        )
    except Exception as e:
        logger.error(f"Random Profile Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.post("/predict")
def predict():
    """
    Proxies prediction requests to the KAIROS Inference API.
    """
    try:
        raw_features = request.json.get("raw_features")
        dataset = request.json.get("dataset", CURRENT_DATASET)

        if not raw_features:
            return jsonify({"error": "No features provided"}), 400

        # Generalized API format with discriminator support
        instances = []
        for feat in [raw_features] if isinstance(raw_features, dict) else raw_features:
            record = feat.copy()
            record["dataset_type"] = dataset
            instances.append(record)

        payload = {"dataset": dataset, "instances": instances}

        API_KEY = os.getenv("API_KEY", "kairos_dev_key_2026")
        headers = {"X-API-KEY": API_KEY}
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=10)

        if resp.status_code == 200:
            data = resp.json()

            def clean_obj(obj):
                if isinstance(obj, list):
                    return [clean_obj(x) for x in obj]
                if isinstance(obj, dict):
                    return {k: clean_obj(v) for k, v in obj.items()}
                if isinstance(obj, float) and not np.isfinite(obj):
                    return 0.5
                return obj

            cleaned_data = clean_obj(data[0] if isinstance(data, list) else data)
            return jsonify(cleaned_data)
        else:
            return jsonify({"error": f"API Error: {resp.text}"}), resp.status_code

    except Exception as e:
        logger.error(f"Frontend Proxy Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "datasets_ready": [k for k, v in DATASETS.items() if v is not None],
            "current_dataset": CURRENT_DATASET,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)  # nosec B104
