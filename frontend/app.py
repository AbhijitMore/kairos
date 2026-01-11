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

from src.kairos.data.loader import load_adult_data
from src.kairos.utils.privacy import mask_for_review

app = Flask(__name__)

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KAIROS-FRONTEND")

# Dataset storage for demo
DATASET = {}

def load_dataset():
    """Loads the dataset into memory for random sampling in the demo."""
    global DATASET
    if DATASET:
        return

    logger.info("KAIROS Frontend: Warming up data cache...")
    try:
        # Load raw data using the new modular loader
        df = load_adult_data()
        
        # We'll use the combined set for the demo dashboard
        DATASET['raw'] = df
        logger.info(f"KAIROS Frontend: Cache Warmup Complete. {len(df)} samples ready.")
    except Exception as e:
        logger.error(f"KAIROS Frontend: Cache Warmup Failed: {e}")
        DATASET = {}

@app.route('/')
def index():
    if not DATASET:
        threading.Thread(target=load_dataset).start()
    return render_template('index.html')

@app.route('/random_profile', methods=['GET'])
def random_profile():
    if not DATASET or 'raw' not in DATASET:
        return jsonify({"error": "Data engine warming up..."}), 503
        
    try:
        df = DATASET['raw']
        idx = random.choice(df.index)
        row = df.loc[idx]
        
        # Convert to dictionary and clean for JSON serialization
        raw_features = row.to_dict()
        
        # Clean up columns not needed for the prediction API
        prediction_payload = {k: v for k, v in raw_features.items() if k not in ['target', 'income', 'fnlwgt', 'education', 'split']}
        
        # --- SCHEMA-AWARE SANITIZATION: Force proper types for Pydantic compliance ---
        def sanitize(key, val):
            numeric_fields = {"age", "education_num", "capital_gain", "capital_loss", "hours_per_week"}
            
            if pd.isna(val) or (isinstance(val, float) and not np.isfinite(val)):
                return 0 if key in numeric_fields else "?"
            
            if isinstance(val, (np.integer, int)):
                return int(val)
            if isinstance(val, (np.floating, float)):
                return float(val)
            return str(val)

        sanitized_payload = {k: sanitize(k, v) for k, v in prediction_payload.items()}

        # Format for display (rounding, stringifying)
        display_features = {}
        for k, v in sanitized_payload.items():
            if v == "?":
                display_features[k] = "Unknown"
            elif isinstance(v, float):
                display_features[k] = round(v, 2)
            else:
                display_features[k] = v

        # Apply privacy masking for the dashboard review section
        masked_features = mask_for_review(display_features)
        
        logger.info(f"Dashboard: Case Generated. Sanitized Keys: {len(sanitized_payload)}")
        
        return jsonify({
            "raw_features": sanitized_payload,
            "features_display": display_features,
            "features_masked": masked_features,
            "ground_truth": int(raw_features.get('target', 0))
        })
    except Exception as e:
        logger.error(f"Random Profile Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.post('/predict')
def predict():
    """
    Proxies prediction requests to the KAIROS Inference API.
    Handles the conversion to the new raw JSON format.
    """
    try:
        # The frontend now expects 'raw_features'
        raw_features = request.json.get('raw_features')
        if not raw_features:
             return jsonify({"error": "No features provided"}), 400
             
        # New API format: {"instances": [{"age": 35, ...}]}
        payload = {"instances": [raw_features]}
        
        logger.info(f"Forwarding prediction request to {API_URL}")
        resp = requests.post(API_URL, json=payload, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            # Secondary sanitization for frontend safety
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
            logger.error(f"API returned error: {resp.status_code} - {resp.text}")
            return jsonify({"error": f"API Error: {resp.text}"}), resp.status_code
            
    except Exception as e:
        logger.error(f"Frontend Proxy Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "ok", "dataset_loaded": 'raw' in DATASET})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
