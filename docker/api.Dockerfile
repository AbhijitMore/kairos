# --- BASE STAGE ---
FROM python:3.10-slim AS base

WORKDIR /app

# Install system dependencies for LightGBM and CatBoost
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- SOURCE STAGE ---
FROM base AS source
COPY src/ ./src/
COPY configs/ ./configs/
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY main.py .

# --- API STAGE ---
FROM source AS api
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- TRAINER STAGE ---
FROM source AS trainer
# Entrypoint for training/HPO
ENTRYPOINT ["python", "main.py"]
