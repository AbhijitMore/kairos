from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    KAIROS Service Settings
    Loads from environment variables or .env file.
    """

    # Security
    api_key: str = "kairos_dev_key_2026"  # Fallback for dev
    rate_limit: str = "20/minute"

    # Model Thresholds
    tau_low: float = 0.20
    tau_high: float = 0.80

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5050"

    # App Config
    model_dir: str = "outputs/kairos_model"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
