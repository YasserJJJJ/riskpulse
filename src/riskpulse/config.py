from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RISKPULSE_",
        extra="ignore",
    )

    app_name: str = "RiskPulse"
    app_version: str = "0.1.0"
    model_path: Path = Path("artifacts/fraud_model.joblib")
    calibrated_model_path: Path = Path("artifacts/calibrated_creditcard_model.joblib")
    database_url: str = "sqlite:///artifacts/riskpulse.db"
    database_echo: bool = False
    review_threshold: float = 0.50
    decline_threshold: float = 0.85
    log_level: str = "INFO"

    @field_validator("review_threshold", "decline_threshold")
    @classmethod
    def threshold_is_probability(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("thresholds must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> "Settings":
        if self.review_threshold >= self.decline_threshold:
            raise ValueError("review_threshold must be lower than decline_threshold")
        return self
