from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class Decision(StrEnum):
    APPROVE = "approve"
    MANUAL_REVIEW = "manual_review"
    DECLINE = "decline"


class TransactionRequest(BaseModel):
    """Features available at transaction authorization time."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "amount": 850.0,
                "account_age_days": 12,
                "hour_of_day": 2,
                "is_international": True,
                "is_new_device": True,
                "failed_attempts_24h": 3,
                "transactions_1h": 9,
                "distance_from_home_km": 620.0,
            }
        }
    )

    transaction_id: UUID = Field(default_factory=uuid4)
    amount: float = Field(gt=0, le=1_000_000)
    account_age_days: int = Field(ge=0, le=36_500)
    hour_of_day: int = Field(ge=0, le=23)
    is_international: bool
    is_new_device: bool
    failed_attempts_24h: int = Field(ge=0, le=100)
    transactions_1h: int = Field(ge=0, le=1_000)
    distance_from_home_km: float = Field(ge=0, le=20_050)


class ScoreResponse(BaseModel):
    transaction_id: UUID
    risk_score: float = Field(ge=0, le=1)
    decision: Decision
    reasons: list[str]
    model_version: str
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelMetadataResponse(BaseModel):
    model_version: str
    trained_at: datetime
    feature_names: list[str]
    validation_metrics: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
