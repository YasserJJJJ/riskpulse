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


class CreditCardTransactionRequest(BaseModel):
    """Anonymized features from public OpenML credit-card dataset 1597."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "time": 86400.0,
                "amount": 125.5,
                **{f"v{index}": 0.0 for index in range(1, 29)},
            }
        },
    )

    transaction_id: UUID = Field(default_factory=uuid4)
    time: float = Field(ge=0)
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    amount: float = Field(ge=0)


class ReviewRoute(StrEnum):
    STANDARD_PROCESSING = "standard_processing"
    MANUAL_REVIEW = "manual_review"


class CalibratedScoreResponse(BaseModel):
    transaction_id: UUID
    fraud_probability: float = Field(ge=0, le=1)
    route: ReviewRoute
    decision_threshold: float = Field(ge=0, le=1)
    model_version: str
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CalibratedModelMetadataResponse(BaseModel):
    artifact_schema_version: str
    model_version: str
    trained_at: datetime
    model_type: str
    calibration_method: str
    feature_names: list[str]
    decision_threshold: float = Field(ge=0, le=1)
    dataset_id: int
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
