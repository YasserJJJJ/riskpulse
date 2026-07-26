from typing import Annotated

from fastapi import APIRouter, Depends, status

from riskpulse.api.dependencies import get_risk_model, get_settings
from riskpulse.config import Settings
from riskpulse.domain.decisioning import build_reason_codes, make_decision
from riskpulse.domain.schemas import (
    ModelMetadataResponse,
    ScoreResponse,
    TransactionRequest,
)
from riskpulse.ml.service import RiskModel

router = APIRouter(prefix="/v1", tags=["risk"])


@router.post(
    "/transactions/score",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
)
def score_transaction(
    transaction: TransactionRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    risk_model: Annotated[RiskModel, Depends(get_risk_model)],
) -> ScoreResponse:
    risk_score = risk_model.score(transaction)
    decision = make_decision(
        risk_score,
        review_threshold=settings.review_threshold,
        decline_threshold=settings.decline_threshold,
    )
    return ScoreResponse(
        transaction_id=transaction.transaction_id,
        risk_score=round(risk_score, 6),
        decision=decision,
        reasons=build_reason_codes(transaction, risk_score),
        model_version=risk_model.model_version,
    )


@router.get("/model", response_model=ModelMetadataResponse)
def get_model_metadata(
    risk_model: Annotated[RiskModel, Depends(get_risk_model)],
) -> ModelMetadataResponse:
    return risk_model.metadata()
