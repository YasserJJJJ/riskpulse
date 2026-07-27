from typing import Annotated

from fastapi import APIRouter, Depends, status

from riskpulse.api.dependencies import get_calibrated_model
from riskpulse.domain.schemas import (
    CalibratedModelMetadataResponse,
    CalibratedScoreResponse,
    CreditCardTransactionRequest,
    ReviewRoute,
)
from riskpulse.ml.calibrated_service import CalibratedFraudModel

router = APIRouter(prefix="/v1/credit-card", tags=["real-data risk"])


@router.post(
    "/transactions/score",
    response_model=CalibratedScoreResponse,
    status_code=status.HTTP_200_OK,
)
def score_credit_card_transaction(
    transaction: CreditCardTransactionRequest,
    model: Annotated[CalibratedFraudModel, Depends(get_calibrated_model)],
) -> CalibratedScoreResponse:
    fraud_probability = model.score(transaction)
    route = (
        ReviewRoute.MANUAL_REVIEW
        if fraud_probability >= model.decision_threshold
        else ReviewRoute.STANDARD_PROCESSING
    )
    return CalibratedScoreResponse(
        transaction_id=transaction.transaction_id,
        fraud_probability=round(fraud_probability, 8),
        route=route,
        decision_threshold=model.decision_threshold,
        model_version=model.model_version,
    )


@router.get("/model", response_model=CalibratedModelMetadataResponse)
def get_calibrated_model_metadata(
    model: Annotated[CalibratedFraudModel, Depends(get_calibrated_model)],
) -> CalibratedModelMetadataResponse:
    return model.metadata()
