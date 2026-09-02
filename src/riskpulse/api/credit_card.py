from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session

from riskpulse.api.dependencies import (
    get_calibrated_model,
    get_monitoring_metrics,
    get_session,
)
from riskpulse.domain.schemas import (
    CalibratedModelMetadataResponse,
    CalibratedScoreResponse,
    CreditCardTransactionRequest,
    ReviewFeedbackRequest,
    ReviewFeedbackResponse,
    ReviewRoute,
    ScoringEventResponse,
)
from riskpulse.ml.calibrated_service import CalibratedFraudModel
from riskpulse.monitoring.metrics import MonitoringMetrics
from riskpulse.persistence.models import ScoringEvent
from riskpulse.persistence.repository import (
    AuditConflictError,
    AuditRepository,
    FeedbackConflictError,
    ScoringEventNotFoundError,
    feature_payload,
    hash_features,
)

router = APIRouter(prefix="/v1/credit-card", tags=["real-data risk"])


def _with_utc(timestamp: datetime) -> datetime:
    return timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=UTC)


def _feedback_response(event: ScoringEvent) -> ReviewFeedbackResponse | None:
    if event.review_outcome is None:
        return None
    return ReviewFeedbackResponse(
        outcome=event.review_outcome,
        reviewer_id=event.reviewer_id or "",
        notes=event.review_notes,
        reviewed_at=_with_utc(event.reviewed_at or event.scored_at),
    )


def _audit_response(event: ScoringEvent) -> ScoringEventResponse:
    return ScoringEventResponse(
        transaction_id=event.transaction_id,
        idempotency_key=event.idempotency_key,
        features={str(name): float(value) for name, value in event.features.items()},
        fraud_probability=event.fraud_probability,
        route=event.route,
        decision_threshold=event.decision_threshold,
        model_version=event.model_version,
        scored_at=_with_utc(event.scored_at),
        feedback=_feedback_response(event),
    )


def _score_response(
    event: ScoringEvent,
    *,
    replayed: bool,
) -> CalibratedScoreResponse:
    return CalibratedScoreResponse(
        transaction_id=event.transaction_id,
        idempotency_key=event.idempotency_key,
        idempotency_replayed=replayed,
        fraud_probability=round(event.fraud_probability, 8),
        route=event.route,
        decision_threshold=event.decision_threshold,
        model_version=event.model_version,
        scored_at=_with_utc(event.scored_at),
    )


@router.post(
    "/transactions/score",
    response_model=CalibratedScoreResponse,
    status_code=status.HTTP_200_OK,
)
def score_credit_card_transaction(
    transaction: CreditCardTransactionRequest,
    model: Annotated[CalibratedFraudModel, Depends(get_calibrated_model)],
    session: Annotated[Session, Depends(get_session)],
    metrics: Annotated[MonitoringMetrics, Depends(get_monitoring_metrics)],
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", min_length=1, max_length=128),
    ] = None,
) -> CalibratedScoreResponse:
    resolved_key = (
        f"generated:{transaction.transaction_id}"
        if idempotency_key is None
        else idempotency_key.strip()
    )
    if not resolved_key:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Idempotency-Key cannot be blank",
        )
    features = feature_payload(transaction)
    request_hash = hash_features(features)
    repository = AuditRepository(session)
    try:
        replay = repository.find_replay(resolved_key, request_hash)
    except AuditConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    if replay is not None:
        metrics.observe_score(
            model_version=replay.model_version,
            route=replay.route,
            replayed=True,
            fraud_probability=replay.fraud_probability,
        )
        return _score_response(replay, replayed=True)

    fraud_probability = model.score(transaction)
    route = (
        ReviewRoute.MANUAL_REVIEW
        if fraud_probability >= model.decision_threshold
        else ReviewRoute.STANDARD_PROCESSING
    )
    try:
        event, replayed = repository.create_event(
            transaction_id=transaction.transaction_id,
            idempotency_key=resolved_key,
            request_hash=request_hash,
            features=features,
            fraud_probability=fraud_probability,
            route=route,
            decision_threshold=model.decision_threshold,
            model_version=model.model_version,
            scored_at=datetime.now(UTC),
        )
    except AuditConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    metrics.observe_score(
        model_version=event.model_version,
        route=event.route,
        replayed=replayed,
        fraud_probability=event.fraud_probability,
    )
    return _score_response(event, replayed=replayed)


@router.get("/model", response_model=CalibratedModelMetadataResponse)
def get_calibrated_model_metadata(
    model: Annotated[CalibratedFraudModel, Depends(get_calibrated_model)],
) -> CalibratedModelMetadataResponse:
    return model.metadata()


@router.get("/reviews", response_model=list[ScoringEventResponse])
def list_pending_reviews(
    session: Annotated[Session, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[ScoringEventResponse]:
    events = AuditRepository(session).pending_reviews(limit)
    return [_audit_response(event) for event in events]


@router.get(
    "/transactions/{transaction_id}",
    response_model=ScoringEventResponse,
)
def get_scoring_event(
    transaction_id: UUID,
    session: Annotated[Session, Depends(get_session)],
) -> ScoringEventResponse:
    try:
        event = AuditRepository(session).require_event(transaction_id)
    except ScoringEventNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    return _audit_response(event)


@router.post(
    "/transactions/{transaction_id}/feedback",
    response_model=ScoringEventResponse,
)
def record_review_feedback(
    transaction_id: UUID,
    feedback: ReviewFeedbackRequest,
    session: Annotated[Session, Depends(get_session)],
    metrics: Annotated[MonitoringMetrics, Depends(get_monitoring_metrics)],
) -> ScoringEventResponse:
    try:
        event = AuditRepository(session).record_feedback(transaction_id, feedback)
    except ScoringEventNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except FeedbackConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    metrics.observe_feedback(outcome=event.review_outcome or feedback.outcome.value)
    return _audit_response(event)
