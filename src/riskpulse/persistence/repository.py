import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from riskpulse.domain.schemas import (
    CreditCardTransactionRequest,
    ReviewFeedbackRequest,
    ReviewRoute,
)
from riskpulse.persistence.models import ScoringEvent


class AuditConflictError(RuntimeError):
    """Raised when a unique request identity is reused inconsistently."""


class ScoringEventNotFoundError(LookupError):
    """Raised when an audit event cannot be found."""


class FeedbackConflictError(RuntimeError):
    """Raised when immutable review feedback would be overwritten."""


def feature_payload(
    transaction: CreditCardTransactionRequest,
) -> dict[str, float]:
    return {
        "Time": transaction.time,
        **{f"V{index}": getattr(transaction, f"v{index}") for index in range(1, 29)},
        "Amount": transaction.amount,
    }


def hash_features(features: dict[str, float]) -> str:
    canonical = json.dumps(
        features,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class AuditRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_transaction_id(self, transaction_id: UUID) -> ScoringEvent | None:
        return self.session.get(ScoringEvent, transaction_id)

    def get_by_idempotency_key(self, idempotency_key: str) -> ScoringEvent | None:
        statement = select(ScoringEvent).where(ScoringEvent.idempotency_key == idempotency_key)
        return self.session.scalar(statement)

    def find_replay(
        self,
        idempotency_key: str,
        request_hash: str,
    ) -> ScoringEvent | None:
        event = self.get_by_idempotency_key(idempotency_key)
        if event is not None and event.request_hash != request_hash:
            raise AuditConflictError(
                "idempotency key was already used with a different feature payload"
            )
        return event

    def create_event(
        self,
        *,
        transaction_id: UUID,
        idempotency_key: str,
        request_hash: str,
        features: dict[str, float],
        fraud_probability: float,
        route: ReviewRoute,
        decision_threshold: float,
        model_version: str,
        scored_at: datetime,
    ) -> tuple[ScoringEvent, bool]:
        replay = self.find_replay(idempotency_key, request_hash)
        if replay is not None:
            return replay, True
        if self.get_by_transaction_id(transaction_id) is not None:
            raise AuditConflictError(
                "transaction_id was already scored with a different idempotency key"
            )

        event = ScoringEvent(
            transaction_id=transaction_id,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            features=features,
            fraud_probability=fraud_probability,
            route=route.value,
            decision_threshold=decision_threshold,
            model_version=model_version,
            scored_at=scored_at,
        )
        self.session.add(event)
        try:
            self.session.commit()
        except IntegrityError as error:
            self.session.rollback()
            replay = self.find_replay(idempotency_key, request_hash)
            if replay is not None:
                return replay, True
            raise AuditConflictError(
                "transaction identity conflicts with an existing scoring event"
            ) from error
        self.session.refresh(event)
        return event, False

    def require_event(self, transaction_id: UUID) -> ScoringEvent:
        event = self.get_by_transaction_id(transaction_id)
        if event is None:
            raise ScoringEventNotFoundError("scoring event was not found")
        return event

    def pending_reviews(self, limit: int) -> list[ScoringEvent]:
        statement: Select[tuple[ScoringEvent]] = (
            select(ScoringEvent)
            .where(
                ScoringEvent.route == ReviewRoute.MANUAL_REVIEW.value,
                ScoringEvent.review_outcome.is_(None),
            )
            .order_by(ScoringEvent.scored_at.asc())
            .limit(limit)
        )
        return list(self.session.scalars(statement))

    def recent_events(
        self,
        *,
        model_version: str,
        limit: int,
    ) -> list[ScoringEvent]:
        statement: Select[tuple[ScoringEvent]] = (
            select(ScoringEvent)
            .where(ScoringEvent.model_version == model_version)
            .order_by(ScoringEvent.scored_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(statement))

    def record_feedback(
        self,
        transaction_id: UUID,
        feedback: ReviewFeedbackRequest,
    ) -> ScoringEvent:
        event = self.require_event(transaction_id)
        if event.review_outcome is not None:
            if (
                event.review_outcome == feedback.outcome.value
                and event.reviewer_id == feedback.reviewer_id
                and event.review_notes == feedback.notes
            ):
                return event
            raise FeedbackConflictError("review feedback has already been recorded")

        event.review_outcome = feedback.outcome.value
        event.reviewer_id = feedback.reviewer_id
        event.review_notes = feedback.notes
        event.reviewed_at = datetime.now(UTC)
        self.session.commit()
        self.session.refresh(event)
        return event
