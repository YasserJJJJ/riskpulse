from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import JSON, DateTime, Float, Index, String, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ScoringEvent(Base):
    __tablename__ = "scoring_events"
    __table_args__ = (
        Index(
            "ix_scoring_events_pending_reviews",
            "route",
            "review_outcome",
            "scored_at",
        ),
    )

    transaction_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    idempotency_key: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        index=True,
        nullable=False,
    )
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    route: Mapped[str] = mapped_column(String(32), nullable=False)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    review_outcome: Mapped[str | None] = mapped_column(String(32))
    reviewer_id: Mapped[str | None] = mapped_column(String(128))
    review_notes: Mapped[str | None] = mapped_column(String(2_000))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
