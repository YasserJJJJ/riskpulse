"""Create auditable scoring events.

Revision ID: 20260727_01
Revises:
Create Date: 2026-07-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260727_01"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "scoring_events",
        sa.Column("transaction_id", sa.Uuid(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("fraud_probability", sa.Float(), nullable=False),
        sa.Column("route", sa.String(length=32), nullable=False),
        sa.Column("decision_threshold", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("review_outcome", sa.String(length=32), nullable=True),
        sa.Column("reviewer_id", sa.String(length=128), nullable=True),
        sa.Column("review_notes", sa.String(length=2000), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("transaction_id"),
    )
    op.create_index(
        "ix_scoring_events_idempotency_key",
        "scoring_events",
        ["idempotency_key"],
        unique=True,
    )
    op.create_index(
        "ix_scoring_events_pending_reviews",
        "scoring_events",
        ["route", "review_outcome", "scored_at"],
        unique=False,
    )
    op.create_index(
        "ix_scoring_events_scored_at",
        "scoring_events",
        ["scored_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_scoring_events_scored_at", table_name="scoring_events")
    op.drop_index("ix_scoring_events_pending_reviews", table_name="scoring_events")
    op.drop_index("ix_scoring_events_idempotency_key", table_name="scoring_events")
    op.drop_table("scoring_events")
