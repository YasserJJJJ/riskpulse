import pytest

from riskpulse.domain.decisioning import build_reason_codes, make_decision
from riskpulse.domain.schemas import Decision, TransactionRequest


@pytest.mark.parametrize(
    ("risk_score", "expected"),
    [
        (0.10, Decision.APPROVE),
        (0.50, Decision.MANUAL_REVIEW),
        (0.85, Decision.DECLINE),
    ],
)
def test_make_decision(risk_score: float, expected: Decision) -> None:
    assert (
        make_decision(
            risk_score,
            review_threshold=0.50,
            decline_threshold=0.85,
        )
        == expected
    )


def test_builds_all_relevant_reason_codes() -> None:
    transaction = TransactionRequest(
        amount=1_500,
        account_age_days=2,
        hour_of_day=3,
        is_international=True,
        is_new_device=True,
        failed_attempts_24h=3,
        transactions_1h=10,
        distance_from_home_km=700,
    )

    assert build_reason_codes(transaction, 0.95) == [
        "unusually_high_amount",
        "new_device",
        "international_transaction",
        "multiple_failed_attempts",
        "high_transaction_velocity",
        "large_location_change",
        "unusual_transaction_time",
    ]


def test_uses_combined_risk_fallback() -> None:
    transaction = TransactionRequest(
        amount=20,
        account_age_days=500,
        hour_of_day=12,
        is_international=False,
        is_new_device=False,
        failed_attempts_24h=0,
        transactions_1h=1,
        distance_from_home_km=2,
    )

    assert build_reason_codes(transaction, 0.60) == ["elevated_combined_risk"]
    assert build_reason_codes(transaction, 0.10) == []
