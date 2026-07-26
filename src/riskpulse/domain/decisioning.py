from riskpulse.domain.schemas import Decision, TransactionRequest


def make_decision(
    risk_score: float,
    *,
    review_threshold: float,
    decline_threshold: float,
) -> Decision:
    if risk_score >= decline_threshold:
        return Decision.DECLINE
    if risk_score >= review_threshold:
        return Decision.MANUAL_REVIEW
    return Decision.APPROVE


def build_reason_codes(transaction: TransactionRequest, risk_score: float) -> list[str]:
    """Return stable, human-readable signals; these are not causal explanations."""

    reasons: list[str] = []
    if transaction.amount >= 1_000:
        reasons.append("unusually_high_amount")
    if transaction.is_new_device:
        reasons.append("new_device")
    if transaction.is_international:
        reasons.append("international_transaction")
    if transaction.failed_attempts_24h >= 2:
        reasons.append("multiple_failed_attempts")
    if transaction.transactions_1h >= 8:
        reasons.append("high_transaction_velocity")
    if transaction.distance_from_home_km >= 500:
        reasons.append("large_location_change")
    if transaction.hour_of_day <= 4:
        reasons.append("unusual_transaction_time")
    if risk_score >= 0.5 and not reasons:
        reasons.append("elevated_combined_risk")
    return reasons
