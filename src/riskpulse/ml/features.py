from collections.abc import Sequence

import numpy as np

from riskpulse.domain.schemas import TransactionRequest

FEATURE_NAMES: tuple[str, ...] = (
    "amount",
    "account_age_days",
    "hour_of_day",
    "is_international",
    "is_new_device",
    "failed_attempts_24h",
    "transactions_1h",
    "distance_from_home_km",
)


def transaction_to_features(transaction: TransactionRequest) -> np.ndarray:
    return np.asarray(
        [
            [
                transaction.amount,
                transaction.account_age_days,
                transaction.hour_of_day,
                float(transaction.is_international),
                float(transaction.is_new_device),
                transaction.failed_attempts_24h,
                transaction.transactions_1h,
                transaction.distance_from_home_km,
            ]
        ],
        dtype=np.float64,
    )


def validate_feature_names(feature_names: Sequence[str]) -> None:
    if tuple(feature_names) != FEATURE_NAMES:
        raise ValueError(
            "model feature schema does not match the API schema; "
            f"expected {FEATURE_NAMES}, received {tuple(feature_names)}"
        )
