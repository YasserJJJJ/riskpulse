import pytest
from pydantic import ValidationError

from riskpulse.config import Settings


@pytest.mark.parametrize("field", ["review_threshold", "decline_threshold"])
def test_rejects_threshold_outside_probability_range(field: str) -> None:
    with pytest.raises(ValidationError, match="between 0 and 1"):
        Settings(**{field: 1.1})


def test_rejects_unordered_thresholds() -> None:
    with pytest.raises(ValidationError, match="must be lower"):
        Settings(review_threshold=0.9, decline_threshold=0.8)


def test_rejects_invalid_monitoring_settings() -> None:
    with pytest.raises(ValidationError, match="at least 20"):
        Settings(drift_minimum_events=19)
    with pytest.raises(ValidationError, match="PSI thresholds must be positive"):
        Settings(drift_warning_psi=0)
    with pytest.raises(ValidationError, match="must be lower"):
        Settings(drift_warning_psi=0.3, drift_critical_psi=0.2)
