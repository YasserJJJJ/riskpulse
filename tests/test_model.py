from pathlib import Path

import joblib
import pytest

from riskpulse.domain.schemas import TransactionRequest
from riskpulse.ml.features import FEATURE_NAMES, validate_feature_names
from riskpulse.ml.service import ModelLoadError, RiskModel
from riskpulse.ml.training import generate_demo_dataset, train_and_save_model


def test_training_creates_loadable_model(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"

    result = train_and_save_model(path, samples=2_000, random_state=10)
    model = RiskModel.load(path)
    score = model.score(
        TransactionRequest(
            amount=100,
            account_age_days=200,
            hour_of_day=10,
            is_international=False,
            is_new_device=False,
            failed_attempts_24h=0,
            transactions_1h=1,
            distance_from_home_km=5,
        )
    )

    assert result.output_path == path
    assert 0 < result.positive_rate < 1
    assert result.validation_metrics["roc_auc"] > 0.5
    assert 0 <= score <= 1
    assert model.metadata().model_version == result.model_version


def test_demo_dataset_requires_enough_samples() -> None:
    with pytest.raises(ValueError, match="at least 1000"):
        generate_demo_dataset(samples=999)


def test_rejects_mismatched_feature_schema() -> None:
    with pytest.raises(ValueError, match="does not match"):
        validate_feature_names([*FEATURE_NAMES[:-1], "wrong_feature"])


def test_missing_model_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="make train"):
        RiskModel.load(tmp_path / "missing.joblib")


def test_rejects_non_dictionary_artifact(tmp_path: Path) -> None:
    path = tmp_path / "invalid.joblib"
    joblib.dump(["not", "a", "dictionary"], path)

    with pytest.raises(ModelLoadError, match="dictionary"):
        RiskModel.load(path)


def test_rejects_incomplete_artifact(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.joblib"
    joblib.dump({"model": object()}, path)

    with pytest.raises(ModelLoadError, match="missing keys"):
        RiskModel.load(path)
