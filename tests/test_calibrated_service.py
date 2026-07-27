from pathlib import Path

import joblib
import pytest

from riskpulse.domain.schemas import CreditCardTransactionRequest
from riskpulse.ml.calibrated_service import (
    CalibratedFraudModel,
    CalibratedModelLoadError,
)
from riskpulse.ml.datasets import FEATURE_COLUMNS


def transaction() -> CreditCardTransactionRequest:
    return CreditCardTransactionRequest(
        time=86_400.0,
        amount=125.5,
        **{f"v{index}": 0.0 for index in range(1, 29)},
    )


def test_loads_scores_and_describes_artifact(calibrated_model_path: Path) -> None:
    model = CalibratedFraudModel.load(calibrated_model_path)

    assert model.score(transaction()) == pytest.approx(0.5)
    metadata = model.metadata()
    assert metadata.model_version == "creditcard-test-model"
    assert metadata.feature_names == list(FEATURE_COLUMNS)
    assert metadata.decision_threshold == 0.4
    assert metadata.dataset_id == 1597


def test_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(CalibratedModelLoadError, match="make calibrate"):
        CalibratedFraudModel.load(tmp_path / "missing.joblib")


def test_rejects_unreadable_artifact(tmp_path: Path) -> None:
    path = tmp_path / "broken.joblib"
    path.write_text("not a Joblib artifact")

    with pytest.raises(CalibratedModelLoadError, match="could not load"):
        CalibratedFraudModel.load(path)


def test_rejects_non_dictionary_artifact(tmp_path: Path) -> None:
    path = tmp_path / "list.joblib"
    joblib.dump(["unexpected"], path)

    with pytest.raises(CalibratedModelLoadError, match="must be a dictionary"):
        CalibratedFraudModel.load(path)


def test_rejects_missing_artifact_keys(calibrated_model_path: Path) -> None:
    artifact = joblib.load(calibrated_model_path)
    del artifact["model_version"]

    with pytest.raises(CalibratedModelLoadError, match="model_version"):
        CalibratedFraudModel(artifact)


def test_rejects_unsupported_schema_version(calibrated_model_path: Path) -> None:
    artifact = joblib.load(calibrated_model_path)
    artifact["artifact_schema_version"] = "2.0"

    with pytest.raises(CalibratedModelLoadError, match="schema version"):
        CalibratedFraudModel(artifact)


def test_rejects_feature_schema_mismatch(calibrated_model_path: Path) -> None:
    artifact = joblib.load(calibrated_model_path)
    artifact["feature_names"] = list(reversed(FEATURE_COLUMNS))

    with pytest.raises(CalibratedModelLoadError, match="feature schema"):
        CalibratedFraudModel(artifact)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, "invalid"])
def test_rejects_invalid_threshold(
    calibrated_model_path: Path,
    threshold: object,
) -> None:
    artifact = joblib.load(calibrated_model_path)
    artifact["decision_threshold"] = threshold

    with pytest.raises(CalibratedModelLoadError, match="threshold|metadata"):
        CalibratedFraudModel(artifact)


def test_rejects_model_without_probability_interface(
    calibrated_model_path: Path,
) -> None:
    artifact = joblib.load(calibrated_model_path)
    artifact["model"] = object()

    with pytest.raises(CalibratedModelLoadError, match="predict_proba"):
        CalibratedFraudModel(artifact)


def test_rejects_invalid_metrics(calibrated_model_path: Path) -> None:
    artifact = joblib.load(calibrated_model_path)
    artifact["test_metrics"] = None

    with pytest.raises(CalibratedModelLoadError, match="metadata"):
        CalibratedFraudModel(artifact)
