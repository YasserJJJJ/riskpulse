from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from riskpulse.config import Settings
from riskpulse.main import create_app
from riskpulse.ml.datasets import FEATURE_COLUMNS
from riskpulse.ml.training import train_and_save_model


@pytest.fixture(scope="session")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models") / "fraud_model.joblib"
    train_and_save_model(path, samples=3_000, random_state=7)
    return path


@pytest.fixture(scope="session")
def calibrated_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("calibrated-models") / "creditcard.joblib"
    model = DummyClassifier(strategy="prior")
    features = pd.DataFrame(
        [[0.0] * len(FEATURE_COLUMNS) for _ in range(4)],
        columns=list(FEATURE_COLUMNS),
    )
    model.fit(features, [0, 1, 0, 1])
    metrics = {"pr_auc": 0.75, "roc_auc": 0.96}
    joblib.dump(
        {
            "artifact_schema_version": "1.0",
            "model": model,
            "model_version": "creditcard-test-model",
            "trained_at": datetime(2026, 7, 27, tzinfo=UTC).isoformat(),
            "model_type": "calibrated_hist_gradient_boosting",
            "calibration_method": "sigmoid",
            "feature_names": list(FEATURE_COLUMNS),
            "decision_threshold": 0.4,
            "dataset_id": 1597,
            "validation_metrics": metrics,
            "test_metrics": metrics,
        },
        path,
    )
    return path


@pytest.fixture
def settings(model_path: Path, calibrated_model_path: Path) -> Settings:
    return Settings(
        model_path=model_path,
        calibrated_model_path=calibrated_model_path,
        review_threshold=0.35,
        decline_threshold=0.80,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
