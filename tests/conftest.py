from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from riskpulse.config import Settings
from riskpulse.main import create_app
from riskpulse.ml.training import train_and_save_model


@pytest.fixture(scope="session")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models") / "fraud_model.joblib"
    train_and_save_model(path, samples=3_000, random_state=7)
    return path


@pytest.fixture
def settings(model_path: Path) -> Settings:
    return Settings(
        model_path=model_path,
        review_threshold=0.35,
        decline_threshold=0.80,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
