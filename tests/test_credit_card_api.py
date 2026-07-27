from pathlib import Path

from fastapi.testclient import TestClient

from riskpulse.config import Settings
from riskpulse.main import create_app


def credit_card_transaction() -> dict[str, float]:
    return {
        "time": 86_400.0,
        "amount": 125.5,
        **{f"v{index}": 0.0 for index in range(1, 29)},
    }


def test_scores_with_calibrated_model(client: TestClient) -> None:
    response = client.post(
        "/v1/credit-card/transactions/score",
        json=credit_card_transaction(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["fraud_probability"] == 0.5
    assert body["route"] == "manual_review"
    assert body["decision_threshold"] == 0.4
    assert body["model_version"] == "creditcard-test-model"
    assert body["transaction_id"]
    assert body["scored_at"]


def test_routes_low_probability_to_standard_processing(client: TestClient) -> None:
    client.app.state.calibrated_model.decision_threshold = 0.6

    response = client.post(
        "/v1/credit-card/transactions/score",
        json=credit_card_transaction(),
    )

    assert response.status_code == 200
    assert response.json()["route"] == "standard_processing"


def test_returns_calibrated_model_metadata(client: TestClient) -> None:
    response = client.get("/v1/credit-card/model")

    assert response.status_code == 200
    body = response.json()
    assert body["artifact_schema_version"] == "1.0"
    assert body["dataset_id"] == 1597
    assert body["calibration_method"] == "sigmoid"
    assert len(body["feature_names"]) == 30
    assert body["feature_names"][0] == "Time"
    assert body["feature_names"][-1] == "Amount"


def test_rejects_incomplete_credit_card_features(client: TestClient) -> None:
    payload = credit_card_transaction()
    del payload["v28"]

    response = client.post("/v1/credit-card/transactions/score", json=payload)

    assert response.status_code == 422


def test_rejects_unknown_credit_card_features(client: TestClient) -> None:
    payload = credit_card_transaction()
    payload["future_feature"] = 1.0

    response = client.post("/v1/credit-card/transactions/score", json=payload)

    assert response.status_code == 422


def test_rejects_negative_amount(client: TestClient) -> None:
    payload = credit_card_transaction()
    payload["amount"] = -1.0

    response = client.post("/v1/credit-card/transactions/score", json=payload)

    assert response.status_code == 422


def test_real_data_endpoint_is_optional(
    model_path: Path,
    tmp_path: Path,
) -> None:
    settings = Settings(
        model_path=model_path,
        calibrated_model_path=tmp_path / "missing.joblib",
    )

    with TestClient(create_app(settings)) as client:
        unavailable = client.get("/v1/credit-card/model")
        ready = client.get("/health/ready")

    assert unavailable.status_code == 503
    assert unavailable.json()["detail"].startswith("calibrated real-data model")
    assert ready.status_code == 200
