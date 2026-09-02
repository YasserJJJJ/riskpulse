from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


def credit_card_transaction(*, value: float = 0.0) -> dict[str, float]:
    return {
        "time": value,
        "amount": value,
        **{f"v{index}": value for index in range(1, 29)},
    }


def test_exports_prometheus_metrics_and_request_ids(client: TestClient) -> None:
    score = client.post(
        "/v1/credit-card/transactions/score",
        json=credit_card_transaction(value=50.0),
        headers={
            "Idempotency-Key": "metrics-score-001",
            "X-Request-ID": "portfolio-request-001",
        },
    )
    replay = client.post(
        "/v1/credit-card/transactions/score",
        json=credit_card_transaction(value=50.0),
        headers={"Idempotency-Key": "metrics-score-001"},
    )
    metrics = client.get("/metrics")

    assert score.headers["X-Request-ID"] == "portfolio-request-001"
    assert replay.headers["X-Request-ID"]
    assert metrics.status_code == 200
    assert "text/plain" in metrics.headers["content-type"]
    assert "riskpulse_http_requests_total" in metrics.text
    assert 'route="/v1/credit-card/transactions/score"' in metrics.text
    assert 'replayed="false"' in metrics.text
    assert 'replayed="true"' in metrics.text
    assert "riskpulse_fraud_probability_bucket" in metrics.text


def test_replaces_unsafe_request_id(client: TestClient) -> None:
    response = client.get("/", headers={"X-Request-ID": "unsafe request id"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "unsafe request id"
    assert len(response.headers["X-Request-ID"]) == 32


def test_exception_metrics_use_route_template(client: TestClient) -> None:
    def fail_request(transaction_id: str) -> None:
        raise RuntimeError(f"failed transaction {transaction_id}")

    client.app.add_api_route(
        "/testing/fail/{transaction_id}",
        fail_request,
        methods=["GET"],
    )

    with pytest.raises(RuntimeError, match="failed transaction transaction-123"):
        client.get(
            "/testing/fail/transaction-123",
            headers={"X-Request-ID": "exception-request-001"},
        )

    metrics = client.get("/metrics")
    assert (
        'riskpulse_http_requests_total{method="GET",route="/testing/fail/{transaction_id}",'
        'status="500"} 1.0'
    ) in metrics.text
    assert 'route="/testing/fail/transaction-123"' not in metrics.text


def test_returns_insufficient_data_drift_report(client: TestClient) -> None:
    response = client.get("/v1/monitoring/drift")

    assert response.status_code == 200
    assert response.json() == {
        "generated_at": response.json()["generated_at"],
        "model_version": "creditcard-test-model",
        "reference_rows": 200,
        "current_rows": 0,
        "minimum_events": 20,
        "status": "insufficient_data",
        "drifted_features": 0,
        "features": [],
    }


def test_detects_drift_in_persisted_scoring_events(client: TestClient) -> None:
    for index in range(20):
        response = client.post(
            "/v1/credit-card/transactions/score",
            json={
                **credit_card_transaction(value=10_000.0 + index),
                "transaction_id": str(uuid4()),
            },
            headers={"Idempotency-Key": f"drift-event-{index}"},
        )
        assert response.status_code == 200

    report = client.get("/v1/monitoring/drift?limit=20")

    assert report.status_code == 200
    body = report.json()
    assert body["status"] == "critical"
    assert body["current_rows"] == 20
    assert body["drifted_features"] >= 1
    assert body["features"][0]["population_stability_index"] > 0.25


def test_monitoring_reference_is_optional(
    model_path,
    calibrated_model_path,
    tmp_path,
) -> None:
    from riskpulse.config import Settings
    from riskpulse.main import create_app
    from riskpulse.persistence.database import Database

    settings = Settings(
        model_path=model_path,
        calibrated_model_path=calibrated_model_path,
        monitoring_reference_path=tmp_path / "missing-reference.json",
        database_url=f"sqlite:///{tmp_path / 'monitoring-optional.db'}",
        drift_minimum_events=20,
    )
    database = Database(settings.database_url)
    database.create_schema()
    database.close()

    with TestClient(create_app(settings)) as optional_client:
        response = optional_client.get("/v1/monitoring/drift")

    assert response.status_code == 503
    assert response.json()["detail"].startswith("drift reference unavailable")


def test_rejects_invalid_drift_window(client: TestClient) -> None:
    response = client.get("/v1/monitoring/drift?limit=19")

    assert response.status_code == 422
