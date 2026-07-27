from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from riskpulse.config import Settings
from riskpulse.main import create_app
from riskpulse.persistence.database import Database


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
    assert body["idempotency_key"].startswith("generated:")
    assert body["idempotency_replayed"] is False
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


def test_replays_same_idempotent_request(client: TestClient) -> None:
    headers = {"Idempotency-Key": "checkout-attempt-001"}
    first = client.post(
        "/v1/credit-card/transactions/score",
        json=credit_card_transaction(),
        headers=headers,
    )
    second = client.post(
        "/v1/credit-card/transactions/score",
        json=credit_card_transaction(),
        headers=headers,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["transaction_id"] == second.json()["transaction_id"]
    assert first.json()["idempotency_replayed"] is False
    assert second.json()["idempotency_replayed"] is True


def test_rejects_idempotency_key_reuse_with_different_payload(
    client: TestClient,
) -> None:
    headers = {"Idempotency-Key": "checkout-attempt-002"}
    first = client.post(
        "/v1/credit-card/transactions/score",
        json=credit_card_transaction(),
        headers=headers,
    )
    changed = credit_card_transaction()
    changed["amount"] = 999.0
    second = client.post(
        "/v1/credit-card/transactions/score",
        json=changed,
        headers=headers,
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert "different feature payload" in second.json()["detail"]


def test_rejects_transaction_reuse_with_different_idempotency_key(
    client: TestClient,
) -> None:
    payload = credit_card_transaction()
    payload["transaction_id"] = str(uuid4())
    first = client.post(
        "/v1/credit-card/transactions/score",
        json=payload,
        headers={"Idempotency-Key": "transaction-key-001"},
    )
    second = client.post(
        "/v1/credit-card/transactions/score",
        json=payload,
        headers={"Idempotency-Key": "transaction-key-002"},
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert "transaction_id" in second.json()["detail"]


def test_rejects_blank_idempotency_key(client: TestClient) -> None:
    response = client.post(
        "/v1/credit-card/transactions/score",
        json=credit_card_transaction(),
        headers={"Idempotency-Key": "   "},
    )

    assert response.status_code == 422


def test_returns_persisted_audit_event(client: TestClient) -> None:
    score = client.post(
        "/v1/credit-card/transactions/score",
        json=credit_card_transaction(),
        headers={"Idempotency-Key": "audit-event-001"},
    )

    response = client.get(f"/v1/credit-card/transactions/{score.json()['transaction_id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["idempotency_key"] == "audit-event-001"
    assert body["features"]["Time"] == 86_400.0
    assert body["features"]["Amount"] == 125.5
    assert len(body["features"]) == 30
    assert body["feedback"] is None


def test_returns_not_found_for_unknown_audit_event(client: TestClient) -> None:
    response = client.get(f"/v1/credit-card/transactions/{uuid4()}")

    assert response.status_code == 404


def test_lists_pending_reviews_and_records_feedback(client: TestClient) -> None:
    score = client.post(
        "/v1/credit-card/transactions/score",
        json=credit_card_transaction(),
        headers={"Idempotency-Key": "review-event-001"},
    )
    transaction_id = score.json()["transaction_id"]

    pending = client.get("/v1/credit-card/reviews")
    feedback_payload = {
        "outcome": "legitimate",
        "reviewer_id": "analyst-42",
        "notes": "Verified with the customer.",
    }
    feedback = client.post(
        f"/v1/credit-card/transactions/{transaction_id}/feedback",
        json=feedback_payload,
    )
    repeated = client.post(
        f"/v1/credit-card/transactions/{transaction_id}/feedback",
        json=feedback_payload,
    )
    remaining = client.get("/v1/credit-card/reviews")

    assert pending.status_code == 200
    assert [event["transaction_id"] for event in pending.json()] == [transaction_id]
    assert feedback.status_code == 200
    assert feedback.json()["feedback"]["outcome"] == "legitimate"
    assert feedback.json()["feedback"]["reviewer_id"] == "analyst-42"
    assert feedback.json()["feedback"]["reviewed_at"]
    assert repeated.status_code == 200
    assert remaining.json() == []


def test_rejects_conflicting_or_unknown_feedback(client: TestClient) -> None:
    score = client.post(
        "/v1/credit-card/transactions/score",
        json=credit_card_transaction(),
        headers={"Idempotency-Key": "review-event-002"},
    )
    transaction_id = score.json()["transaction_id"]
    first = client.post(
        f"/v1/credit-card/transactions/{transaction_id}/feedback",
        json={"outcome": "legitimate", "reviewer_id": "analyst-1"},
    )
    conflict = client.post(
        f"/v1/credit-card/transactions/{transaction_id}/feedback",
        json={"outcome": "confirmed_fraud", "reviewer_id": "analyst-2"},
    )
    missing = client.post(
        f"/v1/credit-card/transactions/{uuid4()}/feedback",
        json={"outcome": "legitimate", "reviewer_id": "analyst-1"},
    )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert missing.status_code == 404


def test_validates_review_queue_and_feedback_inputs(client: TestClient) -> None:
    invalid_limit = client.get("/v1/credit-card/reviews?limit=0")
    invalid_feedback = client.post(
        f"/v1/credit-card/transactions/{uuid4()}/feedback",
        json={"outcome": "legitimate", "reviewer_id": ""},
    )

    assert invalid_limit.status_code == 422
    assert invalid_feedback.status_code == 422


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
        database_url=f"sqlite:///{tmp_path / 'optional.db'}",
    )
    database = Database(settings.database_url)
    database.create_schema()
    database.close()

    with TestClient(create_app(settings)) as client:
        unavailable = client.get("/v1/credit-card/model")
        ready = client.get("/health/ready")

    assert unavailable.status_code == 503
    assert unavailable.json()["detail"].startswith("calibrated real-data model")
    assert ready.status_code == 200
