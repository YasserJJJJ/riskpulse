from fastapi.testclient import TestClient


def low_risk_transaction() -> dict[str, object]:
    return {
        "amount": 18.5,
        "account_age_days": 1_800,
        "hour_of_day": 14,
        "is_international": False,
        "is_new_device": False,
        "failed_attempts_24h": 0,
        "transactions_1h": 1,
        "distance_from_home_km": 4.0,
    }


def high_risk_transaction() -> dict[str, object]:
    return {
        "amount": 4_800.0,
        "account_age_days": 3,
        "hour_of_day": 2,
        "is_international": True,
        "is_new_device": True,
        "failed_attempts_24h": 5,
        "transactions_1h": 14,
        "distance_from_home_km": 1_200.0,
    }


def test_root_points_to_documentation(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["documentation"] == "/docs"


def test_health_endpoints(client: TestClient) -> None:
    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_scores_transaction(client: TestClient) -> None:
    response = client.post("/v1/transactions/score", json=low_risk_transaction())

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["decision"] in {"approve", "manual_review", "decline"}
    assert body["model_version"].startswith("baseline-")
    assert body["transaction_id"]
    assert body["scored_at"]


def test_high_risk_transaction_scores_above_low_risk(client: TestClient) -> None:
    low_response = client.post(
        "/v1/transactions/score",
        json=low_risk_transaction(),
    )
    high_response = client.post(
        "/v1/transactions/score",
        json=high_risk_transaction(),
    )

    low_body = low_response.json()
    high_body = high_response.json()
    assert high_body["risk_score"] > low_body["risk_score"]
    assert high_body["decision"] == "decline"
    assert "new_device" in high_body["reasons"]
    assert "high_transaction_velocity" in high_body["reasons"]


def test_rejects_invalid_transaction(client: TestClient) -> None:
    transaction = low_risk_transaction()
    transaction["amount"] = -1

    response = client.post("/v1/transactions/score", json=transaction)

    assert response.status_code == 422


def test_returns_model_metadata(client: TestClient) -> None:
    response = client.get("/v1/model")

    assert response.status_code == 200
    body = response.json()
    assert len(body["feature_names"]) == 8
    assert set(body["validation_metrics"]) == {
        "roc_auc",
        "pr_auc",
        "precision_at_0_5",
        "recall_at_0_5",
    }
