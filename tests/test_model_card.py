from riskpulse.ml.model_card import render_model_card


def test_renders_measured_model_card_sections() -> None:
    card = render_model_card(sample_report())

    assert "# RiskPulse Credit-Card Fraud Model Card" in card
    assert "model-123" in card
    assert "| Fit | 100 |" in card
    assert "| PR-AUC | 0.8000 | 0.7000 |" in card
    assert "not validated for real financial decisions" in card
    assert "Temporal drift" not in card
    assert "temporal" in card.lower()


def sample_report() -> dict:
    metrics = {
        "roc_auc": 0.9,
        "pr_auc": 0.8,
        "precision": 0.7,
        "recall": 0.6,
        "f1": 0.65,
        "alert_rate": 0.01,
        "fraud_amount_capture_rate": 0.75,
        "estimated_cost": 100.0,
        "cost_reduction_vs_no_alert": 300.0,
    }
    test_metrics = {**metrics, "pr_auc": 0.7}
    reliability = {
        "brier_score": 0.01,
        "log_loss": 0.05,
        "expected_calibration_error": 0.02,
    }
    return {
        "model_version": "model-123",
        "selected_threshold": 0.2,
        "dataset": {
            "openml_data_id": 1597,
            "feature_count": 30,
            "transactions": 200,
            "frauds": 20,
            "fraud_rate": 0.1,
        },
        "chronological_partitions": {
            "fit": 100,
            "calibration": 40,
            "validation": 30,
            "test": 30,
        },
        "validation": metrics,
        "test": test_metrics,
        "reliability": {
            "validation_uncalibrated": reliability,
            "validation_calibrated": reliability,
            "test_calibrated": reliability,
        },
        "cost_policy": {
            "false_positive_review_cost": 5.0,
            "missed_fraud_amount_multiplier": 1.0,
        },
    }
