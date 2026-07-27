import numpy as np
import pandas as pd
import pytest

from riskpulse.ml.benchmarking import (
    CostPolicy,
    benchmark_models,
    build_candidate_models,
    estimate_business_cost,
    evaluate_scores,
    select_cost_optimal_threshold,
)
from riskpulse.ml.datasets import TemporalSplit


def test_selects_threshold_with_lowest_validation_cost() -> None:
    selection = select_cost_optimal_threshold(
        target=np.array([0, 1, 0, 1]),
        probabilities=np.array([0.9, 0.8, 0.7, 0.1]),
        amounts=np.array([20.0, 100.0, 20.0, 10.0]),
        policy=CostPolicy(false_positive_review_cost=30.0),
    )

    assert selection.threshold == pytest.approx(0.8)
    assert selection.estimated_cost == pytest.approx(40.0)
    assert selection.alerts == 2
    assert selection.false_positives == 1
    assert selection.false_negatives == 1


def test_threshold_tie_prefers_fewer_alerts() -> None:
    selection = select_cost_optimal_threshold(
        target=np.array([0, 1]),
        probabilities=np.array([0.9, 0.8]),
        amounts=np.array([1.0, 5.0]),
        policy=CostPolicy(false_positive_review_cost=5.0),
    )

    assert selection.threshold > 0.9
    assert selection.estimated_cost == pytest.approx(5.0)
    assert selection.alerts == 0


def test_evaluates_ranking_classification_and_business_metrics() -> None:
    metrics = evaluate_scores(
        target=np.array([0, 0, 1, 1]),
        probabilities=np.array([0.1, 0.8, 0.7, 0.9]),
        amounts=np.array([10.0, 20.0, 100.0, 50.0]),
        threshold=0.75,
        policy=CostPolicy(false_positive_review_cost=5.0),
    )

    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.pr_auc == pytest.approx(5 / 6)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.alert_rate == pytest.approx(0.5)
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.fraud_amount_captured == pytest.approx(50.0)
    assert metrics.estimated_cost == pytest.approx(105.0)
    assert metrics.cost_reduction_vs_no_alert == pytest.approx(45.0)


def test_estimates_business_cost() -> None:
    cost = estimate_business_cost(
        target=np.array([0, 1, 1]),
        predictions=np.array([1, 0, 1]),
        amounts=np.array([25.0, 80.0, 20.0]),
        policy=CostPolicy(
            false_positive_review_cost=7.0,
            missed_fraud_amount_multiplier=1.5,
        ),
    )

    assert cost == pytest.approx(127.0)


@pytest.mark.parametrize(
    ("target", "probabilities", "amounts", "message"),
    [
        ([0, 1], [0.2], [10.0, 20.0], "equal lengths"),
        ([0, 2], [0.2, 0.8], [10.0, 20.0], "binary labels"),
        ([0, 1], [0.2, 1.2], [10.0, 20.0], "between zero and one"),
        ([0, 1], [0.2, 0.8], [10.0, -20.0], "non-negative"),
    ],
)
def test_rejects_invalid_threshold_inputs(
    target: list[int],
    probabilities: list[float],
    amounts: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_cost_optimal_threshold(
            target,
            probabilities,
            amounts,
            CostPolicy(),
        )


def test_builds_three_reproducible_candidate_models() -> None:
    models = build_candidate_models()

    assert set(models) == {
        "dummy_prior",
        "logistic_regression",
        "hist_gradient_boosting",
    }


def test_benchmark_selects_on_validation_and_reports_test_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def __init__(self, probability_column: str) -> None:
            self.probability_column = probability_column

        def fit(self, _features: pd.DataFrame, _target: pd.Series) -> "FakeModel":
            return self

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            probabilities = features[self.probability_column].to_numpy()
            return np.column_stack([1 - probabilities, probabilities])

    monkeypatch.setattr(
        "riskpulse.ml.benchmarking.build_candidate_models",
        lambda: {
            "weaker": FakeModel("weak_score"),
            "stronger": FakeModel("strong_score"),
        },
    )
    split = _sample_split()

    report = benchmark_models(
        split,
        policy=CostPolicy(false_positive_review_cost=20.0),
    )

    assert report["selected_model"] == "stronger"
    assert set(report["candidates"]) == {"weaker", "stronger"}
    assert report["test"]["pr_auc"] == pytest.approx(1.0)
    assert "test" not in report["candidates"]["weaker"]


def _sample_split() -> TemporalSplit:
    def features(
        amounts: list[float],
        weak_scores: list[float],
        strong_scores: list[float],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Amount": amounts,
                "weak_score": weak_scores,
                "strong_score": strong_scores,
            }
        )

    train_target = pd.Series([0, 1, 0, 1], dtype="int8")
    validation_target = pd.Series([0, 1, 0, 1], dtype="int8")
    test_target = pd.Series([0, 1, 0, 1], dtype="int8")
    return TemporalSplit(
        x_train=features([10, 50, 10, 50], [0.2, 0.8, 0.3, 0.7], [0.1, 0.9, 0.2, 0.8]),
        y_train=train_target,
        x_validation=features(
            [10, 100, 10, 100],
            [0.6, 0.7, 0.5, 0.4],
            [0.1, 0.9, 0.2, 0.8],
        ),
        y_validation=validation_target,
        x_test=features(
            [10, 80, 10, 90],
            [0.6, 0.7, 0.5, 0.4],
            [0.1, 0.9, 0.2, 0.8],
        ),
        y_test=test_target,
    )
