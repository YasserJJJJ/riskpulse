from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from riskpulse.ml.benchmarking import CostPolicy
from riskpulse.ml.calibration import (
    chronological_calibration_partition,
    reliability_metrics,
    train_calibrated_model,
)
from riskpulse.ml.datasets import (
    FEATURE_COLUMNS,
    CreditCardFraudDataset,
    DatasetSummary,
    temporal_split,
)


def test_calibration_partition_preserves_chronology() -> None:
    dataset = sample_dataset()
    split = temporal_split(dataset)

    partition = chronological_calibration_partition(split, calibration_fraction=0.25)

    assert len(partition.y_fit) == 105
    assert len(partition.y_calibration) == 35
    assert partition.x_fit["Time"].max() < partition.x_calibration["Time"].min()


@pytest.mark.parametrize("fraction", [0.0, 0.50])
def test_rejects_invalid_calibration_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="between zero and 0.50"):
        chronological_calibration_partition(
            temporal_split(sample_dataset()),
            calibration_fraction=fraction,
        )


def test_measures_probability_reliability() -> None:
    metrics = reliability_metrics(
        target=np.array([0, 0, 1, 1]),
        probabilities=np.array([0.1, 0.2, 0.8, 0.9]),
        bins=5,
    )

    assert metrics.brier_score == pytest.approx(0.025)
    assert metrics.log_loss == pytest.approx(0.164252, rel=1e-5)
    assert metrics.expected_calibration_error == pytest.approx(0.15)


def test_trains_serializable_calibrated_artifact(tmp_path: Path) -> None:
    result = train_calibrated_model(
        sample_dataset(rows=400),
        policy=CostPolicy(false_positive_review_cost=10.0),
        base_model=LogisticRegression(max_iter=500),
    )
    artifact_path = tmp_path / "calibrated.joblib"
    joblib.dump(result.artifact, artifact_path)
    loaded = joblib.load(artifact_path)

    assert loaded["artifact_schema_version"] == "1.0"
    assert loaded["model_type"] == "calibrated_hist_gradient_boosting"
    assert loaded["calibration_method"] == "sigmoid"
    assert loaded["feature_names"] == list(FEATURE_COLUMNS)
    assert 0 <= loaded["decision_threshold"] <= 1
    assert loaded["model"].predict_proba(sample_dataset(20).features).shape == (20, 2)
    assert result.report["chronological_partitions"] == {
        "fit": 224,
        "calibration": 56,
        "validation": 60,
        "test": 60,
    }
    assert result.report["reliability"]["test_calibrated"]["brier_score"] >= 0


def sample_dataset(rows: int = 200) -> CreditCardFraudDataset:
    rng = np.random.default_rng(42)
    labels = np.zeros(rows, dtype=np.int8)
    labels[::10] = 1
    signal = labels + rng.normal(0, 0.35, rows)
    data: dict[str, np.ndarray] = {
        "Time": np.arange(rows, dtype=float),
        "Amount": rng.uniform(5, 500, rows),
    }
    for index, feature_name in enumerate(FEATURE_COLUMNS[1:-1], start=1):
        data[feature_name] = signal if index == 1 else rng.normal(size=rows)
    features = pd.DataFrame(data).loc[:, FEATURE_COLUMNS]
    target = pd.Series(labels, name="Class", dtype="int8")
    return CreditCardFraudDataset(
        features=features,
        target=target,
        summary=DatasetSummary(
            openml_data_id=1597,
            transactions=rows,
            frauds=int(target.sum()),
            fraud_rate=float(target.mean()),
            feature_count=len(FEATURE_COLUMNS),
            time_start_seconds=0,
            time_end_seconds=rows - 1,
        ),
    )
