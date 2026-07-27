from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss

from riskpulse.ml.benchmarking import (
    CostPolicy,
    build_hist_gradient_boosting_model,
    evaluate_scores,
    select_cost_optimal_threshold,
)
from riskpulse.ml.datasets import (
    AMOUNT_COLUMN,
    FEATURE_COLUMNS,
    CreditCardFraudDataset,
    TemporalSplit,
    temporal_split,
)


@dataclass(frozen=True)
class CalibrationPartition:
    x_fit: pd.DataFrame
    y_fit: pd.Series
    x_calibration: pd.DataFrame
    y_calibration: pd.Series


@dataclass(frozen=True)
class ReliabilityMetrics:
    brier_score: float
    log_loss: float
    expected_calibration_error: float


@dataclass(frozen=True)
class CalibratedTrainingResult:
    artifact: dict[str, Any]
    report: dict[str, Any]


def chronological_calibration_partition(
    split: TemporalSplit,
    *,
    calibration_fraction: float = 0.20,
) -> CalibrationPartition:
    if not 0 < calibration_fraction < 0.50:
        raise ValueError("calibration fraction must be between zero and 0.50")

    rows = len(split.y_train)
    if rows < 20:
        raise ValueError("at least 20 training rows are required for calibration")
    fit_end = int(rows * (1 - calibration_fraction))
    y_fit = split.y_train.iloc[:fit_end].copy()
    y_calibration = split.y_train.iloc[fit_end:].copy()
    if y_fit.nunique() != 2 or y_calibration.nunique() != 2:
        raise ValueError("fit and calibration periods must each contain both classes")

    return CalibrationPartition(
        x_fit=split.x_train.iloc[:fit_end].copy(),
        y_fit=y_fit,
        x_calibration=split.x_train.iloc[fit_end:].copy(),
        y_calibration=y_calibration,
    )


def reliability_metrics(
    target: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    *,
    bins: int = 10,
) -> ReliabilityMetrics:
    labels = np.asarray(target, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("target and probabilities must be equal-length one-dimensional arrays")
    if len(labels) == 0:
        raise ValueError("at least one probability is required")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("target must contain only binary labels")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be finite values between zero and one")
    if bins < 2:
        raise ValueError("at least two calibration bins are required")

    bin_ids = np.minimum((scores * bins).astype(int), bins - 1)
    expected_calibration_error = 0.0
    for bin_id in range(bins):
        members = bin_ids == bin_id
        if members.any():
            expected_calibration_error += float(
                members.mean() * abs(scores[members].mean() - labels[members].mean())
            )

    return ReliabilityMetrics(
        brier_score=float(brier_score_loss(labels, scores)),
        log_loss=float(log_loss(labels, scores, labels=[0, 1])),
        expected_calibration_error=expected_calibration_error,
    )


def train_calibrated_model(
    dataset: CreditCardFraudDataset,
    *,
    policy: CostPolicy | None = None,
    calibration_fraction: float = 0.20,
    base_model: Any | None = None,
) -> CalibratedTrainingResult:
    """Train, calibrate, threshold, and evaluate without test-set model selection."""

    if policy is None:
        policy = CostPolicy()
    split = temporal_split(dataset)
    partition = chronological_calibration_partition(
        split,
        calibration_fraction=calibration_fraction,
    )
    model = base_model or build_hist_gradient_boosting_model()
    model.fit(partition.x_fit, partition.y_fit)

    uncalibrated_validation_probabilities = model.predict_proba(split.x_validation)[:, 1]
    calibrator = CalibratedClassifierCV(
        FrozenEstimator(model),
        method="sigmoid",
    )
    calibrator.fit(partition.x_calibration, partition.y_calibration)

    validation_probabilities = calibrator.predict_proba(split.x_validation)[:, 1]
    threshold = select_cost_optimal_threshold(
        split.y_validation,
        validation_probabilities,
        split.x_validation[AMOUNT_COLUMN],
        policy,
    )
    validation_metrics = evaluate_scores(
        split.y_validation,
        validation_probabilities,
        split.x_validation[AMOUNT_COLUMN],
        threshold=threshold.threshold,
        policy=policy,
    )
    test_probabilities = calibrator.predict_proba(split.x_test)[:, 1]
    test_metrics = evaluate_scores(
        split.y_test,
        test_probabilities,
        split.x_test[AMOUNT_COLUMN],
        threshold=threshold.threshold,
        policy=policy,
    )

    validation_uncalibrated = reliability_metrics(
        split.y_validation,
        uncalibrated_validation_probabilities,
    )
    validation_calibrated = reliability_metrics(
        split.y_validation,
        validation_probabilities,
    )
    test_reliability = reliability_metrics(split.y_test, test_probabilities)

    trained_at = datetime.now(UTC)
    model_version = f"creditcard-hgb-sigmoid-{trained_at:%Y%m%d}-{uuid4().hex[:8]}"
    artifact = {
        "artifact_schema_version": "1.0",
        "model": calibrator,
        "model_version": model_version,
        "trained_at": trained_at.isoformat(),
        "model_type": "calibrated_hist_gradient_boosting",
        "calibration_method": "sigmoid",
        "feature_names": list(FEATURE_COLUMNS),
        "decision_threshold": threshold.threshold,
        "dataset_id": dataset.summary.openml_data_id,
        "cost_policy": asdict(policy),
        "validation_metrics": asdict(validation_metrics),
        "test_metrics": asdict(test_metrics),
    }
    report = {
        "model_version": model_version,
        "trained_at": trained_at.isoformat(),
        "model_type": artifact["model_type"],
        "calibration_method": artifact["calibration_method"],
        "dataset": asdict(dataset.summary),
        "chronological_partitions": {
            "fit": len(partition.y_fit),
            "calibration": len(partition.y_calibration),
            "validation": len(split.y_validation),
            "test": len(split.y_test),
        },
        "cost_policy": asdict(policy),
        "selected_threshold": threshold.threshold,
        "validation": asdict(validation_metrics),
        "test": asdict(test_metrics),
        "reliability": {
            "validation_uncalibrated": asdict(validation_uncalibrated),
            "validation_calibrated": asdict(validation_calibrated),
            "test_calibrated": asdict(test_reliability),
            "validation_brier_improvement": (
                validation_uncalibrated.brier_score - validation_calibrated.brier_score
            ),
            "validation_log_loss_improvement": (
                validation_uncalibrated.log_loss - validation_calibrated.log_loss
            ),
        },
    }
    return CalibratedTrainingResult(artifact=artifact, report=report)
