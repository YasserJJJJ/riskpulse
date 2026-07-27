from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from riskpulse.ml.datasets import AMOUNT_COLUMN, TemporalSplit


@dataclass(frozen=True)
class CostPolicy:
    """Explicit business assumptions used to turn errors into comparable cost."""

    false_positive_review_cost: float = 5.0
    missed_fraud_amount_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.false_positive_review_cost < 0:
            raise ValueError("false-positive review cost cannot be negative")
        if self.missed_fraud_amount_multiplier < 0:
            raise ValueError("missed-fraud amount multiplier cannot be negative")


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    estimated_cost: float
    alerts: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class EvaluationMetrics:
    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    alert_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    fraud_amount_total: float
    fraud_amount_captured: float
    fraud_amount_capture_rate: float
    estimated_cost: float
    no_alert_cost: float
    cost_reduction_vs_no_alert: float


def _validated_arrays(
    target: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    amounts: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(target, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=float)
    transaction_amounts = np.asarray(amounts, dtype=float)

    if labels.ndim != 1 or scores.ndim != 1 or transaction_amounts.ndim != 1:
        raise ValueError("target, probabilities, and amounts must be one-dimensional")
    if not (len(labels) == len(scores) == len(transaction_amounts)):
        raise ValueError("target, probabilities, and amounts must have equal lengths")
    if len(labels) == 0:
        raise ValueError("at least one prediction is required")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("target must contain only binary labels")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be finite values between zero and one")
    if not np.isfinite(transaction_amounts).all() or (transaction_amounts < 0).any():
        raise ValueError("amounts must be finite and non-negative")

    return labels, scores, transaction_amounts


def estimate_business_cost(
    target: pd.Series | np.ndarray,
    predictions: pd.Series | np.ndarray,
    amounts: pd.Series | np.ndarray,
    policy: CostPolicy,
) -> float:
    labels = np.asarray(target, dtype=np.int8)
    predicted_labels = np.asarray(predictions, dtype=np.int8)
    transaction_amounts = np.asarray(amounts, dtype=float)
    if not (len(labels) == len(predicted_labels) == len(transaction_amounts)):
        raise ValueError("target, predictions, and amounts must have equal lengths")

    false_positives = (labels == 0) & (predicted_labels == 1)
    false_negatives = (labels == 1) & (predicted_labels == 0)
    return float(
        false_positives.sum() * policy.false_positive_review_cost
        + transaction_amounts[false_negatives].sum() * policy.missed_fraud_amount_multiplier
    )


def select_cost_optimal_threshold(
    target: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    amounts: pd.Series | np.ndarray,
    policy: CostPolicy,
) -> ThresholdSelection:
    """Choose a threshold on validation data without scanning the held-out test set."""

    labels, scores, transaction_amounts = _validated_arrays(target, probabilities, amounts)
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ordered_amounts = transaction_amounts[order]

    cumulative_false_positives = np.cumsum(ordered_labels == 0)
    cumulative_captured_fraud = np.cumsum(np.where(ordered_labels == 1, ordered_amounts, 0.0))
    total_fraud_amount = float(ordered_amounts[ordered_labels == 1].sum())
    candidate_ends = np.append(
        np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]),
        len(ordered_scores) - 1,
    )

    best = ThresholdSelection(
        threshold=float(np.nextafter(ordered_scores[0], np.inf)),
        estimated_cost=total_fraud_amount * policy.missed_fraud_amount_multiplier,
        alerts=0,
        false_positives=0,
        false_negatives=int((labels == 1).sum()),
    )
    total_frauds = int((labels == 1).sum())

    for end_index in candidate_ends:
        alerts = int(end_index + 1)
        false_positives = int(cumulative_false_positives[end_index])
        captured_frauds = int(ordered_labels[:alerts].sum())
        missed_fraud_amount = total_fraud_amount - float(cumulative_captured_fraud[end_index])
        estimated_cost = (
            false_positives * policy.false_positive_review_cost
            + missed_fraud_amount * policy.missed_fraud_amount_multiplier
        )
        if estimated_cost < best.estimated_cost:
            best = ThresholdSelection(
                threshold=float(ordered_scores[end_index]),
                estimated_cost=float(estimated_cost),
                alerts=alerts,
                false_positives=false_positives,
                false_negatives=total_frauds - captured_frauds,
            )

    return best


def evaluate_scores(
    target: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    amounts: pd.Series | np.ndarray,
    *,
    threshold: float,
    policy: CostPolicy,
) -> EvaluationMetrics:
    labels, scores, transaction_amounts = _validated_arrays(target, probabilities, amounts)
    if len(np.unique(labels)) != 2:
        raise ValueError("evaluation requires both fraud and legitimate transactions")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    predictions = (scores >= threshold).astype(np.int8)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    fraud_mask = labels == 1
    fraud_amount_total = float(transaction_amounts[fraud_mask].sum())
    fraud_amount_captured = float(transaction_amounts[fraud_mask & (predictions == 1)].sum())
    no_alert_cost = fraud_amount_total * policy.missed_fraud_amount_multiplier
    estimated_cost = estimate_business_cost(labels, predictions, transaction_amounts, policy)

    return EvaluationMetrics(
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(labels, scores)),
        pr_auc=float(average_precision_score(labels, scores)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        alert_rate=float(predictions.mean()),
        true_positives=int(true_positives),
        false_positives=int(false_positives),
        true_negatives=int(true_negatives),
        false_negatives=int(false_negatives),
        fraud_amount_total=fraud_amount_total,
        fraud_amount_captured=fraud_amount_captured,
        fraud_amount_capture_rate=fraud_amount_captured / fraud_amount_total,
        estimated_cost=estimated_cost,
        no_alert_cost=no_alert_cost,
        cost_reduction_vs_no_alert=no_alert_cost - estimated_cost,
    )


def build_hist_gradient_boosting_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        early_stopping=False,
        l2_regularization=1.0,
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=31,
        random_state=42,
    )


def build_candidate_models() -> dict[str, Any]:
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": build_hist_gradient_boosting_model(),
    }


def benchmark_models(
    split: TemporalSplit,
    *,
    policy: CostPolicy | None = None,
) -> dict[str, Any]:
    """Fit on train, select model/threshold on validation, evaluate test once."""

    if policy is None:
        policy = CostPolicy()

    candidates: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}

    for name, model in build_candidate_models().items():
        fit_started = perf_counter()
        model.fit(split.x_train, split.y_train)
        fit_seconds = perf_counter() - fit_started

        validation_probabilities = model.predict_proba(split.x_validation)[:, 1]
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
        candidates[name] = {
            "fit_seconds": round(fit_seconds, 6),
            "validation": asdict(validation_metrics),
        }
        fitted_models[name] = model

    selected_model_name = min(
        candidates,
        key=lambda name: (
            candidates[name]["validation"]["estimated_cost"],
            -candidates[name]["validation"]["pr_auc"],
            name,
        ),
    )
    selected_threshold = candidates[selected_model_name]["validation"]["threshold"]
    selected_model = fitted_models[selected_model_name]
    test_probabilities = selected_model.predict_proba(split.x_test)[:, 1]
    test_metrics = evaluate_scores(
        split.y_test,
        test_probabilities,
        split.x_test[AMOUNT_COLUMN],
        threshold=selected_threshold,
        policy=policy,
    )

    return {
        "selection_rule": "lowest validation estimated cost; PR-AUC breaks ties",
        "cost_policy": asdict(policy),
        "candidates": candidates,
        "selected_model": selected_model_name,
        "selected_threshold": selected_threshold,
        "test": asdict(test_metrics),
    }
