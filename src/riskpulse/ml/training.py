from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from riskpulse.ml.features import FEATURE_NAMES


@dataclass(frozen=True)
class TrainingResult:
    model_version: str
    output_path: Path
    validation_metrics: dict[str, float]
    positive_rate: float


def generate_demo_dataset(
    *,
    samples: int = 12_000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible demo data until a public dataset is integrated."""

    if samples < 1_000:
        raise ValueError("samples must be at least 1000")

    rng = np.random.default_rng(random_state)
    amount = np.clip(rng.lognormal(mean=4.2, sigma=1.15, size=samples), 1, 20_000)
    account_age_days = rng.integers(0, 3_650, size=samples)
    hour_of_day = rng.integers(0, 24, size=samples)
    is_international = rng.binomial(1, 0.12, size=samples)
    is_new_device = rng.binomial(1, 0.16, size=samples)
    failed_attempts = np.clip(rng.poisson(0.25, size=samples), 0, 8)
    transactions_1h = np.clip(rng.poisson(2.2, size=samples), 0, 20)
    distance_from_home = np.clip(rng.exponential(90, size=samples), 0, 5_000)

    log_odds = (
        -5.8
        + 0.48 * np.log1p(amount)
        + 0.95 * is_international
        + 1.25 * is_new_device
        + 0.58 * failed_attempts
        + 0.12 * np.maximum(transactions_1h - 3, 0)
        + 0.0018 * np.maximum(distance_from_home - 100, 0)
        + 0.65 * (account_age_days < 30)
        + 0.40 * (hour_of_day <= 4)
    )
    fraud_probability = 1.0 / (1.0 + np.exp(-log_odds))
    label = rng.binomial(1, fraud_probability)

    features = np.column_stack(
        [
            amount,
            account_age_days,
            hour_of_day,
            is_international,
            is_new_device,
            failed_attempts,
            transactions_1h,
            distance_from_home,
        ]
    ).astype(np.float64)
    return features, label.astype(np.int64)


def train_and_save_model(
    output_path: Path,
    *,
    samples: int = 12_000,
    random_state: int = 42,
) -> TrainingResult:
    features, label = generate_demo_dataset(samples=samples, random_state=random_state)
    x_train, x_validation, y_train, y_validation = train_test_split(
        features,
        label,
        test_size=0.20,
        random_state=random_state,
        stratify=label,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    probability = pipeline.predict_proba(x_validation)[:, 1]
    prediction = (probability >= 0.50).astype(np.int64)
    metrics = {
        "roc_auc": float(roc_auc_score(y_validation, probability)),
        "pr_auc": float(average_precision_score(y_validation, probability)),
        "precision_at_0_5": float(precision_score(y_validation, prediction, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_validation, prediction, zero_division=0)),
    }

    trained_at = datetime.now(UTC)
    model_version = f"baseline-{trained_at:%Y%m%d}-{uuid4().hex[:8]}"
    artifact = {
        "model": pipeline,
        "model_version": model_version,
        "trained_at": trained_at.isoformat(),
        "feature_names": list(FEATURE_NAMES),
        "validation_metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return TrainingResult(
        model_version=model_version,
        output_path=output_path,
        validation_metrics=metrics,
        positive_rate=float(label.mean()),
    )
