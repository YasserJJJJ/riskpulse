from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskpulse.domain.schemas import DriftSeverity

REFERENCE_SCHEMA_VERSION = "1.0"
PREDICTION_FEATURE = "fraud_probability"


class DriftReferenceError(ValueError):
    """Raised when a monitoring reference artifact is invalid."""


@dataclass(frozen=True)
class FeatureReference:
    bin_edges: list[float]
    expected_proportions: list[float]
    mean: float


@dataclass(frozen=True)
class DriftReference:
    schema_version: str
    generated_at: str
    model_version: str
    dataset_id: int
    rows: int
    features: dict[str, FeatureReference]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> DriftReference:
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DriftReferenceError(f"could not read drift reference at {path}") from error
        if not isinstance(payload, dict):
            raise DriftReferenceError("drift reference must be a JSON object")
        try:
            raw_features = payload["features"]
            if not isinstance(raw_features, dict) or not raw_features:
                raise TypeError("features must be a non-empty object")
            features = {
                str(name): FeatureReference(
                    bin_edges=[float(value) for value in reference["bin_edges"]],
                    expected_proportions=[
                        float(value) for value in reference["expected_proportions"]
                    ],
                    mean=float(reference["mean"]),
                )
                for name, reference in raw_features.items()
            }
            reference = cls(
                schema_version=str(payload["schema_version"]),
                generated_at=str(payload["generated_at"]),
                model_version=str(payload["model_version"]),
                dataset_id=int(payload["dataset_id"]),
                rows=int(payload["rows"]),
                features=features,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DriftReferenceError(f"invalid drift reference: {error}") from error
        reference.validate()
        return reference

    def validate(self) -> None:
        if self.schema_version != REFERENCE_SCHEMA_VERSION:
            raise DriftReferenceError("unsupported drift reference schema version")
        if self.rows <= 0:
            raise DriftReferenceError("reference rows must be positive")
        if not self.model_version:
            raise DriftReferenceError("reference model_version cannot be empty")
        for name, reference in self.features.items():
            if not name:
                raise DriftReferenceError("reference feature names cannot be empty")
            if len(reference.expected_proportions) != len(reference.bin_edges) + 1:
                raise DriftReferenceError(
                    f"feature {name} has inconsistent bin edges and proportions"
                )
            values = np.asarray(
                [*reference.bin_edges, *reference.expected_proportions, reference.mean],
                dtype=float,
            )
            if not np.isfinite(values).all():
                raise DriftReferenceError(f"feature {name} contains non-finite values")
            if reference.bin_edges != sorted(set(reference.bin_edges)):
                raise DriftReferenceError(f"feature {name} bin edges must be unique and sorted")
            if any(value < 0 for value in reference.expected_proportions):
                raise DriftReferenceError(f"feature {name} has a negative proportion")
            if not np.isclose(sum(reference.expected_proportions), 1.0):
                raise DriftReferenceError(f"feature {name} proportions must sum to one")


def _proportions(values: np.ndarray, edges: list[float]) -> list[float]:
    counts, _ = np.histogram(values, bins=[-np.inf, *edges, np.inf])
    return (counts / counts.sum()).astype(float).tolist()


def build_drift_reference(
    frame: pd.DataFrame,
    *,
    model_version: str,
    dataset_id: int,
    bins: int = 10,
    generated_at: datetime | None = None,
) -> DriftReference:
    if frame.empty:
        raise ValueError("reference frame cannot be empty")
    if bins < 2:
        raise ValueError("at least two reference bins are required")
    if not model_version:
        raise ValueError("model_version cannot be empty")

    features: dict[str, FeatureReference] = {}
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    for name in frame.columns:
        values = frame[name].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"reference feature {name} contains non-finite values")
        edges = np.unique(np.quantile(values, quantiles)).astype(float).tolist()
        features[str(name)] = FeatureReference(
            bin_edges=edges,
            expected_proportions=_proportions(values, edges),
            mean=float(values.mean()),
        )

    timestamp = generated_at or datetime.now(UTC)
    reference = DriftReference(
        schema_version=REFERENCE_SCHEMA_VERSION,
        generated_at=timestamp.isoformat(),
        model_version=model_version,
        dataset_id=dataset_id,
        rows=len(frame),
        features=features,
    )
    reference.validate()
    return reference


def population_stability_index(
    expected: list[float],
    current: list[float],
    *,
    epsilon: float = 1e-6,
) -> float:
    if len(expected) != len(current) or not expected:
        raise ValueError("expected and current distributions must have equal non-zero length")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    expected_values = np.asarray(expected, dtype=float)
    current_values = np.asarray(current, dtype=float)
    if (
        not np.isfinite(expected_values).all()
        or not np.isfinite(current_values).all()
        or (expected_values < 0).any()
        or (current_values < 0).any()
    ):
        raise ValueError("distributions must contain finite non-negative values")
    if not np.isclose(expected_values.sum(), 1.0) or not np.isclose(current_values.sum(), 1.0):
        raise ValueError("distributions must each sum to one")

    expected_smoothed = np.clip(expected_values, epsilon, None)
    current_smoothed = np.clip(current_values, epsilon, None)
    expected_smoothed /= expected_smoothed.sum()
    current_smoothed /= current_smoothed.sum()
    return float(
        np.sum(
            (current_smoothed - expected_smoothed) * np.log(current_smoothed / expected_smoothed)
        )
    )


def _severity(
    psi: float,
    *,
    warning_threshold: float,
    critical_threshold: float,
) -> DriftSeverity:
    if psi >= critical_threshold:
        return DriftSeverity.CRITICAL
    if psi >= warning_threshold:
        return DriftSeverity.WARNING
    return DriftSeverity.STABLE


def calculate_drift(
    reference: DriftReference,
    current: pd.DataFrame,
    *,
    minimum_events: int,
    warning_threshold: float,
    critical_threshold: float,
) -> dict[str, Any]:
    if warning_threshold <= 0 or warning_threshold >= critical_threshold:
        raise ValueError("drift thresholds must be positive and ordered")
    if minimum_events <= 0:
        raise ValueError("minimum_events must be positive")

    if len(current) < minimum_events:
        return {
            "generated_at": datetime.now(UTC),
            "model_version": reference.model_version,
            "reference_rows": reference.rows,
            "current_rows": len(current),
            "minimum_events": minimum_events,
            "status": DriftSeverity.INSUFFICIENT_DATA,
            "drifted_features": 0,
            "features": [],
        }

    missing = sorted(set(reference.features) - set(current.columns))
    if missing:
        raise ValueError(f"current frame is missing monitored features: {missing}")

    feature_results: list[dict[str, Any]] = []
    severities: list[DriftSeverity] = []
    for name, feature_reference in reference.features.items():
        values = current[name].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"current feature {name} contains non-finite values")
        current_proportions = _proportions(values, feature_reference.bin_edges)
        psi = population_stability_index(
            feature_reference.expected_proportions,
            current_proportions,
        )
        severity = _severity(
            psi,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
        )
        severities.append(severity)
        feature_results.append(
            {
                "feature": name,
                "population_stability_index": round(psi, 6),
                "severity": severity,
                "reference_mean": feature_reference.mean,
                "current_mean": float(values.mean()),
            }
        )

    severity_rank = {
        DriftSeverity.STABLE: 0,
        DriftSeverity.WARNING: 1,
        DriftSeverity.CRITICAL: 2,
    }
    status = max(severities, key=severity_rank.__getitem__)
    feature_results.sort(
        key=lambda result: result["population_stability_index"],
        reverse=True,
    )
    return {
        "generated_at": datetime.now(UTC),
        "model_version": reference.model_version,
        "reference_rows": reference.rows,
        "current_rows": len(current),
        "minimum_events": minimum_events,
        "status": status,
        "drifted_features": sum(
            severity in {DriftSeverity.WARNING, DriftSeverity.CRITICAL} for severity in severities
        ),
        "features": feature_results,
    }
