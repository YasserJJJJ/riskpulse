from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskpulse.monitoring.drift import (
    DriftReference,
    DriftReferenceError,
    build_drift_reference,
    calculate_drift,
    population_stability_index,
)


def reference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": np.arange(100, dtype=float),
            "velocity": np.tile(np.arange(10, dtype=float), 10),
        }
    )


def build_reference() -> DriftReference:
    return build_drift_reference(
        reference_frame(),
        model_version="model-v1",
        dataset_id=1597,
        bins=5,
        generated_at=datetime(2026, 7, 28, tzinfo=UTC),
    )


def test_builds_saves_and_loads_reference(tmp_path: Path) -> None:
    reference = build_reference()
    path = tmp_path / "nested" / "reference.json"

    reference.save(path)
    loaded = DriftReference.load(path)

    assert loaded == reference
    assert loaded.rows == 100
    assert loaded.features["amount"].mean == 49.5
    assert len(loaded.features["amount"].expected_proportions) == 5


def test_reports_stable_distribution() -> None:
    report = calculate_drift(
        build_reference(),
        reference_frame(),
        minimum_events=20,
        warning_threshold=0.10,
        critical_threshold=0.25,
    )

    assert report["status"] == "stable"
    assert report["drifted_features"] == 0
    assert report["current_rows"] == 100
    assert report["features"][0]["population_stability_index"] == 0.0


def test_detects_critical_drift_and_sorts_features() -> None:
    current = reference_frame()
    current["amount"] += 10_000

    report = calculate_drift(
        build_reference(),
        current,
        minimum_events=20,
        warning_threshold=0.10,
        critical_threshold=0.25,
    )

    assert report["status"] == "critical"
    assert report["drifted_features"] == 1
    assert report["features"][0]["feature"] == "amount"
    assert report["features"][0]["severity"] == "critical"


def test_reports_insufficient_data_without_requiring_columns() -> None:
    report = calculate_drift(
        build_reference(),
        pd.DataFrame(),
        minimum_events=20,
        warning_threshold=0.10,
        critical_threshold=0.25,
    )

    assert report["status"] == "insufficient_data"
    assert report["current_rows"] == 0
    assert report["features"] == []


@pytest.mark.parametrize(
    ("expected", "current", "message"),
    [
        ([1.0], [0.5, 0.5], "equal non-zero length"),
        ([0.5, 0.4], [0.5, 0.5], "sum to one"),
        ([1.0], [-1.0], "finite non-negative"),
    ],
)
def test_rejects_invalid_psi_distributions(
    expected: list[float],
    current: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        population_stability_index(expected, current)


def test_rejects_invalid_reference_inputs() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        build_drift_reference(pd.DataFrame(), model_version="v1", dataset_id=1)
    with pytest.raises(ValueError, match="two reference bins"):
        build_drift_reference(
            reference_frame(),
            model_version="v1",
            dataset_id=1,
            bins=1,
        )
    invalid = reference_frame()
    invalid.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        build_drift_reference(invalid, model_version="v1", dataset_id=1)


def test_rejects_missing_current_feature_and_bad_thresholds() -> None:
    reference = build_reference()
    with pytest.raises(ValueError, match="thresholds"):
        calculate_drift(
            reference,
            reference_frame(),
            minimum_events=20,
            warning_threshold=0.3,
            critical_threshold=0.2,
        )
    with pytest.raises(ValueError, match="minimum_events"):
        calculate_drift(
            reference,
            reference_frame(),
            minimum_events=0,
            warning_threshold=0.1,
            critical_threshold=0.2,
        )
    with pytest.raises(ValueError, match="missing monitored features"):
        calculate_drift(
            reference,
            reference_frame().drop(columns="velocity"),
            minimum_events=20,
            warning_threshold=0.1,
            critical_threshold=0.2,
        )


def test_rejects_invalid_reference_file(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_text('{"schema_version": "2.0", "features": {}}', encoding="utf-8")

    with pytest.raises(DriftReferenceError, match="invalid drift reference"):
        DriftReference.load(path)

    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(DriftReferenceError, match="could not read"):
        DriftReference.load(path)
