from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pandas as pd
import pytest

from riskpulse.ml.datasets import (
    ANONYMIZED_FEATURES,
    FEATURE_COLUMNS,
    PARQUET_CACHE_FILENAME,
    CreditCardFraudDataset,
    DatasetSchemaError,
    DatasetSummary,
    load_credit_card_fraud,
    temporal_split,
)


def sample_frame(rows: int = 20) -> pd.DataFrame:
    data: dict[str, list[float | int]] = {
        "Time": list(reversed(range(rows))),
        "Amount": [float(index + 1) for index in range(rows)],
        "Class": [1 if index in {2, rows - 1} else 0 for index in range(rows)],
    }
    for feature_index, feature_name in enumerate(ANONYMIZED_FEATURES, start=1):
        data[feature_name] = [float(feature_index * 100 + row_index) for row_index in range(rows)]
    return pd.DataFrame(data)


def test_loads_validates_and_orders_openml_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = sample_frame()

    def fake_fetch_openml(**kwargs: object) -> SimpleNamespace:
        assert kwargs["data_id"] == 1597
        assert kwargs["as_frame"] is True
        assert kwargs["parser"] == "pandas"
        return SimpleNamespace(frame=frame)

    monkeypatch.setattr("riskpulse.ml.datasets.fetch_openml", fake_fetch_openml)

    dataset = load_credit_card_fraud(tmp_path / "openml")

    assert dataset.features.columns.tolist() == list(FEATURE_COLUMNS)
    assert dataset.features["Time"].is_monotonic_increasing
    assert dataset.target.dtype == "int8"
    assert dataset.summary.transactions == 20
    assert dataset.summary.frauds == 2
    assert dataset.summary.fraud_rate == pytest.approx(0.10)
    assert dataset.summary.feature_count == 30


def test_uses_cached_parquet_without_calling_openml_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = sample_frame()
    cache_path = tmp_path / PARQUET_CACHE_FILENAME
    cache_path.write_bytes(b"cached parquet placeholder")

    def unexpected_openml_call(**_: object) -> None:
        raise AssertionError("cached data should avoid an OpenML API request")

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        assert path == cache_path
        return frame

    monkeypatch.setattr("riskpulse.ml.datasets.fetch_openml", unexpected_openml_call)
    monkeypatch.setattr("riskpulse.ml.datasets.pd.read_parquet", fake_read_parquet)

    dataset = load_credit_card_fraud(tmp_path)

    assert dataset.summary.transactions == 20
    assert dataset.features["Time"].is_monotonic_increasing


def test_downloads_parquet_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = sample_frame()

    class FakeResponse:
        def __init__(self) -> None:
            self.consumed = False

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            del size
            if self.consumed:
                return b""
            self.consumed = True
            return b"parquet bytes"

    monkeypatch.setattr(
        "riskpulse.ml.datasets.fetch_openml",
        lambda **_: (_ for _ in ()).throw(
            HTTPError("https://www.openml.org", 504, "Gateway Time-out", {}, None)
        ),
    )
    monkeypatch.setattr("riskpulse.ml.datasets.urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr("riskpulse.ml.datasets.pd.read_parquet", lambda _path: frame)

    with pytest.warns(RuntimeWarning):
        load_credit_card_fraud(tmp_path)

    assert (tmp_path / PARQUET_CACHE_FILENAME).read_bytes() == b"parquet bytes"
    assert not (tmp_path / f"{PARQUET_CACHE_FILENAME}.part").exists()


def test_rejects_missing_dataset_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = sample_frame().drop(columns=["Amount"])
    monkeypatch.setattr(
        "riskpulse.ml.datasets.fetch_openml",
        lambda **_: SimpleNamespace(frame=frame),
    )

    with pytest.raises(DatasetSchemaError, match="Amount"):
        load_credit_card_fraud(tmp_path)


def test_rejects_invalid_target_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = sample_frame()
    frame["Class"] = 2
    monkeypatch.setattr(
        "riskpulse.ml.datasets.fetch_openml",
        lambda **_: SimpleNamespace(frame=frame),
    )

    with pytest.raises(DatasetSchemaError, match="binary labels"):
        load_credit_card_fraud(tmp_path)


def test_temporal_split_preserves_order() -> None:
    frame = sample_frame().sort_values("Time").reset_index(drop=True)
    target = frame.pop("Class").astype("int8")
    dataset = CreditCardFraudDataset(
        features=frame.loc[:, FEATURE_COLUMNS],
        target=target,
        summary=DatasetSummary(
            openml_data_id=1597,
            transactions=20,
            frauds=2,
            fraud_rate=0.10,
            feature_count=30,
            time_start_seconds=0,
            time_end_seconds=19,
        ),
    )

    split = temporal_split(
        dataset,
        train_fraction=0.60,
        validation_fraction=0.20,
    )

    assert len(split.y_train) == 12
    assert len(split.y_validation) == 4
    assert len(split.y_test) == 4
    assert split.x_train["Time"].max() < split.x_validation["Time"].min()
    assert split.x_validation["Time"].max() < split.x_test["Time"].min()


@pytest.mark.parametrize(
    ("train_fraction", "validation_fraction", "message"),
    [
        (0.0, 0.2, "must be positive"),
        (0.8, 0.2, "leave data for testing"),
    ],
)
def test_temporal_split_rejects_invalid_fractions(
    train_fraction: float,
    validation_fraction: float,
    message: str,
) -> None:
    frame = sample_frame()
    target = frame.pop("Class").astype("int8")
    dataset = CreditCardFraudDataset(
        features=frame.loc[:, FEATURE_COLUMNS],
        target=target,
        summary=DatasetSummary(1597, 20, 2, 0.10, 30, 0, 19),
    )

    with pytest.raises(ValueError, match=message):
        temporal_split(
            dataset,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
