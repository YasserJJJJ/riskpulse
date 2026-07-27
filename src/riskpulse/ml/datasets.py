import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
from sklearn.datasets import fetch_openml

OPENML_CREDIT_CARD_DATA_ID = 1597
OPENML_PARQUET_URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
PARQUET_CACHE_FILENAME = "creditcard-1597.parquet"
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
TARGET_COLUMN = "Class"
ANONYMIZED_FEATURES = tuple(f"V{index}" for index in range(1, 29))
FEATURE_COLUMNS = (TIME_COLUMN, *ANONYMIZED_FEATURES, AMOUNT_COLUMN)
REQUIRED_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)


class DatasetSchemaError(ValueError):
    """Raised when downloaded data does not match the pinned dataset schema."""


@dataclass(frozen=True)
class DatasetSummary:
    openml_data_id: int
    transactions: int
    frauds: int
    fraud_rate: float
    feature_count: int
    time_start_seconds: float
    time_end_seconds: float


@dataclass(frozen=True)
class CreditCardFraudDataset:
    features: pd.DataFrame
    target: pd.Series
    summary: DatasetSummary


@dataclass(frozen=True)
class TemporalSplit:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_validation: pd.DataFrame
    y_validation: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def _download_parquet(cache_path: Path) -> None:
    temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.part")
    try:
        with (
            urlopen(OPENML_PARQUET_URL, timeout=120) as response,
            temporary_path.open("wb") as destination,
        ):
            shutil.copyfileobj(response, destination)
        temporary_path.replace(cache_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_fallback_frame(data_home: Path) -> pd.DataFrame:
    cache_path = data_home / PARQUET_CACHE_FILENAME
    if not cache_path.exists():
        _download_parquet(cache_path)
    return pd.read_parquet(cache_path)


def _load_source_frame(data_home: Path) -> pd.DataFrame:
    if (data_home / PARQUET_CACHE_FILENAME).exists():
        return _load_fallback_frame(data_home)

    try:
        downloaded: Any = fetch_openml(
            data_id=OPENML_CREDIT_CARD_DATA_ID,
            data_home=str(data_home),
            as_frame=True,
            parser="pandas",
        )
    except (HTTPError, URLError, TimeoutError):
        warnings.warn(
            "OpenML API unavailable; using the pinned Parquet download instead",
            RuntimeWarning,
            stacklevel=2,
        )
        return _load_fallback_frame(data_home)

    if not isinstance(downloaded.frame, pd.DataFrame):
        raise DatasetSchemaError("OpenML response did not include a pandas DataFrame")
    return downloaded.frame


def load_credit_card_fraud(
    data_home: Path = Path("data/openml"),
) -> CreditCardFraudDataset:
    """Download, validate, and chronologically order OpenML dataset 1597."""

    data_home.mkdir(parents=True, exist_ok=True)
    source_frame = _load_source_frame(data_home)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(source_frame.columns))
    if missing_columns:
        raise DatasetSchemaError(f"dataset is missing required columns: {missing_columns}")

    frame = source_frame.loc[:, REQUIRED_COLUMNS].copy()
    try:
        for column in REQUIRED_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise DatasetSchemaError("dataset contains non-numeric values") from error

    if frame.isna().any().any():
        raise DatasetSchemaError("dataset contains missing values")

    labels = set(frame[TARGET_COLUMN].astype(int).unique())
    if labels != {0, 1}:
        raise DatasetSchemaError(f"expected binary labels {{0, 1}}, received {labels}")

    frame = frame.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    features = frame.loc[:, FEATURE_COLUMNS].astype(float)
    target = frame[TARGET_COLUMN].astype("int8").rename(TARGET_COLUMN)
    frauds = int(target.sum())
    transactions = len(target)
    summary = DatasetSummary(
        openml_data_id=OPENML_CREDIT_CARD_DATA_ID,
        transactions=transactions,
        frauds=frauds,
        fraud_rate=frauds / transactions,
        feature_count=len(FEATURE_COLUMNS),
        time_start_seconds=float(features[TIME_COLUMN].iloc[0]),
        time_end_seconds=float(features[TIME_COLUMN].iloc[-1]),
    )
    return CreditCardFraudDataset(
        features=features,
        target=target,
        summary=summary,
    )


def temporal_split(
    dataset: CreditCardFraudDataset,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Split already ordered transactions without shuffling future into the past."""

    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave data for testing")

    transactions = len(dataset.target)
    if transactions < 10:
        raise ValueError("at least 10 transactions are required for a temporal split")

    train_end = int(transactions * train_fraction)
    validation_end = train_end + int(transactions * validation_fraction)

    return TemporalSplit(
        x_train=dataset.features.iloc[:train_end].copy(),
        y_train=dataset.target.iloc[:train_end].copy(),
        x_validation=dataset.features.iloc[train_end:validation_end].copy(),
        y_validation=dataset.target.iloc[train_end:validation_end].copy(),
        x_test=dataset.features.iloc[validation_end:].copy(),
        y_test=dataset.target.iloc[validation_end:].copy(),
    )
