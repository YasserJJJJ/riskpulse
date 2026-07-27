from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from riskpulse.domain.schemas import (
    CalibratedModelMetadataResponse,
    CreditCardTransactionRequest,
)
from riskpulse.ml.datasets import FEATURE_COLUMNS


class CalibratedModelLoadError(RuntimeError):
    """Raised when a calibrated real-data artifact cannot be safely loaded."""


class CalibratedFraudModel:
    def __init__(self, artifact: dict[str, Any]) -> None:
        required_keys = {
            "artifact_schema_version",
            "model",
            "model_version",
            "trained_at",
            "model_type",
            "calibration_method",
            "feature_names",
            "decision_threshold",
            "dataset_id",
            "validation_metrics",
            "test_metrics",
        }
        missing_keys = required_keys - artifact.keys()
        if missing_keys:
            raise CalibratedModelLoadError(
                f"calibrated artifact is missing keys: {sorted(missing_keys)}"
            )
        if artifact["artifact_schema_version"] != "1.0":
            raise CalibratedModelLoadError("unsupported calibrated artifact schema version")
        try:
            feature_names = list(artifact["feature_names"])
        except TypeError as error:
            raise CalibratedModelLoadError(
                "calibrated artifact feature schema is invalid"
            ) from error
        if feature_names != list(FEATURE_COLUMNS):
            raise CalibratedModelLoadError("calibrated artifact feature schema does not match")
        if not hasattr(artifact["model"], "predict_proba"):
            raise CalibratedModelLoadError("calibrated model must implement predict_proba")

        try:
            trained_at = datetime.fromisoformat(artifact["trained_at"])
            decision_threshold = float(artifact["decision_threshold"])
            dataset_id = int(artifact["dataset_id"])
            validation_metrics = {
                str(name): float(value) for name, value in artifact["validation_metrics"].items()
            }
            test_metrics = {
                str(name): float(value) for name, value in artifact["test_metrics"].items()
            }
        except (AttributeError, TypeError, ValueError) as error:
            raise CalibratedModelLoadError(f"invalid calibrated model metadata: {error}") from error
        if not 0 <= decision_threshold <= 1:
            raise CalibratedModelLoadError("decision threshold must be between zero and one")

        self._model = artifact["model"]
        self.artifact_schema_version = str(artifact["artifact_schema_version"])
        self.model_version = str(artifact["model_version"])
        self.trained_at = trained_at
        self.model_type = str(artifact["model_type"])
        self.calibration_method = str(artifact["calibration_method"])
        self.feature_names = feature_names
        self.decision_threshold = decision_threshold
        self.dataset_id = dataset_id
        self.validation_metrics = validation_metrics
        self.test_metrics = test_metrics

    @classmethod
    def load(cls, path: Path) -> "CalibratedFraudModel":
        if not path.is_file():
            raise CalibratedModelLoadError(
                f"calibrated model not found at {path}; run `make calibrate` first"
            )
        try:
            artifact = joblib.load(path)
        except Exception as error:
            raise CalibratedModelLoadError(f"could not load calibrated model at {path}") from error
        if not isinstance(artifact, dict):
            raise CalibratedModelLoadError("calibrated artifact must be a dictionary")
        return cls(artifact)

    def score(self, transaction: CreditCardTransactionRequest) -> float:
        values = [
            transaction.time,
            *(getattr(transaction, f"v{index}") for index in range(1, 29)),
            transaction.amount,
        ]
        features = pd.DataFrame([values], columns=list(FEATURE_COLUMNS))
        probability = float(self._model.predict_proba(features)[0, 1])
        return min(max(probability, 0.0), 1.0)

    def metadata(self) -> CalibratedModelMetadataResponse:
        return CalibratedModelMetadataResponse(
            artifact_schema_version=self.artifact_schema_version,
            model_version=self.model_version,
            trained_at=self.trained_at,
            model_type=self.model_type,
            calibration_method=self.calibration_method,
            feature_names=self.feature_names,
            decision_threshold=self.decision_threshold,
            dataset_id=self.dataset_id,
            validation_metrics=self.validation_metrics,
            test_metrics=self.test_metrics,
        )
