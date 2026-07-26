from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from riskpulse.domain.schemas import ModelMetadataResponse, TransactionRequest
from riskpulse.ml.features import transaction_to_features, validate_feature_names


class ModelLoadError(RuntimeError):
    """Raised when an artifact cannot be safely loaded."""


class RiskModel:
    def __init__(self, artifact: dict[str, Any]) -> None:
        required_keys = {
            "model",
            "model_version",
            "trained_at",
            "feature_names",
            "validation_metrics",
        }
        missing_keys = required_keys - artifact.keys()
        if missing_keys:
            raise ModelLoadError(f"model artifact is missing keys: {sorted(missing_keys)}")

        try:
            validate_feature_names(artifact["feature_names"])
            trained_at = datetime.fromisoformat(artifact["trained_at"])
        except (TypeError, ValueError) as error:
            raise ModelLoadError(f"invalid model metadata: {error}") from error

        if not hasattr(artifact["model"], "predict_proba"):
            raise ModelLoadError("model must implement predict_proba")

        self._model = artifact["model"]
        self.model_version = str(artifact["model_version"])
        self.trained_at = trained_at
        self.feature_names = list(artifact["feature_names"])
        self.validation_metrics = {
            str(name): float(value) for name, value in artifact["validation_metrics"].items()
        }

    @classmethod
    def load(cls, path: Path) -> "RiskModel":
        if not path.is_file():
            raise ModelLoadError(
                f"model artifact not found at {path}; run `make train` before starting the API"
            )
        try:
            artifact = joblib.load(path)
        except Exception as error:
            raise ModelLoadError(f"could not load model artifact at {path}") from error
        if not isinstance(artifact, dict):
            raise ModelLoadError("model artifact must be a dictionary")
        return cls(artifact)

    def score(self, transaction: TransactionRequest) -> float:
        features = transaction_to_features(transaction)
        probability = float(self._model.predict_proba(features)[0, 1])
        return min(max(probability, 0.0), 1.0)

    def metadata(self) -> ModelMetadataResponse:
        return ModelMetadataResponse(
            model_version=self.model_version,
            trained_at=self.trained_at,
            feature_names=self.feature_names,
            validation_metrics=self.validation_metrics,
        )
