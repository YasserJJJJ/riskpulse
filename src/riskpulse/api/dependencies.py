from collections.abc import Iterator

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from riskpulse.config import Settings
from riskpulse.ml.calibrated_service import CalibratedFraudModel
from riskpulse.ml.service import RiskModel
from riskpulse.monitoring.drift import DriftReference
from riskpulse.monitoring.metrics import MonitoringMetrics
from riskpulse.persistence.database import Database


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_risk_model(request: Request) -> RiskModel:
    return request.app.state.risk_model


def get_calibrated_model(request: Request) -> CalibratedFraudModel:
    model: CalibratedFraudModel | None = request.app.state.calibrated_model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "calibrated real-data model unavailable; run `make calibrate` and restart the API"
            ),
        )
    return model


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_monitoring_metrics(request: Request) -> MonitoringMetrics:
    return request.app.state.monitoring_metrics


def get_drift_reference(request: Request) -> DriftReference:
    reference: DriftReference | None = request.app.state.drift_reference
    if reference is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "drift reference unavailable; run `make monitoring-reference` and restart the API"
            ),
        )
    return reference


def get_session(request: Request) -> Iterator[Session]:
    database = get_database(request)
    yield from database.sessions()
