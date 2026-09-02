from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST
from sqlalchemy.orm import Session

from riskpulse.api.dependencies import (
    get_calibrated_model,
    get_drift_reference,
    get_monitoring_metrics,
    get_session,
    get_settings,
)
from riskpulse.config import Settings
from riskpulse.domain.schemas import DriftReportResponse
from riskpulse.ml.calibrated_service import CalibratedFraudModel
from riskpulse.monitoring.drift import (
    PREDICTION_FEATURE,
    DriftReference,
    calculate_drift,
)
from riskpulse.monitoring.metrics import MonitoringMetrics
from riskpulse.persistence.repository import AuditRepository

router = APIRouter(tags=["monitoring"])


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics(
    metrics: Annotated[MonitoringMetrics, Depends(get_monitoring_metrics)],
) -> Response:
    return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)


@router.get(
    "/v1/monitoring/drift",
    response_model=DriftReportResponse,
)
def get_drift_report(
    session: Annotated[Session, Depends(get_session)],
    model: Annotated[CalibratedFraudModel, Depends(get_calibrated_model)],
    reference: Annotated[DriftReference, Depends(get_drift_reference)],
    settings: Annotated[Settings, Depends(get_settings)],
    limit: Annotated[int, Query(ge=20, le=10_000)] = 1_000,
) -> DriftReportResponse:
    events = AuditRepository(session).recent_events(
        model_version=model.model_version,
        limit=limit,
    )
    rows = [
        {
            **{str(name): float(value) for name, value in event.features.items()},
            PREDICTION_FEATURE: event.fraud_probability,
        }
        for event in events
    ]
    current = pd.DataFrame(rows)
    report = calculate_drift(
        reference,
        current,
        minimum_events=settings.drift_minimum_events,
        warning_threshold=settings.drift_warning_psi,
        critical_threshold=settings.drift_critical_psi,
    )
    return DriftReportResponse.model_validate(report)
