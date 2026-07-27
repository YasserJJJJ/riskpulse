from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from riskpulse.api.dependencies import get_database, get_risk_model, get_settings
from riskpulse.config import Settings
from riskpulse.domain.schemas import HealthResponse
from riskpulse.ml.service import RiskModel
from riskpulse.persistence.database import Database

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse)
def liveness(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.get("/ready", response_model=HealthResponse)
def readiness(
    settings: Annotated[Settings, Depends(get_settings)],
    risk_model: Annotated[RiskModel, Depends(get_risk_model)],
    database: Annotated[Database, Depends(get_database)],
) -> HealthResponse:
    _ = risk_model.model_version
    if not database.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database is unavailable",
        )
    return HealthResponse(
        status="ready",
        service=settings.app_name,
        version=settings.app_version,
    )
