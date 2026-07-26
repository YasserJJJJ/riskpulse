from typing import Annotated

from fastapi import APIRouter, Depends

from riskpulse.api.dependencies import get_risk_model, get_settings
from riskpulse.config import Settings
from riskpulse.domain.schemas import HealthResponse
from riskpulse.ml.service import RiskModel

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
) -> HealthResponse:
    _ = risk_model.model_version
    return HealthResponse(
        status="ready",
        service=settings.app_name,
        version=settings.app_version,
    )
