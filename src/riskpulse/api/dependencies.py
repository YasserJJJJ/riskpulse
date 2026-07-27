from fastapi import HTTPException, Request, status

from riskpulse.config import Settings
from riskpulse.ml.calibrated_service import CalibratedFraudModel
from riskpulse.ml.service import RiskModel


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
