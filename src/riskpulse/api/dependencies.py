from fastapi import Request

from riskpulse.config import Settings
from riskpulse.ml.service import RiskModel


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_risk_model(request: Request) -> RiskModel:
    return request.app.state.risk_model
