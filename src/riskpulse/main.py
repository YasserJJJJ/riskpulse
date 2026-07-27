import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from riskpulse.api.credit_card import router as credit_card_router
from riskpulse.api.health import router as health_router
from riskpulse.api.scoring import router as scoring_router
from riskpulse.config import Settings
from riskpulse.ml.calibrated_service import CalibratedFraudModel
from riskpulse.ml.service import RiskModel


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(
            level=runtime_settings.log_level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        app.state.settings = runtime_settings
        app.state.risk_model = RiskModel.load(runtime_settings.model_path)
        app.state.calibrated_model = (
            CalibratedFraudModel.load(runtime_settings.calibrated_model_path)
            if runtime_settings.calibrated_model_path.is_file()
            else None
        )
        logging.getLogger(__name__).info(
            "loaded model version=%s",
            app.state.risk_model.model_version,
        )
        if app.state.calibrated_model is None:
            logging.getLogger(__name__).warning(
                "calibrated model unavailable path=%s",
                runtime_settings.calibrated_model_path,
            )
        else:
            logging.getLogger(__name__).info(
                "loaded calibrated model version=%s",
                app.state.calibrated_model.model_version,
            )
        yield

    application = FastAPI(
        title=runtime_settings.app_name,
        version=runtime_settings.app_version,
        description=(
            "Real-time transaction risk scoring API with a reproducible demo "
            "model and an optional calibrated public-data model."
        ),
        lifespan=lifespan,
    )
    application.include_router(health_router)
    application.include_router(scoring_router)
    application.include_router(credit_card_router)

    @application.get("/", tags=["service"])
    def root() -> dict[str, str]:
        return {
            "service": runtime_settings.app_name,
            "documentation": "/docs",
            "health": "/health/ready",
        }

    return application


app = create_app()
