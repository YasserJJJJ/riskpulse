import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request

from riskpulse.api.credit_card import router as credit_card_router
from riskpulse.api.health import router as health_router
from riskpulse.api.monitoring import router as monitoring_router
from riskpulse.api.scoring import router as scoring_router
from riskpulse.config import Settings
from riskpulse.ml.calibrated_service import CalibratedFraudModel
from riskpulse.ml.service import RiskModel
from riskpulse.monitoring.drift import DriftReference, DriftReferenceError
from riskpulse.monitoring.metrics import MonitoringMetrics
from riskpulse.persistence.database import Database


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(
            level=runtime_settings.log_level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        app.state.settings = runtime_settings
        app.state.monitoring_metrics = MonitoringMetrics()
        database = Database(
            runtime_settings.database_url,
            echo=runtime_settings.database_echo,
        )
        app.state.database = database
        try:
            app.state.risk_model = RiskModel.load(runtime_settings.model_path)
            app.state.calibrated_model = (
                CalibratedFraudModel.load(runtime_settings.calibrated_model_path)
                if runtime_settings.calibrated_model_path.is_file()
                else None
            )
            app.state.drift_reference = None
            if runtime_settings.monitoring_reference_path.is_file():
                try:
                    reference = DriftReference.load(runtime_settings.monitoring_reference_path)
                    if (
                        app.state.calibrated_model is not None
                        and reference.model_version != app.state.calibrated_model.model_version
                    ):
                        logging.getLogger(__name__).warning(
                            "drift reference model mismatch reference=%s loaded=%s",
                            reference.model_version,
                            app.state.calibrated_model.model_version,
                        )
                    else:
                        app.state.drift_reference = reference
                except DriftReferenceError as error:
                    logging.getLogger(__name__).warning(
                        "invalid drift reference path=%s error=%s",
                        runtime_settings.monitoring_reference_path,
                        error,
                    )
            logging.getLogger(__name__).info(
                "loaded model version=%s database=%s",
                app.state.risk_model.model_version,
                database.engine.url.render_as_string(hide_password=True),
            )
            if not database.is_ready():
                logging.getLogger(__name__).warning(
                    "database schema unavailable; run `make migrate`"
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
            if app.state.drift_reference is None:
                logging.getLogger(__name__).warning(
                    "drift reference unavailable path=%s",
                    runtime_settings.monitoring_reference_path,
                )
            else:
                logging.getLogger(__name__).info(
                    "loaded drift reference model=%s rows=%s",
                    app.state.drift_reference.model_version,
                    app.state.drift_reference.rows,
                )
            yield
        finally:
            database.close()

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
    application.include_router(monitoring_router)

    @application.middleware("http")
    async def observe_http_request(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = perf_counter()
        request_id = request.headers.get("X-Request-ID", "").strip()
        if (
            not request_id
            or len(request_id) > 128
            or re.fullmatch(r"[A-Za-z0-9._:-]+", request_id) is None
        ):
            request_id = uuid4().hex
        try:
            response = await call_next(request)
        except Exception:
            route = request.scope.get("route")
            route_path = getattr(route, "path", "unmatched")
            duration = perf_counter() - started
            application.state.monitoring_metrics.observe_http(
                method=request.method,
                route=route_path,
                status_code=500,
                duration_seconds=duration,
            )
            logging.getLogger("riskpulse.request").exception(
                "request_id=%s method=%s route=%s status=500 duration_ms=%.3f",
                request_id,
                request.method,
                route_path,
                duration * 1_000,
            )
            raise
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        duration = perf_counter() - started
        application.state.monitoring_metrics.observe_http(
            method=request.method,
            route=route_path,
            status_code=response.status_code,
            duration_seconds=duration,
        )
        response.headers["X-Request-ID"] = request_id
        logging.getLogger("riskpulse.request").info(
            "request_id=%s method=%s route=%s status=%s duration_ms=%.3f",
            request_id,
            request.method,
            route_path,
            response.status_code,
            duration * 1_000,
        )
        return response

    @application.get("/", tags=["service"])
    def root() -> dict[str, str]:
        return {
            "service": runtime_settings.app_name,
            "documentation": "/docs",
            "health": "/health/ready",
        }

    return application


app = create_app()
