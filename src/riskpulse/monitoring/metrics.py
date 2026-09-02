from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


class MonitoringMetrics:
    """Application-local Prometheus collectors that remain isolated in tests."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.http_requests = Counter(
            "riskpulse_http_requests_total",
            "HTTP requests handled by the RiskPulse API.",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "riskpulse_http_request_duration_seconds",
            "HTTP request duration in seconds.",
            ("method", "route"),
            registry=self.registry,
        )
        self.scoring_requests = Counter(
            "riskpulse_scoring_requests_total",
            "Calibrated scoring requests, including idempotent replays.",
            ("model_version", "route", "replayed"),
            registry=self.registry,
        )
        self.fraud_probability = Histogram(
            "riskpulse_fraud_probability",
            "Distribution of calibrated fraud probabilities.",
            ("model_version", "route"),
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
            registry=self.registry,
        )
        self.review_feedback = Counter(
            "riskpulse_review_feedback_total",
            "Successful analyst-feedback API requests.",
            ("outcome",),
            registry=self.registry,
        )

    def observe_http(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        self.http_requests.labels(
            method=method,
            route=route,
            status=str(status_code),
        ).inc()
        self.http_duration.labels(method=method, route=route).observe(duration_seconds)

    def observe_score(
        self,
        *,
        model_version: str,
        route: str,
        replayed: bool,
        fraud_probability: float,
    ) -> None:
        self.scoring_requests.labels(
            model_version=model_version,
            route=route,
            replayed=str(replayed).lower(),
        ).inc()
        self.fraud_probability.labels(
            model_version=model_version,
            route=route,
        ).observe(fraud_probability)

    def observe_feedback(self, *, outcome: str) -> None:
        self.review_feedback.labels(outcome=outcome).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)
