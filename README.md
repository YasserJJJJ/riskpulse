# RiskPulse

RiskPulse is a production-minded real-time transaction risk scoring service. It
combines a calibrated fraud model with a typed FastAPI backend, strict artifact
validation, automated tests, containerization, and CI.

The API keeps the original deterministic demo endpoint and can also serve the
calibrated model trained on a documented public fraud dataset. Its metrics are
development metrics, not claims about real-world fraud performance.

## Results

The calibrated model was evaluated once on the last chronological 15% of OpenML dataset 1597 (284,807 card transactions, 0.17% fraud), after model selection, calibration and threshold choice on earlier periods.

| Test-period metric | Value |
| --- | --- |
| ROC-AUC | 0.963 |
| PR-AUC | 0.743 |
| Precision at the chosen threshold | 0.826 |
| Recall at the chosen threshold | 0.731 |
| Transactions sent to manual review | 0.11% |
| Fraud amount captured | 60.9% |

These are development metrics on a public, anonymized dataset, not claims about production fraud performance. The full report is in [`artifacts/calibrated_model_report.json`](artifacts/calibrated_model_report.json) and the [model card](artifacts/MODEL_CARD.md) explains the method and its limits.

## Current capabilities

- Train and serialize a reproducible logistic-regression baseline
- Validate transaction inputs with Pydantic
- Return risk scores, decisions, and stable reason codes
- Expose liveness, readiness, and model-metadata endpoints
- Generate interactive OpenAPI documentation
- Run locally or in Docker
- Enforce linting, formatting, tests, and coverage in GitHub Actions
- Load and validate OpenML credit-card fraud dataset 1597
- Preserve chronology with a 70/15/15 train/validation/test split
- Tune alert thresholds against explicit fraud-loss and review-cost assumptions
- Calibrate fraud probabilities on a dedicated chronological period
- Serve the versioned real-data artifact behind a typed scoring contract
- Fail gracefully when the optional calibrated artifact is unavailable
- Persist scoring events and review feedback through SQLAlchemy
- Protect scoring writes with payload-aware idempotency keys
- Manage the database schema with Alembic migrations
- Run PostgreSQL alongside the API with Docker Compose
- Export bounded-cardinality HTTP and model metrics for Prometheus
- Detect feature and prediction drift against a versioned training reference
- Propagate request IDs and emit structured request-completion logs

## Architecture

```mermaid
flowchart TD
    C["Client"] --> A["FastAPI scoring API"]
    A --> V["Pydantic validation"]
    V --> M["Versioned risk model"]
    M --> D["Decision and audit event"]
    D --> P["PostgreSQL or SQLite"]
    D --> R["Typed JSON response"]
    P --> Q["Manual-review queue"]
    Q --> F["Analyst feedback"]
    F --> P
    P --> O["Drift report"]
    A --> X["Prometheus metrics"]
```

## Quick start

Requirements: Python 3.12 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m riskpulse.ml.train --output artifacts/fraud_model.joblib
make calibrate
make monitoring-reference
make migrate
python -m uvicorn riskpulse.main:app --app-dir src --reload
```

Open:

- API documentation: `http://127.0.0.1:8000/docs`
- Readiness check: `http://127.0.0.1:8000/health/ready`
- Demo model metadata: `http://127.0.0.1:8000/v1/model`
- Calibrated model metadata: `http://127.0.0.1:8000/v1/credit-card/model`
- Prometheus metrics: `http://127.0.0.1:8000/metrics`
- Drift report: `http://127.0.0.1:8000/v1/monitoring/drift`

Score a transaction:

```bash
curl -X POST "http://127.0.0.1:8000/v1/transactions/score" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 850.0,
    "account_age_days": 12,
    "hour_of_day": 2,
    "is_international": true,
    "is_new_device": true,
    "failed_attempts_24h": 3,
    "transactions_1h": 9,
    "distance_from_home_km": 620.0
  }'
```

Example response:

```json
{
  "transaction_id": "generated-uuid",
  "risk_score": 0.91,
  "decision": "decline",
  "reasons": [
    "new_device",
    "international_transaction",
    "multiple_failed_attempts",
    "high_transaction_velocity",
    "large_location_change",
    "unusual_transaction_time"
  ],
  "model_version": "baseline-YYYYMMDD-identifier",
  "scored_at": "generated-utc-timestamp"
}
```

Actual values vary with the trained artifact.

The real-data endpoint is available at
`POST /v1/credit-card/transactions/score`. Its request contains the public
dataset's `time`, `amount`, and anonymized `v1` through `v28` PCA features.
Swagger UI provides the complete editable request example. The response returns
a calibrated fraud probability and routes an alert to `manual_review`; it does
not automatically decline a transaction. Clients should send an
`Idempotency-Key` header. Repeating the same key and feature payload returns the
original event, while changing the payload produces HTTP 409.

Local commands default to SQLite at `artifacts/riskpulse.db`, keeping setup
small. Docker Compose runs the same SQLAlchemy repository against PostgreSQL.

## Development commands

```bash
make install
make data
make benchmark
make calibrate
make monitoring-reference
make migrate
make train
make run
make lint
make test
make check
```

`make benchmark` fits a dummy reference, class-weighted logistic regression,
and histogram gradient boosting. Models and thresholds are selected using only
the validation period. The selected pair is then evaluated once on the
untouched test period. The report is written to
`artifacts/real_data_benchmark.json`.

`make calibrate` keeps four chronological periods separate: model fitting,
probability calibration, threshold selection, and final testing. It writes a
versioned Joblib model, a machine-readable evaluation report, and a
human-readable model card under `artifacts/`.

`make monitoring-reference` scores the chronological training period with the
selected calibrated artifact and stores decile distributions for all 30 input
features plus the model output. The runtime drift endpoint compares recent,
persisted scoring events with this versioned reference using Population
Stability Index (PSI). The report stays `insufficient_data` until the configured
minimum sample size is reached, warns at PSI 0.10, and becomes critical at PSI
0.25 by default.

For PostgreSQL and the API in Docker:

```bash
docker compose up --build
```

Compose starts PostgreSQL, waits for its health check, applies the Alembic
migration, mounts the locally generated calibrated Joblib artifact read-only,
and then starts the API as a non-root user. It also starts Prometheus at
`http://127.0.0.1:9090` and scrapes the API every 15 seconds. Run
`make calibrate` and `make monitoring-reference` before the first Compose
startup so that both read-only artifacts exist.

## API surface

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health/live` | Confirm that the service process is running |
| `GET` | `/health/ready` | Confirm that the model is loaded |
| `GET` | `/v1/model` | Inspect model version, features, and metrics |
| `POST` | `/v1/transactions/score` | Score one demo-schema transaction |
| `GET` | `/v1/credit-card/model` | Inspect the calibrated artifact contract |
| `POST` | `/v1/credit-card/transactions/score` | Score anonymized public-data features |
| `GET` | `/v1/credit-card/transactions/{id}` | Retrieve an immutable scoring audit event |
| `GET` | `/v1/credit-card/reviews` | List unresolved manual-review events |
| `POST` | `/v1/credit-card/transactions/{id}/feedback` | Record an analyst outcome |
| `GET` | `/v1/monitoring/drift` | Compare recent traffic with the training reference |
| `GET` | `/metrics` | Export Prometheus latency, traffic, scoring, and feedback metrics |

## Important modelling choices

- Inputs contain only information assumed to be available at authorization
  time, reducing leakage risk.
- Accuracy is intentionally not the headline metric. The training pipeline
  records ROC-AUC, PR-AUC, precision, and recall.
- Candidate models address the severe class imbalance through weighting or
  nonlinear learning, then calibrate the selected model on a separate period.
- Cost-sensitive threshold selection charges a configurable fixed review cost
  for false alerts and the transaction amount for missed fraud. These are
  transparent modelling assumptions, not claimed production costs.
- Model artifacts carry their version, training timestamp, feature schema, and
  validation metrics.
- The service rejects artifacts whose feature order no longer matches the API.
- Joblib artifacts can execute code while loading, so the service accepts only
  artifacts produced by a trusted training pipeline.
- Feature payloads are anonymized PCA values from the public dataset. A
  production deployment should apply its own retention and access policies.
- The feedback endpoint intentionally has no authentication in this portfolio
  build; authentication and role-based authorization are required before use
  with real analysts.
- PSI is an unsupervised shift signal, not proof of degraded fraud performance.
  Investigation and labelled feedback are required before retraining or
  promoting another model.
- Prometheus labels use route templates and controlled model fields to avoid
  transaction-level cardinality.
- Reason codes describe triggered business signals. They are not presented as
  causal explanations.

## Possible extensions

The project is complete as a scoring service. These would be the next steps toward a real deployment:

- Redis-backed online features and batch scoring
- Experiment and artifact tracking with MLflow
- Champion/challenger promotion and rollback
- Authentication and role-based access for the review endpoints
- Replaying transactions as an event stream, and published load-test results

## Project structure

```text
src/riskpulse/
├── api/          # FastAPI routes and dependencies
├── domain/       # Request, response, and decision models
├── ml/           # Features, training, artifacts, and inference
├── persistence/  # SQLAlchemy models, sessions, and audit repository
├── config.py     # Environment-based configuration
└── main.py       # Application factory
```

## License

MIT
