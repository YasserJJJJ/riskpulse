FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system riskpulse && useradd --system --gid riskpulse riskpulse

COPY pyproject.toml README.md alembic.ini ./
COPY src ./src
COPY migrations ./migrations

RUN python -m pip install --upgrade pip && \
    python -m pip install . && \
    mkdir -p /app/artifacts && \
    python -m riskpulse.ml.train \
        --output /app/artifacts/fraud_model.joblib \
        --samples 12000 && \
    chown -R riskpulse:riskpulse /app

USER riskpulse

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')"

CMD ["sh", "-c", "alembic upgrade head && uvicorn riskpulse.main:app --host 0.0.0.0 --port 8000"]
