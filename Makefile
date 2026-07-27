.PHONY: install data benchmark calibrate migrate train run test lint format check docker

install:
	python -m pip install -e ".[dev]"

data:
	PYTHONPATH=src python -m riskpulse.ml.inspect_data --data-home data/openml

benchmark:
	PYTHONPATH=src python -m riskpulse.ml.benchmark \
		--data-home data/openml \
		--output artifacts/real_data_benchmark.json

calibrate:
	PYTHONPATH=src python -m riskpulse.ml.calibrate \
		--data-home data/openml \
		--model-output artifacts/calibrated_creditcard_model.joblib \
		--report-output artifacts/calibrated_model_report.json \
		--model-card-output artifacts/MODEL_CARD.md

migrate:
	alembic upgrade head

train:
	python -m riskpulse.ml.train --output artifacts/fraud_model.joblib

run:
	python -m uvicorn riskpulse.main:app --app-dir src --reload

test:
	pytest

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

check: lint test

docker:
	docker compose up --build
