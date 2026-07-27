.PHONY: install data benchmark train run test lint format check docker

install:
	python -m pip install -e ".[dev]"

data:
	PYTHONPATH=src python -m riskpulse.ml.inspect_data --data-home data/openml

benchmark:
	PYTHONPATH=src python -m riskpulse.ml.benchmark \
		--data-home data/openml \
		--output artifacts/real_data_benchmark.json

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
