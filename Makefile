.PHONY: install train run test lint format check docker

install:
	python -m pip install -e ".[dev]"

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
