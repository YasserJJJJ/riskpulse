import argparse
import json
from pathlib import Path

import joblib

from riskpulse.ml.benchmarking import CostPolicy
from riskpulse.ml.calibration import train_calibrated_model
from riskpulse.ml.datasets import load_credit_card_fraud
from riskpulse.ml.model_card import render_model_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and package the calibrated real-data fraud model."
    )
    parser.add_argument("--data-home", type=Path, default=Path("data/openml"))
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/calibrated_creditcard_model.joblib"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/calibrated_model_report.json"),
    )
    parser.add_argument(
        "--model-card-output",
        type=Path,
        default=Path("artifacts/MODEL_CARD.md"),
    )
    parser.add_argument("--false-positive-cost", type=float, default=5.0)
    parser.add_argument("--missed-fraud-multiplier", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_credit_card_fraud(args.data_home)
    result = train_calibrated_model(
        dataset,
        policy=CostPolicy(
            false_positive_review_cost=args.false_positive_cost,
            missed_fraud_amount_multiplier=args.missed_fraud_multiplier,
        ),
    )

    for output_path in (args.model_output, args.report_output, args.model_card_output):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.artifact, args.model_output)
    args.report_output.write_text(json.dumps(result.report, indent=2) + "\n")
    args.model_card_output.write_text(render_model_card(result.report))

    summary = {
        "model_version": result.report["model_version"],
        "model_output": str(args.model_output),
        "report_output": str(args.report_output),
        "model_card_output": str(args.model_card_output),
        "selected_threshold": result.report["selected_threshold"],
        "test": result.report["test"],
        "test_reliability": result.report["reliability"]["test_calibrated"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
