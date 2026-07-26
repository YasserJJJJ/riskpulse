import argparse
import json
from pathlib import Path

from riskpulse.ml.training import train_and_save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RiskPulse baseline model.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/fraud_model.joblib"),
        help="Destination for the serialized model artifact.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=12_000,
        help="Number of reproducible demo transactions to generate.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_save_model(
        args.output,
        samples=args.samples,
        random_state=args.random_state,
    )
    summary = {
        "model_version": result.model_version,
        "output_path": str(result.output_path),
        "positive_rate": round(result.positive_rate, 4),
        "validation_metrics": {
            key: round(value, 4) for key, value in result.validation_metrics.items()
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
