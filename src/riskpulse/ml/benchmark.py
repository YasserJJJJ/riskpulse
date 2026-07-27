import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from riskpulse.ml.benchmarking import CostPolicy, benchmark_models
from riskpulse.ml.datasets import load_credit_card_fraud, temporal_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fraud models with temporal and cost-sensitive evaluation."
    )
    parser.add_argument("--data-home", type=Path, default=Path("data/openml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/real_data_benchmark.json"),
    )
    parser.add_argument("--false-positive-cost", type=float, default=5.0)
    parser.add_argument("--missed-fraud-multiplier", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_credit_card_fraud(args.data_home)
    split = temporal_split(dataset)
    policy = CostPolicy(
        false_positive_review_cost=args.false_positive_cost,
        missed_fraud_amount_multiplier=args.missed_fraud_multiplier,
    )
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": asdict(dataset.summary),
        "temporal_split": {
            "train": len(split.y_train),
            "validation": len(split.y_validation),
            "test": len(split.y_test),
        },
        **benchmark_models(split, policy=policy),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
