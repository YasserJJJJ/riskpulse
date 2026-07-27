import argparse
import json
from dataclasses import asdict
from pathlib import Path

from riskpulse.ml.datasets import load_credit_card_fraud, temporal_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and inspect the pinned credit-card fraud dataset."
    )
    parser.add_argument(
        "--data-home",
        type=Path,
        default=Path("data/openml"),
        help="Local OpenML cache directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_credit_card_fraud(args.data_home)
    split = temporal_split(dataset)
    output = {
        "dataset": asdict(dataset.summary),
        "temporal_split": {
            "train": len(split.y_train),
            "validation": len(split.y_validation),
            "test": len(split.y_test),
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
