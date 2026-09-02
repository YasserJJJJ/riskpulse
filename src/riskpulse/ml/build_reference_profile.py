import argparse
import json
from pathlib import Path

from riskpulse.ml.calibrated_service import CalibratedFraudModel
from riskpulse.ml.datasets import load_credit_card_fraud, temporal_split
from riskpulse.monitoring.drift import PREDICTION_FEATURE, build_drift_reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a versioned training-data drift reference.",
    )
    parser.add_argument("--data-home", type=Path, default=Path("data/openml"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/calibrated_creditcard_model.joblib"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/creditcard_reference_profile.json"),
    )
    parser.add_argument("--bins", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_credit_card_fraud(args.data_home)
    training_features = temporal_split(dataset).x_train
    model = CalibratedFraudModel.load(args.model)
    reference_frame = training_features.copy()
    reference_frame[PREDICTION_FEATURE] = model.score_features(training_features)
    reference = build_drift_reference(
        reference_frame,
        model_version=model.model_version,
        dataset_id=model.dataset_id,
        bins=args.bins,
    )
    reference.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_version": reference.model_version,
                "reference_rows": reference.rows,
                "monitored_features": len(reference.features),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
