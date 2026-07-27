from __future__ import annotations

from typing import Any


def _metric(value: float) -> str:
    return f"{value:.4f}"


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_model_card(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    partitions = report["chronological_partitions"]
    validation = report["validation"]
    test = report["test"]
    reliability = report["reliability"]
    policy = report["cost_policy"]
    captured_amount_row = (
        "| Fraud amount captured | "
        f"{_percent(validation['fraud_amount_capture_rate'])} | "
        f"{_percent(test['fraud_amount_capture_rate'])} |"
    )
    cost_reduction_row = (
        "| Cost reduction vs. no alerts | "
        f"{validation['cost_reduction_vs_no_alert']:.2f} | "
        f"{test['cost_reduction_vs_no_alert']:.2f} |"
    )
    raw_reliability = reliability["validation_uncalibrated"]
    calibrated_reliability = reliability["validation_calibrated"]
    test_reliability = reliability["test_calibrated"]
    brier_row = (
        f"| Brier score | {_metric(raw_reliability['brier_score'])} | "
        f"{_metric(calibrated_reliability['brier_score'])} | "
        f"{_metric(test_reliability['brier_score'])} |"
    )
    log_loss_row = (
        f"| Log loss | {_metric(raw_reliability['log_loss'])} | "
        f"{_metric(calibrated_reliability['log_loss'])} | "
        f"{_metric(test_reliability['log_loss'])} |"
    )
    calibration_error_row = (
        "| Expected calibration error | "
        f"{_metric(raw_reliability['expected_calibration_error'])} | "
        f"{_metric(calibrated_reliability['expected_calibration_error'])} | "
        f"{_metric(test_reliability['expected_calibration_error'])} |"
    )

    return f"""# RiskPulse Credit-Card Fraud Model Card

## Model overview

- **Version:** `{report["model_version"]}`
- **Model:** calibrated histogram gradient boosting
- **Calibration:** sigmoid calibration on a dedicated chronological period
- **Decision threshold:** `{report["selected_threshold"]:.8f}`
- **Dataset:** OpenML `{dataset["openml_data_id"]}`
- **Features:** `{dataset["feature_count"]}` numeric authorization-time features

## Intended use

This model is a portfolio and research demonstration of fraud-risk ranking,
probability calibration, and cost-sensitive alerting. It may support simulated
review prioritization. It is not validated for real financial decisions or
autonomous transaction declines.

## Evaluation protocol

Transactions are sorted by time before any split. Validation and test periods
never influence model fitting or probability calibration, and the test period
is not used for threshold selection.

| Period | Transactions | Purpose |
| --- | ---: | --- |
| Fit | {partitions["fit"]:,} | Fit the gradient-boosted classifier |
| Calibration | {partitions["calibration"]:,} | Fit the sigmoid probability mapping |
| Validation | {partitions["validation"]:,} | Select the cost-sensitive threshold |
| Test | {partitions["test"]:,} | Final one-time evaluation |

## Performance

| Metric | Validation | Test |
| --- | ---: | ---: |
| ROC-AUC | {_metric(validation["roc_auc"])} | {_metric(test["roc_auc"])} |
| PR-AUC | {_metric(validation["pr_auc"])} | {_metric(test["pr_auc"])} |
| Precision | {_percent(validation["precision"])} | {_percent(test["precision"])} |
| Recall | {_percent(validation["recall"])} | {_percent(test["recall"])} |
| F1 | {_metric(validation["f1"])} | {_metric(test["f1"])} |
| Alert rate | {_percent(validation["alert_rate"])} | {_percent(test["alert_rate"])} |
{captured_amount_row}
| Estimated cost | {validation["estimated_cost"]:.2f} | {test["estimated_cost"]:.2f} |
{cost_reduction_row}

## Probability reliability

| Metric | Validation, raw | Validation, calibrated | Test, calibrated |
| --- | ---: | ---: | ---: |
{brier_row}
{log_loss_row}
{calibration_error_row}

Lower values indicate more reliable probabilities. Calibration choices and the
decision threshold were fixed before the test period was evaluated.

## Cost policy

- False-positive review cost: `{policy["false_positive_review_cost"]:.2f}`
- Missed-fraud cost: transaction amount multiplied by
  `{policy["missed_fraud_amount_multiplier"]:.2f}`

These values make the decision rule reproducible, but they are modelling
assumptions rather than measured production costs.

## Data and limitations

- The dataset contains `{dataset["transactions"]:,}` transactions and
  `{dataset["frauds"]:,}` fraud labels, a fraud rate of
  `{_percent(dataset["fraud_rate"])}`.
- Transactions cover two days of European card activity from 2013, so the
  results may not transfer to newer populations, regions, or fraud patterns.
- V1 through V28 are anonymized PCA components. Their confidentiality protects
  cardholders but limits feature-level explanations.
- Protected demographic attributes are unavailable, so subgroup fairness
  cannot be audited from this dataset.
- Performance changes between validation and test periods indicate temporal
  drift risk. Production use would require monitoring and frequent
  reevaluation.
- Cost estimates exclude operational delays, investigation capacity, customer
  friction, and fraud-recovery rates.

## Reproducibility

Run `make calibrate` after `make data`. The command creates the versioned
Joblib artifact, JSON evaluation report, and this model card from the same
pipeline.
"""
