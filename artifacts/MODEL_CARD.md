# RiskPulse Credit-Card Fraud Model Card

## Model overview

- **Version:** `creditcard-hgb-sigmoid-20260727-cbbc97d3`
- **Model:** calibrated histogram gradient boosting
- **Calibration:** sigmoid calibration on a dedicated chronological period
- **Decision threshold:** `0.07350753`
- **Dataset:** OpenML `1597`
- **Features:** `30` numeric authorization-time features

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
| Fit | 159,491 | Fit the gradient-boosted classifier |
| Calibration | 39,873 | Fit the sigmoid probability mapping |
| Validation | 42,721 | Select the cost-sensitive threshold |
| Test | 42,722 | Final one-time evaluation |

## Performance

| Metric | Validation | Test |
| --- | ---: | ---: |
| ROC-AUC | 0.9825 | 0.9634 |
| PR-AUC | 0.8700 | 0.7435 |
| Precision | 86.79% | 82.61% |
| Recall | 82.14% | 73.08% |
| F1 | 0.8440 | 0.7755 |
| Alert rate | 0.12% | 0.11% |
| Fraud amount captured | 95.23% | 60.90% |
| Estimated cost | 435.58 | 2452.30 |
| Cost reduction vs. no alerts | 7962.12 | 3716.58 |

## Probability reliability

| Metric | Validation, raw | Validation, calibrated | Test, calibrated |
| --- | ---: | ---: | ---: |
| Brier score | 0.0005 | 0.0004 | 0.0004 |
| Log loss | 0.0025 | 0.0021 | 0.0029 |
| Expected calibration error | 0.0006 | 0.0003 | 0.0003 |

Lower values indicate more reliable probabilities. Calibration choices and the
decision threshold were fixed before the test period was evaluated.

## Cost policy

- False-positive review cost: `5.00`
- Missed-fraud cost: transaction amount multiplied by
  `1.00`

These values make the decision rule reproducible, but they are modelling
assumptions rather than measured production costs.

## Data and limitations

- The dataset contains `284,807` transactions and
  `492` fraud labels, a fraud rate of
  `0.17%`.
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
