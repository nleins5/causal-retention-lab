# Causal Retention Lab

> Who should receive a retention offer when the objective is incremental profit, not churn probability?

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![CI](https://github.com/nleins5/causal-retention-lab/actions/workflows/pipeline.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-0f766e)

This project turns a randomized SaaS retention experiment into a targeting policy. Two response models estimate each customer's conditional treatment effect, and a simple value model converts uplift into an operational decision.

![Holdout results](artifacts/retention_results.png)

## Business result

The pipeline separates customers who are likely to churn from customers whose behavior can actually be changed. It reports the expected incremental profit after contact cost, the recommended target rate, and uplift calibration by decile.

| Holdout result | Value |
|---|---:|
| Customers targeted | 2,234 of 3,920 |
| Expected incremental profit | $70,120.78 |
| Uplift rank correlation | 0.6465 |
| Control / treatment ROC AUC | 0.6729 / 0.7228 |

## What this demonstrates

- Causal framing and heterogeneous treatment effects
- Leakage-safe train/holdout evaluation
- Profit-aware decision thresholds
- Reproducible data generation, pipeline artifacts, Docker, and CI

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
make install
make run
```

Outputs are written to `artifacts/`. The generated experiment is synthetic by design: treatment assignment and ground-truth uplift are known, so ranking quality can be evaluated honestly without sharing customer data.

## Repository map

```text
src/retention_lab/       simulation, modeling, policy evaluation
data/                    reproducible experiment snapshot
artifacts/               metrics, decision table, and chart
.github/workflows/       CI reproduction job
```

## Interview discussion

Useful extensions include doubly robust learners, confidence intervals for policy value, treatment budget constraints, and monitoring for changes in treatment propensity.

