from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "retention_experiment.csv"
ARTIFACTS = ROOT / "artifacts"
FEATURES = ["tenure_months", "monthly_spend", "support_tickets", "usage_change", "plan"]


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def generate_experiment(rows: int = 14_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, rows)
    spend = np.clip(rng.lognormal(3.8, 0.48, rows), 12, 260)
    tickets = np.clip(rng.poisson(1.2, rows), 0, 8)
    usage_change = np.clip(rng.normal(-0.04, 0.24, rows), -0.9, 0.7)
    plan = rng.choice(["basic", "pro", "team"], rows, p=[0.48, 0.37, 0.15])
    treatment = rng.binomial(1, 0.5, rows)

    baseline_logit = (
        -1.35
        - 0.018 * tenure
        + 0.012 * (spend - 60)
        + 0.31 * tickets
        - 2.1 * usage_change
        + 0.25 * (plan == "basic")
    )
    baseline_churn = sigmoid(baseline_logit)
    treatment_effect = np.clip(
        0.02 + 0.18 * (usage_change < -0.15) + 0.09 * (tickets >= 2) - 0.08 * (tenure < 4),
        -0.05,
        0.32,
    )
    churn_probability = np.clip(baseline_churn - treatment * treatment_effect, 0.01, 0.98)
    churned = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "customer_id": [f"CUS-{i:05d}" for i in range(rows)],
            "tenure_months": tenure,
            "monthly_spend": spend.round(2),
            "support_tickets": tickets,
            "usage_change": usage_change.round(3),
            "plan": plan,
            "treatment": treatment,
            "churned": churned,
            "true_uplift": treatment_effect.round(4),
        }
    )


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        [("plan", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["plan"])],
        remainder="passthrough",
    )
    return Pipeline(
        [
            ("features", preprocessor),
            ("model", HistGradientBoostingClassifier(max_iter=140, learning_rate=0.07, max_leaf_nodes=18, random_state=42)),
        ]
    )


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    data = generate_experiment()
    data.to_csv(DATA_PATH, index=False)

    train = data.sample(frac=0.72, random_state=42)
    holdout = data.drop(train.index).copy()
    models: dict[int, Pipeline] = {}
    arm_auc: dict[str, float] = {}
    for arm in (0, 1):
        arm_train = train[train["treatment"] == arm]
        model = build_model()
        model.fit(arm_train[FEATURES], arm_train["churned"])
        models[arm] = model
        arm_holdout = holdout[holdout["treatment"] == arm]
        arm_auc[str(arm)] = round(
            roc_auc_score(arm_holdout["churned"], model.predict_proba(arm_holdout[FEATURES])[:, 1]), 4
        )

    holdout["predicted_uplift"] = (
        models[0].predict_proba(holdout[FEATURES])[:, 1]
        - models[1].predict_proba(holdout[FEATURES])[:, 1]
    )
    holdout["uplift_decile"] = pd.qcut(
        holdout["predicted_uplift"].rank(method="first"), 10, labels=False
    )
    holdout["uplift_decile"] = 10 - holdout["uplift_decile"]

    summary = holdout.groupby("uplift_decile", as_index=False).agg(
        customers=("customer_id", "size"),
        predicted_uplift=("predicted_uplift", "mean"),
        true_uplift=("true_uplift", "mean"),
    )
    summary.to_csv(ARTIFACTS / "uplift_by_decile.csv", index=False)

    contact_cost = 8.0
    retained_margin = holdout["monthly_spend"] * 4.2
    holdout["expected_incremental_value"] = holdout["predicted_uplift"] * retained_margin - contact_cost
    targeted = holdout[holdout["expected_incremental_value"] > 0]
    expected_profit = float(targeted["expected_incremental_value"].sum())

    metrics = {
        "holdout_customers": len(holdout),
        "targeted_customers": len(targeted),
        "target_rate": round(len(targeted) / len(holdout), 4),
        "expected_incremental_profit_usd": round(expected_profit, 2),
        "uplift_rank_correlation": round(holdout[["predicted_uplift", "true_uplift"]].corr().iloc[0, 1], 4),
        "arm_roc_auc": arm_auc,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].bar(summary["uplift_decile"], summary["predicted_uplift"], color="#0f766e")
    axes[0].plot(summary["uplift_decile"], summary["true_uplift"], color="#f97316", marker="o", label="True uplift")
    axes[0].invert_xaxis()
    axes[0].set(title="Uplift is concentrated in top deciles", xlabel="Uplift decile (10 = highest)", ylabel="Churn reduction")
    axes[0].legend(frameon=False)
    top = holdout.nlargest(max(1, len(targeted)), "expected_incremental_value")
    axes[1].hist(top["expected_incremental_value"], bins=24, color="#2563eb", alpha=0.85)
    axes[1].axvline(0, color="#111827", linewidth=1)
    axes[1].set(title="Profit-aware targeting", xlabel="Expected value per contact (USD)", ylabel="Customers")
    fig.suptitle("Causal Retention Lab | Holdout Results", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "retention_results.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

