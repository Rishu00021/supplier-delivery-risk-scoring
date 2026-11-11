"""Train a RandomForest model predicting late delivery (>3 days late).

Target:  is_late = 1 when delay_days > 3, else 0.
Features: vendor history (past on-time rate, avg delay — computed from the
TRAIN split only, so no leakage into test), lead time, order value, quantity,
price variance, material category, order month.
Risk bands from predicted probability: High >= 0.60, Medium >= 0.35, Low < 0.35.

Outputs:
  models/risk_model.joblib   — fitted sklearn Pipeline
  results/metrics.json       — precision / recall / F1 / ROC-AUC + counts
  docs/feature_importance.png
  docs/late_delivery_trend.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
METRICS_PATH = ROOT / "results" / "metrics.json"
DOCS_DIR = ROOT / "docs"

NUMERIC = ["vendor_on_time_rate", "vendor_avg_delay", "lead_time_days",
           "order_value", "qty_ordered", "price_variance_pct", "order_month"]
CATEGORICAL = ["material_category"]
FEATURES = NUMERIC + CATEGORICAL


def risk_band(prob: float) -> str:
    if prob >= 0.60:
        return "High"
    if prob >= 0.35:
        return "Medium"
    return "Low"


def build_features(pos: pd.DataFrame) -> pd.DataFrame:
    """Split first, then compute vendor history from TRAIN only."""
    train, test = train_test_split(
        pos, test_size=0.20, random_state=42, stratify=pos["is_late"]
    )
    vendor_stats = (
        train.groupby("vendor_id")
        .agg(vendor_on_time_rate=("is_late", lambda s: 1 - s.mean()),
             vendor_avg_delay=("delay_days", "mean"))
        .reset_index()
    )
    # Fall back to global averages for any vendor missing from train.
    global_on_time = 1 - train["is_late"].mean()
    global_delay = float(train["delay_days"].mean())
    out = {}
    for name, split in (("train", train), ("test", test)):
        df = split.merge(vendor_stats, on="vendor_id", how="left")
        df["vendor_on_time_rate"] = df["vendor_on_time_rate"].fillna(global_on_time)
        df["vendor_avg_delay"] = df["vendor_avg_delay"].fillna(global_delay)
        out[name] = df
    return out["train"], out["test"]


def main():
    pos = pd.read_csv(DATA_DIR / "purchase_orders.csv")
    train, test = build_features(pos)

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    model = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1,
                class_weight="balanced_subsample")),
        ]
    )
    X_train, y_train = train[FEATURES], train["is_late"]
    X_test, y_test = test[FEATURES], test["is_late"]
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "model": "RandomForestClassifier (n_estimators=200, class_weight=balanced_subsample)",
        "target": "is_late (delay_days > 3)",
        "n_rows": int(len(pos)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "late_rate_overall_pct": round(100 * float(pos["is_late"].mean()), 2),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "confusion_matrix": {
            "tn": int(confusion_matrix(y_test, pred)[0, 0]),
            "fp": int(confusion_matrix(y_test, pred)[0, 1]),
            "fn": int(confusion_matrix(y_test, pred)[1, 0]),
            "tp": int(confusion_matrix(y_test, pred)[1, 1]),
        },
    }
    print(classification_report(y_test, pred, zero_division=0))
    print(metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"model saved: {MODEL_PATH}")
    print(f"metrics saved: {METRICS_PATH}")

    # --- Feature importance chart ---
    ohe = model.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL))
    feat_names = NUMERIC + cat_names
    importances = model.named_steps["clf"].feature_importances_
    order = np.argsort(importances)[::-1]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh([feat_names[i] for i in order][::-1],
             importances[order][::-1], color="#2f6f8f")
    plt.xlabel("Importance")
    plt.title("Feature Importance — Late Delivery Risk (RandomForest)")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # --- Late-delivery trend chart ---
    monthly = pos.copy()
    monthly["order_month"] = pd.to_datetime(monthly["order_date"]).dt.to_period("M").astype(str)
    trend = monthly.groupby("order_month")["is_late"].mean() * 100
    plt.figure(figsize=(10, 4.5))
    plt.plot(trend.index, trend.values, marker="o", color="#c0392b")
    plt.ylabel("Late POs (%)")
    plt.xlabel("Order month")
    plt.title("Late-Delivery Rate by Month (synthetic data)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "late_delivery_trend.png", dpi=150)
    plt.close()
    print(f"charts saved: {DOCS_DIR / 'feature_importance.png'}, "
          f"{DOCS_DIR / 'late_delivery_trend.png'}")


if __name__ == "__main__":
    main()
