"""Smoke tests — fast checks that the pipeline outputs exist and are sane."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_data_files_exist_and_have_rows():
    for name, min_rows in (("vendors.csv", 30), ("purchase_orders.csv", 1500),
                           ("deliveries.csv", 1500),
                           ("quality_inspections.csv", 1500)):
        df = pd.read_csv(ROOT / "data" / name)
        assert len(df) >= min_rows, f"{name} has only {len(df)} rows"


def test_purchase_orders_schema():
    df = pd.read_csv(ROOT / "data" / "purchase_orders.csv")
    for col in ("po_id", "vendor_id", "order_date", "promised_date",
                "actual_date", "delay_days", "is_late", "order_value"):
        assert col in df.columns, f"missing column {col}"
    assert df["is_late"].isin([0, 1]).all()
    assert df["delay_days"].notna().all()


def test_metrics_json():
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    for key in ("precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"
    assert metrics["roc_auc"] > 0.6, "ROC-AUC should beat random guessing"


def test_model_and_charts_exist():
    assert (ROOT / "models" / "risk_model.joblib").exists()
    assert (ROOT / "docs" / "feature_importance.png").exists()
    assert (ROOT / "docs" / "late_delivery_trend.png").exists()
    assert (ROOT / "docs" / "feature_importance.png").stat().st_size > 1000


def test_schema_sql_has_tables_and_view():
    sql = (ROOT / "schema.sql").read_text()
    for token in ("CREATE TABLE IF NOT EXISTS vendors",
                  "CREATE TABLE IF NOT EXISTS purchase_orders",
                  "CREATE TABLE IF NOT EXISTS deliveries",
                  "CREATE TABLE IF NOT EXISTS quality_inspections",
                  "vendor_scorecard"):
        assert token in sql, f"schema.sql missing: {token}"
