"""Load the generated CSVs into SQLite (data/supplier.db) and create views.

SQLite is used for the runnable demo (Python stdlib sqlite3, no server).
The MySQL-compatible DDL for the same tables lives in schema.sql.
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "supplier.db"

TABLES = {
    "vendors": "vendors.csv",
    "purchase_orders": "purchase_orders.csv",
    "deliveries": "deliveries.csv",
    "quality_inspections": "quality_inspections.csv",
}

VIEWS_SQL = """
-- Vendor scorecard: on-time %, average delay, rejection %
CREATE VIEW IF NOT EXISTS vendor_scorecard AS
SELECT
    v.vendor_id,
    v.vendor_name,
    v.city,
    v.material_category,
    COUNT(p.po_id) AS total_pos,
    ROUND(100.0 * SUM(CASE WHEN p.delay_days <= 3 THEN 1 ELSE 0 END)
          / COUNT(p.po_id), 2) AS on_time_pct,
    ROUND(AVG(p.delay_days), 2) AS avg_delay_days,
    ROUND(100.0 * SUM(p.is_late) / COUNT(p.po_id), 2) AS late_pct,
    ROUND(AVG(q.rejection_pct), 2) AS avg_rejection_pct,
    ROUND(AVG(p.order_value), 2) AS avg_order_value
FROM vendors v
LEFT JOIN purchase_orders p ON p.vendor_id = v.vendor_id
LEFT JOIN quality_inspections q ON q.po_id = p.po_id
GROUP BY v.vendor_id, v.vendor_name, v.city, v.material_category;

-- Monthly late-delivery trend
CREATE VIEW IF NOT EXISTS monthly_late_trend AS
SELECT
    substr(p.order_date, 1, 7) AS order_month,
    COUNT(*) AS total_pos,
    SUM(p.is_late) AS late_pos,
    ROUND(100.0 * SUM(p.is_late) / COUNT(*), 2) AS late_pct,
    ROUND(AVG(p.delay_days), 2) AS avg_delay_days
FROM purchase_orders p
GROUP BY substr(p.order_date, 1, 7)
ORDER BY order_month;

-- At-risk purchase orders (late, or badly delayed)
CREATE VIEW IF NOT EXISTS at_risk_pos AS
SELECT
    p.po_id, p.vendor_id, v.vendor_name, p.material_category,
    p.order_date, p.promised_date, p.actual_date, p.delay_days,
    p.order_value, q.rejection_pct
FROM purchase_orders p
JOIN vendors v ON v.vendor_id = p.vendor_id
LEFT JOIN quality_inspections q ON q.po_id = p.po_id
WHERE p.delay_days > 3
ORDER BY p.delay_days DESC;
"""


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    try:
        for table, csv_name in TABLES.items():
            path = DATA_DIR / csv_name
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found — run: python src/generate_data.py first"
                )
            df = pd.read_csv(path)
            df.to_sql(table, conn, if_exists="replace", index=False)
            print(f"{table}: {len(df)} rows loaded")
        conn.executescript(VIEWS_SQL)
        conn.commit()
        for view in ("vendor_scorecard", "monthly_late_trend", "at_risk_pos"):
            n = conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
            print(f"view {view}: {n} rows")
    finally:
        conn.close()
    print(f"SQLite DB ready: {DB_PATH}")


if __name__ == "__main__":
    main()
