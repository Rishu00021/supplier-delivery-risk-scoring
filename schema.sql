-- Supplier Delivery Risk Scoring — MySQL-compatible DDL
-- All data in this project is SYNTHETIC. No real employer data is used.
-- Run against MySQL:  mysql -u <user> -p supplier_risk < schema.sql
-- The same tables are created in SQLite by src/load_to_db.py for the demo.

CREATE DATABASE IF NOT EXISTS supplier_risk
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;
USE supplier_risk;

CREATE TABLE IF NOT EXISTS vendors (
    vendor_id            VARCHAR(10)  NOT NULL,
    vendor_name          VARCHAR(255) NOT NULL,
    city                 VARCHAR(100),
    material_category    VARCHAR(100),
    base_on_time_rate    DECIMAL(5, 3),
    base_avg_delay_days  DECIMAL(6, 2),
    contract_start_date  DATE,
    PRIMARY KEY (vendor_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS purchase_orders (
    po_id               VARCHAR(12) NOT NULL,
    vendor_id           VARCHAR(10) NOT NULL,
    material_category   VARCHAR(100),
    order_date          DATE NOT NULL,
    promised_date       DATE NOT NULL,
    actual_date         DATE,
    lead_time_days      INT,
    qty_ordered         INT,
    unit_price          DECIMAL(12, 2),
    order_value         DECIMAL(14, 2),
    price_variance_pct  DECIMAL(6, 2),
    delay_days          INT,
    is_late             TINYINT(1) NOT NULL DEFAULT 0,
    order_month         TINYINT,
    PRIMARY KEY (po_id),
    INDEX idx_po_vendor (vendor_id),
    INDEX idx_po_order_date (order_date),
    CONSTRAINT fk_po_vendor FOREIGN KEY (vendor_id)
        REFERENCES vendors (vendor_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id   VARCHAR(12) NOT NULL,
    po_id         VARCHAR(12) NOT NULL,
    vendor_id     VARCHAR(10) NOT NULL,
    promised_date DATE,
    actual_date   DATE,
    delay_days    INT,
    qty_delivered INT,
    qty_ordered   INT,
    PRIMARY KEY (delivery_id),
    INDEX idx_del_po (po_id),
    CONSTRAINT fk_del_po FOREIGN KEY (po_id)
        REFERENCES purchase_orders (po_id),
    CONSTRAINT fk_del_vendor FOREIGN KEY (vendor_id)
        REFERENCES vendors (vendor_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS quality_inspections (
    inspection_id   VARCHAR(12) NOT NULL,
    po_id           VARCHAR(12) NOT NULL,
    vendor_id       VARCHAR(10) NOT NULL,
    qty_inspected   INT,
    qty_rejected    INT,
    rejection_pct   DECIMAL(6, 2),
    inspection_date DATE,
    PRIMARY KEY (inspection_id),
    INDEX idx_qi_po (po_id),
    CONSTRAINT fk_qi_po FOREIGN KEY (po_id)
        REFERENCES purchase_orders (po_id),
    CONSTRAINT fk_qi_vendor FOREIGN KEY (vendor_id)
        REFERENCES vendors (vendor_id)
) ENGINE=InnoDB;

-- Vendor scorecard: on-time %, average delay, rejection %
CREATE OR REPLACE VIEW vendor_scorecard AS
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
CREATE OR REPLACE VIEW monthly_late_trend AS
SELECT
    DATE_FORMAT(p.order_date, '%Y-%m') AS order_month,
    COUNT(*) AS total_pos,
    SUM(p.is_late) AS late_pos,
    ROUND(100.0 * SUM(p.is_late) / COUNT(*), 2) AS late_pct,
    ROUND(AVG(p.delay_days), 2) AS avg_delay_days
FROM purchase_orders p
GROUP BY DATE_FORMAT(p.order_date, '%Y-%m')
ORDER BY order_month;

-- At-risk purchase orders (late deliveries, worst first)
CREATE OR REPLACE VIEW at_risk_pos AS
SELECT
    p.po_id, p.vendor_id, v.vendor_name, p.material_category,
    p.order_date, p.promised_date, p.actual_date, p.delay_days,
    p.order_value, q.rejection_pct
FROM purchase_orders p
JOIN vendors v ON v.vendor_id = p.vendor_id
LEFT JOIN quality_inspections q ON q.po_id = p.po_id
WHERE p.delay_days > 3
ORDER BY p.delay_days DESC;
