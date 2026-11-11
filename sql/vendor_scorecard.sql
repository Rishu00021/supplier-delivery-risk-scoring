-- Vendor scorecard: on-time %, average delay, rejection % per vendor.
-- Use: run against MySQL (see schema.sql) or SQLite via src/load_to_db.py.
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
GROUP BY v.vendor_id, v.vendor_name, v.city, v.material_category
ORDER BY on_time_pct ASC, avg_delay_days DESC;
