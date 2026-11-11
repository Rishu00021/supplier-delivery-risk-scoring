-- Monthly late-delivery trend + worst vendors per month.
-- Q1: late-delivery rate and average delay by order month.
SELECT
    substr(p.order_date, 1, 7) AS order_month,
    COUNT(*) AS total_pos,
    SUM(p.is_late) AS late_pos,
    ROUND(100.0 * SUM(p.is_late) / COUNT(*), 2) AS late_pct,
    ROUND(AVG(p.delay_days), 2) AS avg_delay_days
FROM purchase_orders p
GROUP BY substr(p.order_date, 1, 7)
ORDER BY order_month;

-- Q2: vendors with the highest rejection % (min 20 POs) — quality risk.
SELECT
    v.vendor_id,
    v.vendor_name,
    COUNT(p.po_id) AS total_pos,
    ROUND(AVG(q.rejection_pct), 2) AS avg_rejection_pct,
    ROUND(100.0 * SUM(p.is_late) / COUNT(p.po_id), 2) AS late_pct
FROM vendors v
JOIN purchase_orders p ON p.vendor_id = v.vendor_id
JOIN quality_inspections q ON q.po_id = p.po_id
GROUP BY v.vendor_id, v.vendor_name
HAVING COUNT(p.po_id) >= 20
ORDER BY avg_rejection_pct DESC
LIMIT 10;
