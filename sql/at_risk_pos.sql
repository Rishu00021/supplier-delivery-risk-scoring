-- At-risk purchase orders: late deliveries, worst delay first,
-- with the vendor's overall on-time record for context.
SELECT
    p.po_id,
    p.vendor_id,
    v.vendor_name,
    p.material_category,
    p.order_date,
    p.promised_date,
    p.actual_date,
    p.delay_days,
    p.order_value,
    q.rejection_pct,
    s.on_time_pct AS vendor_on_time_pct
FROM purchase_orders p
JOIN vendors v ON v.vendor_id = p.vendor_id
LEFT JOIN quality_inspections q ON q.po_id = p.po_id
LEFT JOIN vendor_scorecard s ON s.vendor_id = p.vendor_id
WHERE p.delay_days > 3
ORDER BY p.delay_days DESC
LIMIT 100;
