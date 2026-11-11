"""Generate synthetic supplier / purchase-order data.

All data is SYNTHETIC — invented vendors, dates and numbers for a demo.
No real employer data is used anywhere in this repo.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

RANDOM_SEED = 42
N_VENDORS = 30
N_POS = 2000
START = pd.Timestamp("2025-05-01")
END = pd.Timestamp("2026-10-31")

MATERIAL_CATEGORIES = [
    "Raw Material",
    "Packaging",
    "Electrical Components",
    "Mechanical Parts",
    "Chemicals",
    "Fasteners",
    "Electronics",
    "Consumables",
]

VENDOR_NAMES = [
    "Apex Polymers Pvt Ltd", "Shakti Steel Traders", "Nexa Packaging Solutions",
    "Vikram Electricals", "Sagar Chemicals Ltd", "Precision Fasteners India",
    "Orbit Electronics", "Kaveri Consumables", "Meridian Raw Materials",
    "TrueLine Mechanicals", "BrightPack Industries", "Zenith Copper Works",
    "Anand Alloys Pvt Ltd", "Prime Plastics Co", "Surya Cables Ltd",
    "Nova Bearings India", "Kiran Industrial Supply", "Delta Adhesives Ltd",
    "Everest Rubber Products", "Pioneer Tools & Hardware", "Lotus Paper Mills",
    "Galaxy Semiconductors", "Ridge Valve Systems", "Harmony Textiles Pvt Ltd",
    "Swift Logistics Packaging", "Crest Aluminium Ltd", "Vertex Motors Pvt Ltd",
    "BlueStar Lubricants", "IronGate Forgings", "EcoFoam Insulation Ltd",
]


def generate(seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    vendors = []
    for i, name in enumerate(VENDOR_NAMES, start=1):
        # Each vendor has an underlying reliability level (hidden driver).
        base_on_time = float(rng.uniform(0.55, 0.97))
        base_delay = float(rng.uniform(0.5, 6.0))
        vendors.append(
            {
                "vendor_id": f"V{i:03d}",
                "vendor_name": name,
                "city": random.choice(
                    ["Delhi", "Noida", "Gurugram", "Faridabad", "Ghaziabad",
                     "Mumbai", "Pune", "Chennai", "Bengaluru", "Ahmedabad"]
                ),
                "material_category": random.choice(MATERIAL_CATEGORIES),
                "base_on_time_rate": round(base_on_time, 3),
                "base_avg_delay_days": round(base_delay, 2),
                "contract_start_date": (
                    START - pd.Timedelta(days=int(rng.integers(180, 900)))
                ).date().isoformat(),
            }
        )
    vendors_df = pd.DataFrame(vendors)
    reliability = {
        v["vendor_id"]: (v["base_on_time_rate"], v["base_avg_delay_days"])
        for v in vendors
    }

    po_rows, delivery_rows, inspection_rows = [], [], []
    order_dates = pd.to_datetime(
        rng.integers(START.value // 10**9, END.value // 10**9, size=N_POS), unit="s"
    )
    for n in range(N_POS):
        vendor = vendors_df.iloc[int(rng.integers(0, N_VENDORS))]
        vid = vendor["vendor_id"]
        on_time_rate, avg_delay = reliability[vid]
        category = str(vendor["material_category"])
        order_date = pd.Timestamp(order_dates[n]).normalize()
        lead_time = int(rng.integers(7, 46))
        promised = order_date + pd.Timedelta(days=lead_time)
        qty = int(rng.integers(50, 5001))
        unit_price = round(float(rng.uniform(20, 2500)), 2)
        order_value = round(qty * unit_price, 2)
        price_var = round(float(rng.normal(0, 3.5)), 2)

        # Delay driver: unreliable vendors + long lead times + Dec/Jan
        # seasonality + large orders are more likely to slip.
        month = int(order_date.month)
        seasonal = 1.3 if month in (12, 1) else 1.0
        lam = max(0.4, (1 - on_time_rate) * 8 * seasonal
                  + 0.02 * lead_time + (0.4 if order_value > 500_000 else 0))
        delay = int(rng.poisson(lam)) - int(rng.integers(0, 2))
        delay = max(-3, delay)  # negative = early
        actual = promised + pd.Timedelta(days=delay)
        po_id = f"PO-{n + 1:05d}"

        po_rows.append(
            {
                "po_id": po_id,
                "vendor_id": vid,
                "material_category": category,
                "order_date": order_date.date().isoformat(),
                "promised_date": promised.date().isoformat(),
                "actual_date": actual.date().isoformat(),
                "lead_time_days": lead_time,
                "qty_ordered": qty,
                "unit_price": unit_price,
                "order_value": order_value,
                "price_variance_pct": price_var,
                "delay_days": delay,
                "is_late": int(delay > 3),
                "order_month": month,
            }
        )
        delivered_qty = qty if delay <= 10 else max(0, qty - int(rng.integers(0, qty // 10 + 1)))
        delivery_rows.append(
            {
                "delivery_id": f"D-{n + 1:05d}",
                "po_id": po_id,
                "vendor_id": vid,
                "promised_date": promised.date().isoformat(),
                "actual_date": actual.date().isoformat(),
                "delay_days": delay,
                "qty_delivered": int(delivered_qty),
                "qty_ordered": qty,
            }
        )
        inspected = int(delivered_qty * 0.2) if delivered_qty else 0
        rej_rate = float(np.clip(rng.normal(0.03 + (1 - on_time_rate) * 0.08, 0.02), 0, 0.25))
        rejected = int(round(inspected * rej_rate))
        inspection_rows.append(
            {
                "inspection_id": f"Q-{n + 1:05d}",
                "po_id": po_id,
                "vendor_id": vid,
                "qty_inspected": inspected,
                "qty_rejected": rejected,
                "rejection_pct": round(100 * rejected / inspected, 2) if inspected else 0.0,
                "inspection_date": actual.date().isoformat(),
            }
        )

    return (
        vendors_df,
        pd.DataFrame(po_rows),
        pd.DataFrame(delivery_rows),
        pd.DataFrame(inspection_rows),
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    vendors, pos, deliveries, inspections = generate()
    vendors.to_csv(DATA_DIR / "vendors.csv", index=False)
    pos.to_csv(DATA_DIR / "purchase_orders.csv", index=False)
    deliveries.to_csv(DATA_DIR / "deliveries.csv", index=False)
    inspections.to_csv(DATA_DIR / "quality_inspections.csv", index=False)
    print(f"vendors: {len(vendors)} rows -> data/vendors.csv")
    print(f"purchase_orders: {len(pos)} rows -> data/purchase_orders.csv")
    print(f"deliveries: {len(deliveries)} rows -> data/deliveries.csv")
    print(f"quality_inspections: {len(inspections)} rows -> data/quality_inspections.csv")
    print(f"late POs (>3 days): {int(pos['is_late'].sum())} "
          f"({100 * pos['is_late'].mean():.1f}%)")


if __name__ == "__main__":
    main()
