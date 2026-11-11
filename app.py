"""Supplier Delivery Risk Scoring — Streamlit dashboard.

Reads the synthetic CSVs in data/. If models/risk_model.joblib exists it
scores POs with the trained RandomForest; otherwise it falls back to a
transparent vendor-history heuristic and tells the user to run the pipeline:
    python src/generate_data.py && python src/load_to_db.py && python src/train.py
No hardcoded absolute paths — everything is relative to the repo root.
"""
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "risk_model.joblib"

FEATURES = ["vendor_on_time_rate", "vendor_avg_delay", "lead_time_days",
            "order_value", "qty_ordered", "price_variance_pct", "order_month"]


def risk_band(prob: float) -> str:
    if prob >= 0.60:
        return "High"
    if prob >= 0.35:
        return "Medium"
    return "Low"


@st.cache_data
def load_data():
    vendors = pd.read_csv(DATA_DIR / "vendors.csv")
    pos = pd.read_csv(DATA_DIR / "purchase_orders.csv",
                      parse_dates=["order_date", "promised_date", "actual_date"])
    inspections = pd.read_csv(DATA_DIR / "quality_inspections.csv")
    return vendors, pos, inspections


def score_pos(pos: pd.DataFrame) -> pd.DataFrame:
    """Add risk_probability + risk_band columns (model if present, else heuristic)."""
    df = pos.copy()
    stats = (
        df.groupby("vendor_id")
        .agg(vendor_on_time_rate=("is_late", lambda s: 1 - s.mean()),
             vendor_avg_delay=("delay_days", "mean"))
        .reset_index()
    )
    df = df.merge(stats, on="vendor_id", how="left")
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        feats = df[FEATURES + ["material_category"]]
        df["risk_probability"] = model.predict_proba(feats)[:, 1]
        df["model_used"] = "RandomForest"
    else:
        # Heuristic fallback: vendors that are late often + long lead times.
        df["risk_probability"] = (
            (1 - df["vendor_on_time_rate"]) * 0.7
            + (df["lead_time_days"] / 45) * 0.2
            + (df["order_value"] > 500_000).astype(float) * 0.1
        ).clip(0, 1)
        df["model_used"] = "heuristic (run python src/train.py for the ML model)"
    df["risk_band"] = df["risk_probability"].apply(risk_band)
    return df


def build_scorecard(pos: pd.DataFrame, inspections: pd.DataFrame) -> pd.DataFrame:
    insp = inspections.groupby("po_id")["rejection_pct"].mean().rename("rejection_pct")
    df = pos.merge(insp, left_on="po_id", right_index=True, how="left")
    card = (
        df.groupby(["vendor_id", "material_category"])
        .agg(total_pos=("po_id", "count"),
             on_time_pct=("is_late", lambda s: round(100 * (1 - s.mean()), 1)),
             avg_delay_days=("delay_days", lambda s: round(s.mean(), 2)),
             late_pct=("is_late", lambda s: round(100 * s.mean(), 1)),
             avg_rejection_pct=("rejection_pct", lambda s: round(s.mean(), 2)),
             high_risk_pos=("risk_band", lambda s: int((s == "High").sum())))
        .reset_index()
    )
    order = {"High": 0, "Medium": 1, "Low": 2}
    card["worst_band"] = card["late_pct"].apply(
        lambda p: "High" if p >= 40 else ("Medium" if p >= 25 else "Low"))
    card = card.sort_values(["late_pct", "avg_delay_days"],
                            ascending=False).reset_index(drop=True)
    return card


def recommendations(card: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for _, r in card.iterrows():
        if r["late_pct"] >= 40:
            action = "Escalate: put on watchlist, split orders / dual-source, expedite open POs."
        elif r["late_pct"] >= 25:
            action = "Monitor: confirm promised dates weekly, keep buffer stock."
        elif r["avg_rejection_pct"] >= 5:
            action = "Quality review: raise inspection rate, discuss rejections with vendor."
        else:
            action = "No action: reliable vendor, keep current terms."
        recs.append({
            "Vendor": r["vendor_id"],
            "Late %": r["late_pct"],
            "Avg delay (days)": r["avg_delay_days"],
            "Rejection %": r["avg_rejection_pct"],
            "Recommendation": action,
        })
    return pd.DataFrame(recs)


def main():
    st.set_page_config(page_title="Supplier Delivery Risk Scoring", layout="wide")
    st.title("Supplier Delivery Risk Scoring")
    st.write(
        "Which vendors are likely to deliver late — and which open purchase "
        "orders need attention? All data here is **synthetic** (demo only)."
    )

    try:
        vendors, pos, inspections = load_data()
    except FileNotFoundError:
        st.error("Data files not found. Run first: `python src/generate_data.py`")
        st.stop()

    scored = score_pos(pos)
    card = build_scorecard(scored, inspections)

    if not MODEL_PATH.exists():
        st.warning("Trained model not found — showing heuristic risk scores. "
                   "Run `python src/train.py` to enable the RandomForest model.")

    # --- Filters ---
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        vendor_filter = st.selectbox(
            "Vendor", ["All"] + sorted(vendors["vendor_name"].tolist()))
    with col_f2:
        cat_filter = st.selectbox(
            "Material category",
            ["All"] + sorted(pos["material_category"].unique().tolist()))
    with col_f3:
        risk_filter = st.multiselect(
            "Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])

    view = scored.copy()
    if vendor_filter != "All":
        vid = vendors.loc[vendors["vendor_name"] == vendor_filter, "vendor_id"].iloc[0]
        view = view[view["vendor_id"] == vid]
    if cat_filter != "All":
        view = view[view["material_category"] == cat_filter]
    view = view[view["risk_band"].isin(risk_filter)]

    # --- KPI cards ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Purchase orders", f"{len(view):,}")
    k2.metric("Late POs (>3 days)", f"{int(view['is_late'].sum()):,}",
              f"{100 * view['is_late'].mean():.1f}% of POs" if len(view) else "")
    k3.metric("Avg delay (days)", f"{view['delay_days'].mean():.1f}" if len(view) else "—")
    k4.metric("High-risk POs", f"{int((view['risk_band'] == 'High').sum()):,}")

    # --- Charts ---
    st.subheader("Late-delivery trend")
    monthly = view.copy()
    monthly["month"] = monthly["order_date"].dt.to_period("M").astype(str)
    trend = monthly.groupby("month")["is_late"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trend.index, trend.values, marker="o", color="#c0392b")
    ax.set_ylabel("Late POs (%)")
    ax.set_xlabel("Order month")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Late rate by material category")
    by_cat = view.groupby("material_category")["is_late"].mean().sort_values() * 100
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.barh(by_cat.index, by_cat.values, color="#2f6f8f")
    ax2.set_xlabel("Late POs (%)")
    fig2.tight_layout()
    st.pyplot(fig2)

    # --- Vendor scorecard ---
    st.subheader("Vendor scorecard (sorted by late %)")
    st.dataframe(card, use_container_width=True)

    # --- At-risk POs ---
    st.subheader("At-risk purchase orders")
    at_risk = view[view["risk_band"] == "High"].sort_values(
        "risk_probability", ascending=False)
    cols = ["po_id", "vendor_id", "material_category", "order_date",
            "promised_date", "delay_days", "order_value",
            "risk_probability", "risk_band"]
    st.dataframe(at_risk[cols].head(100), use_container_width=True)

    # --- Vendor detail ---
    st.subheader("Vendor detail")
    detail_vid = st.selectbox("Pick a vendor", sorted(vendors["vendor_id"].tolist()),
                              key="detail_vendor")
    vpos = scored[scored["vendor_id"] == detail_vid]
    vname = vendors.loc[vendors["vendor_id"] == detail_vid, "vendor_name"].iloc[0]
    st.write(f"**{vname} ({detail_vid})** — {len(vpos)} POs, "
             f"{100 * vpos['is_late'].mean():.1f}% late, "
             f"avg delay {vpos['delay_days'].mean():.1f} days, "
             f"high-risk POs: {int((vpos['risk_band'] == 'High').sum())}")
    st.dataframe(vpos[["po_id", "material_category", "order_date", "promised_date",
                       "delay_days", "order_value", "risk_probability",
                       "risk_band"]].sort_values("order_date", ascending=False).head(50),
                 use_container_width=True)

    # --- Recommendations ---
    st.subheader("Actionable recommendations")
    st.dataframe(recommendations(card), use_container_width=True)


if __name__ == "__main__":
    main()
