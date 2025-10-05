import io
import math
from datetime import datetime
from functools import lru_cache

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# ───────────────────────────
# 1) CONFIG: your two GSOM sheets (secondary MTM)
# ───────────────────────────
SHEET_TBILL_ID = "1AQFsSm5BXTD9fjM-MMv13tJx48zKLnSE_Z2S_mxbe7o"
SHEET_TBILL_GID = "603716852"

SHEET_TBOND_ID = "1qm-HDBK3g0T-oZAGCQutsIRF16932s-pQRTUqPZ-tL8"
SHEET_TBOND_GID = "1598348590"

def csv_url(sheet_id: str, gid: str) -> str:
    # public CSV export URL
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

URL_TBILL = csv_url(SHEET_TBILL_ID, SHEET_TBILL_GID)
URL_TBOND = csv_url(SHEET_TBOND_ID, SHEET_TBOND_GID)

# ───────────────────────────
# 2) LOADERS
# ───────────────────────────
@st.cache_data(ttl=60 * 30)  # refresh every 30 minutes
def load_csv(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df

def coerce_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map the columns we need and coerce types based on your layout screenshot."""
    # expected columns by (friendly_name -> possible headers)
    # (We keep this flexible so future changes don't break)
    COLMAP = {
        "Date": ["Date"],
        "ISIN": ["ISIN"],
        "InstrumentText": ["Securities", "Securities "],  # some sheets have trailing space
        "IssueDate": ["IssueDate", "Issue Date"],
        "MaturityDate": ["Maturity/Expiry Date", "Maturity / Expiry Date", "MaturityDate"],
        "IssuePrice": ["IssuePrice", "Issue Price"],
        "RemainingMaturity": ["RemainingMaturity", "Remaining Maturity"],
        "MarketYield": ["MarketYield", "Market Yield"],
        "MarketPrice": ["MarketPrice", "Market Price"],
        "Outstanding": ["Outstanding"],
        "Source": ["Source"],
    }

    # build new df with standardized columns if present
    out = pd.DataFrame()
    for new, candidates in COLMAP.items():
        col = next((c for c in candidates if c in df.columns), None)
        if col is not None:
            out[new] = df[col]
        else:
            out[new] = np.nan

    # types
    for c in ["Date", "IssueDate", "MaturityDate"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce", dayfirst=False)

    # numeric coercions
    for c in ["IssuePrice", "RemainingMaturity", "MarketYield", "MarketPrice", "Outstanding"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # strip instrument text
    if "InstrumentText" in out.columns:
        out["InstrumentText"] = out["InstrumentText"].astype(str).str.strip()

    return out

def add_helpers(df: pd.DataFrame, inst_type: str) -> pd.DataFrame:
    df = df.copy()
    df["Type"] = inst_type  # Bill or Bond
    # Market-time-to-maturity in years (this is what a yield curve uses)
    df["MaturityYears"] = df["RemainingMaturity"] / 365.0
    # keep only positive maturities & yields
    df = df[(df["MaturityYears"].notna()) & (df["MaturityYears"] > 0)]
    df = df[df["MarketYield"].notna()]
    # Month label for filtering
    df["Month"] = df["Date"].dt.to_period("M").astype(str)  # yyyy-mm
    return df

@st.cache_data(ttl=60 * 30)
def get_combined() -> pd.DataFrame:
    tbill_raw = load_csv(URL_TBILL)
    tbond_raw = load_csv(URL_TBOND)

    tbill = add_helpers(coerce_cols(tbill_raw), "Bill")
    tbond = add_helpers(coerce_cols(tbond_raw), "Bond")

    combined = pd.concat([tbill, tbond], ignore_index=True)
    # Remove duplicates if any (same Date + ISIN)
    combined = combined.sort_values(["Date", "ISIN"]).drop_duplicates(subset=["Date", "ISIN"], keep="last")
    return combined

# ───────────────────────────
# 3) APP UI
# ───────────────────────────
st.set_page_config(page_title="UCB AML – Bangladesh Yield Curve (GSOM)", layout="wide")
st.title("Bangladesh Govt Securities – Secondary Market Yield Curve (GSOM)")
st.caption("Data: Bangladesh Bank GSOM MTM for T-Bills & T-Bonds (auto-combined). X-axis = Remaining Maturity (years), Y-axis = Market Yield (%).")

df = get_combined()
if df.empty:
    st.error("No data loaded. Check sheet sharing & CSV export permissions.")
    st.stop()

# Sidebar controls
st.sidebar.header("Filters & View")
view_mode = st.sidebar.radio("View", ["Single date / Latest-in-month", "Compare two months"], index=0)

# common date/month choices
all_dates = sorted(df["Date"].dropna().unique())
all_months = sorted(df["Month"].dropna().unique())

# Helper: for a chosen month, pick latest date within that month
def latest_date_in_month(month_str: str) -> pd.Timestamp | None:
    subset = df.loc[df["Month"] == month_str, "Date"]
    return subset.max() if not subset.empty else None

def curve_for_date(d: pd.Timestamp) -> pd.DataFrame:
    return df[df["Date"] == d].copy()

# ───────────────────────────
# 4) SINGLE-DATE / LATEST-IN-MONTH
# ───────────────────────────
if view_mode == "Single date / Latest-in-month":
    tab1, tab2 = st.tabs(["Pick a specific date", "Pick a month (latest date)"])

    with tab1:
        pick = st.selectbox("Select a date", options=all_dates, format_func=lambda x: x.strftime("%Y-%m-%d"))
        cdf = curve_for_date(pick)

        st.subheader(f"Yield Curve – {pick.strftime('%Y-%m-%d')}")
        fig = px.scatter(
            cdf,
            x="MaturityYears",
            y="MarketYield",
            color="Type",
            hover_data=["ISIN", "InstrumentText", "RemainingMaturity", "MarketPrice", "Outstanding", "Source"],
            labels={"MaturityYears": "Remaining Maturity (years)", "MarketYield": "Yield (%)"},
        )
        fig.update_traces(mode="markers+lines")  # connect points to feel like a curve
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

        # data preview + export
        st.caption("Filtered data")
        st.dataframe(cdf.sort_values("MaturityYears")[["Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]], use_container_width=True)
        st.download_button(
            "Download CSV (this view)",
            cdf.to_csv(index=False).encode("utf-8"),
            file_name=f"yield_curve_{pick.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with tab2:
        pick_m = st.selectbox("Select a month (yyyy-mm)", options=all_months, index=len(all_months)-1 if all_months else 0)
        d_latest = latest_date_in_month(pick_m)
        if d_latest is None:
            st.warning("No data in that month.")
        else:
            cdf = curve_for_date(d_latest)
            st.subheader(f"Yield Curve – latest in {pick_m} (Date: {d_latest.strftime('%Y-%m-%d')})")
            fig = px.scatter(
                cdf,
                x="MaturityYears",
                y="MarketYield",
                color="Type",
                hover_data=["ISIN", "InstrumentText", "RemainingMaturity", "MarketPrice", "Outstanding", "Source"],
                labels={"MaturityYears": "Remaining Maturity (years)", "MarketYield": "Yield (%)"},
            )
            fig.update_traces(mode="markers+lines")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Filtered data")
            st.dataframe(cdf.sort_values("MaturityYears")[["Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]], use_container_width=True)
            st.download_button(
                "Download CSV (this view)",
                cdf.to_csv(index=False).encode("utf-8"),
                file_name=f"yield_curve_{pick_m}_{d_latest.strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

# ───────────────────────────
# 5) COMPARE TWO MONTHS
# ───────────────────────────
else:
    left, right = st.columns(2)
    with left:
        m1 = st.selectbox("Month A", options=all_months, index=max(0, len(all_months)-2))
        d1 = latest_date_in_month(m1)
        st.caption(f"Latest date in Month A = **{d1.strftime('%Y-%m-%d') if d1 else 'N/A'}**")

    with right:
        m2 = st.selectbox("Month B", options=all_months, index=len(all_months)-1)
        d2 = latest_date_in_month(m2)
        st.caption(f"Latest date in Month B = **{d2.strftime('%Y-%m-%d') if d2 else 'N/A'}**")

    if d1 is None or d2 is None:
        st.warning("Select months that contain data.")
        st.stop()

    c1 = curve_for_date(d1).assign(Which=f"{m1} (latest)")
    c2 = curve_for_date(d2).assign(Which=f"{m2} (latest)")
    comp = pd.concat([c1, c2], ignore_index=True)

    st.subheader(f"Yield Curve Comparison – {m1} vs {m2} (latest dates)")
    fig = px.scatter(
        comp,
        x="MaturityYears",
        y="MarketYield",
        color="Which",
        symbol="Type",
        hover_data=["Date","Type","ISIN","InstrumentText","RemainingMaturity","MarketPrice","Outstanding"],
        labels={"MaturityYears": "Remaining Maturity (years)", "MarketYield": "Yield (%)"},
    )
    fig.update_traces(mode="markers+lines")
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Data used in comparison")
    st.dataframe(
        comp.sort_values(["Which","MaturityYears"])[["Date","Which","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
        use_container_width=True
    )
    st.download_button(
        "Download CSV (comparison view)",
        comp.to_csv(index=False).encode("utf-8"),
        file_name=f"yield_curve_compare_{m1}_vs_{m2}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Tip: This chart uses **Remaining Maturity (years)** on the X-axis, which is correct for a yield curve from secondary market MTM data. Use the comparison view to see curve shifts month-to-month (e.g., steepening/flattening).")
