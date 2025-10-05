import io
import math
from datetime import datetime
from functools import lru_cache

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ───────────────────────────
# CONFIG: your two GSOM sheets (secondary MTM)
# ───────────────────────────
SHEET_TBILL_ID = "1rD831MnVWUGlitw1jUmdwt5jQPrkKMwrQTWBy6P9tAs"
SHEET_TBILL_GID = "1446111990"

SHEET_TBOND_ID = "1ma25T-_yMlzdrzOYxAr2P6eu1gsbjPzq3jxF4PK-xtk"
SHEET_TBOND_GID = "632609507"

def csv_url(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

URL_TBILL = csv_url(SHEET_TBILL_ID, SHEET_TBILL_GID)
URL_TBOND = csv_url(SHEET_TBOND_ID, SHEET_TBOND_GID)

# ───────────────────────────
# LOADERS & NORMALIZERS
# ───────────────────────────
@st.cache_data(ttl=60 * 30)
def load_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def coerce_cols(df: pd.DataFrame) -> pd.DataFrame:
    COLMAP = {
        "Date": ["Date"],
        "ISIN": ["ISIN"],
        "InstrumentText": ["Securities", "Securities "],
        "IssueDate": ["IssueDate", "Issue Date"],
        "MaturityDate": ["Maturity/Expiry Date", "Maturity / Expiry Date", "MaturityDate"],
        "IssuePrice": ["IssuePrice", "Issue Price"],
        "RemainingMaturity": ["RemainingMaturity", "Remaining Maturity"],
        "MarketYield": ["MarketYield", "Market Yield"],
        "MarketPrice": ["MarketPrice", "Market Price"],
        "Outstanding": ["Outstanding"],
        "Source": ["Source"],
    }
    out = pd.DataFrame()
    for new, candidates in COLMAP.items():
        col = next((c for c in candidates if c in df.columns), None)
        out[new] = df[col] if col else np.nan

    for c in ["Date", "IssueDate", "MaturityDate"]:
        out[c] = pd.to_datetime(out[c], errors="coerce")

    for c in ["IssuePrice", "RemainingMaturity", "MarketYield", "MarketPrice", "Outstanding"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["InstrumentText"] = out["InstrumentText"].astype(str).str.strip()
    return out

def add_helpers(df: pd.DataFrame, inst_type: str) -> pd.DataFrame:
    df = df.copy()
    df["Type"] = inst_type  # Bill or Bond

    # Convert only for Bills (days -> years). Bonds already in years.
    if inst_type.lower() == "bill":
        df["MaturityYears"] = df["RemainingMaturity"] / 365.0
    else:
        df["MaturityYears"] = df["RemainingMaturity"]

    df = df[(df["MaturityYears"].notna()) & (df["MaturityYears"] > 0)]
    df = df[df["MarketYield"].notna()]
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df

@st.cache_data(ttl=60 * 30)
def get_combined() -> pd.DataFrame:
    tbill = add_helpers(coerce_cols(load_csv(URL_TBILL)), "Bill")
    tbond = add_helpers(coerce_cols(load_csv(URL_TBOND)), "Bond")
    combined = pd.concat([tbill, tbond], ignore_index=True)
    combined = combined.sort_values(["Date", "ISIN"]).drop_duplicates(["Date", "ISIN"], keep="last")
    return combined

# ───────────────────────────
# PLOTTING
# ───────────────────────────
def plot_curve(cdf: pd.DataFrame, title: str) -> go.Figure:
    """Sorted, spline-smoothed, fixed axes, one line per Type."""
    # sort within each Type by maturity
    cdf = cdf.copy()
    cdf = cdf.sort_values(["Type", "MaturityYears"])

    fig = go.Figure()

    for typ, df_typ in cdf.groupby("Type"):
        df_typ = df_typ.sort_values("MaturityYears")
        fig.add_trace(
            go.Scatter(
                x=df_typ["MaturityYears"],
                y=df_typ["MarketYield"],
                mode="lines+markers",
                name=str(typ),
                line=dict(shape="spline"),   # smooth line
                connectgaps=True,
                text=df_typ["InstrumentText"],
                hovertemplate=(
                    "Type: %{name}<br>"
                    "Maturity: %{x:.3f} yrs<br>"
                    "Yield: %{y:.3f}%<br>"
                    "ISIN: %{text}<extra></extra>"
                ),
            )
        )

    # axis ranges
    x_max = float(np.nanmax(cdf["MaturityYears"])) if not cdf["MaturityYears"].empty else 1.0
    x_padding = max(0.5, x_max * 0.05)

    fig.update_yaxes(range=[0, 16], title="Yield (%)")          # ← fixed 0–16%
    fig.update_xaxes(range=[0, x_max + x_padding], title="Remaining Maturity (years)")
    fig.update_layout(height=520, title=title)
    return fig

# ───────────────────────────
# APP UI
# ───────────────────────────
st.set_page_config(page_title="UCB AML – Bangladesh Yield Curve (GSOM)", layout="wide")
st.title("Bangladesh Govt Securities – Secondary Market Yield Curve (GSOM)")
st.caption(
    "Data: Bangladesh Bank GSOM MTM for T-Bills & T-Bonds (auto-combined). "
    "X-axis = Remaining Maturity (years), Y-axis = Market Yield (%)."
)

df = get_combined()
if df.empty:
    st.error("No data loaded. Check sheet sharing & CSV export permissions.")
    st.stop()

st.sidebar.header("Filters & View")
view_mode = st.sidebar.radio("View mode:", ["Single date / Latest-in-month", "Compare two months"], 0)

all_dates = sorted(df["Date"].dropna().unique())
all_months = sorted(df["Month"].dropna().unique())

def latest_date_in_month(month_str: str) -> pd.Timestamp | None:
    s = df.loc[df["Month"] == month_str, "Date"]
    return s.max() if not s.empty else None

def curve_for_date(d: pd.Timestamp) -> pd.DataFrame:
    return df[df["Date"] == d].copy()

# ───────────────────────────
# SINGLE-DATE / LATEST-IN-MONTH
# ───────────────────────────
if view_mode == "Single date / Latest-in-month":
    tab1, tab2 = st.tabs(["Pick a specific date", "Pick a month (latest date)"])

    with tab1:
        pick = st.selectbox("Select a date", options=all_dates, format_func=lambda x: x.strftime("%Y-%m-%d"))
        cdf = curve_for_date(pick)
        fig = plot_curve(cdf, f"Yield Curve – {pick.strftime('%Y-%m-%d')}")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Filtered data")
        st.dataframe(
            cdf.sort_values("MaturityYears")[["Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
            use_container_width=True,
        )
        st.download_button(
            "Download CSV (this view)",
            cdf.to_csv(index=False).encode("utf-8"),
            file_name=f"yield_curve_{pick.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with tab2:
        pick_m = st.selectbox("Select a month (yyyy-mm)", options=all_months, index=len(all_months)-1)
        d_latest = latest_date_in_month(pick_m)
        if d_latest is None:
            st.warning("No data in that month.")
        else:
            cdf = curve_for_date(d_latest)
            fig = plot_curve(cdf, f"Yield Curve – latest in {pick_m} (Date: {d_latest.strftime('%Y-%m-%d')})")
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Filtered data")
            st.dataframe(
                cdf.sort_values("MaturityYears")[["Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
                use_container_width=True,
            )
            st.download_button(
                "Download CSV (this view)",
                cdf.to_csv(index=False).encode("utf-8"),
                file_name=f"yield_curve_{pick_m}_{d_latest.strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

# ───────────────────────────
# COMPARE TWO MONTHS
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

    # Build a combined figure with two lines (one per 'Which'), still sorted by maturity
    fig = go.Figure()
    for which, dfx in comp.groupby("Which"):
        dfx = dfx.sort_values("MaturityYears")
        fig.add_trace(
            go.Scatter(
                x=dfx["MaturityYears"],
                y=dfx["MarketYield"],
                mode="lines+markers",
                name=str(which),
                line=dict(shape="spline"),
                connectgaps=True,
                text=dfx["InstrumentText"],
                hovertemplate=(
                    "%{name}<br>"
                    "Maturity: %{x:.3f} yrs<br>"
                    "Yield: %{y:.3f}%<br>"
                    "ISIN: %{text}<extra></extra>"
                ),
            )
        )
    x_max = float(np.nanmax(comp["MaturityYears"])) if not comp["MaturityYears"].empty else 1.0
    x_padding = max(0.5, x_max * 0.05)
    fig.update_yaxes(range=[0, 16], title="Yield (%)")
    fig.update_xaxes(range=[0, x_max + x_padding], title="Remaining Maturity (years)")
    fig.update_layout(height=560, title=f"Yield Curve Comparison – {m1} vs {m2} (latest dates)")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Data used in comparison")
    st.dataframe(
        comp.sort_values(["Which","MaturityYears"])[["Date","Which","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
        use_container_width=True,
    )
    st.download_button(
        "Download CSV (comparison view)",
        comp.to_csv(index=False).encode("utf-8"),
        file_name=f"yield_curve_compare_{m1}_vs_{m2}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Lines are sorted by maturity and drawn with a spline. Y-axis fixed to 0–16%. "
    "Bills use days→years; bonds are already in years."
)
