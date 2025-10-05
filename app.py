import io
from datetime import date
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ───────────────────────────
# CONFIG: your GSOM Google Sheets (secondary MTM) – UPDATED
# ───────────────────────────
# T-Bill
SHEET_TBILL_ID = "1rD831MnVWUGlitw1jUmdwt5jQPrkKMwrQTWBy6P9tAs"
SHEET_TBILL_GID = "1446111990"

# T-Bond
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
    """Standardize the column names and types based on your Sheet layout."""
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
    """Add calculated fields (Type, MaturityYears, Month)."""
    df = df.copy()
    df["Type"] = inst_type  # Bill or Bond

    # Bills: RemainingMaturity is in DAYS → convert; Bonds already in YEARS
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
    cdf = cdf.copy().sort_values(["Type", "MaturityYears"])
    fig = go.Figure()
    for typ, df_typ in cdf.groupby("Type"):
        df_typ = df_typ.sort_values("MaturityYears")
        fig.add_trace(
            go.Scatter(
                x=df_typ["MaturityYears"],
                y=df_typ["MarketYield"],
                mode="lines+markers",
                name=str(typ),
                line=dict(shape="spline"),
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
    x_max = float(np.nanmax(cdf["MaturityYears"])) if not cdf["MaturityYears"].empty else 1.0
    x_padding = max(0.5, x_max * 0.05)
    fig.update_yaxes(range=[0, 16], title="Yield (%)")
    fig.update_xaxes(range=[0, x_max + x_padding], title="Remaining Maturity (years)")
    fig.update_layout(height=520, title=title)
    return fig

# ───────────────────────────
# APP UI
# ───────────────────────────
st.set_page_config(page_title="UCB AML – Bangladesh Yield Curve (GSOM)", layout="wide")
st.title("Bangladesh Govt Securities – Secondary Market Yield Curve (GSOM)")
st.caption("X-axis = Remaining Maturity (years). Y-axis = Market Yield (%).")

df = get_combined()
if df.empty:
    st.error("No data loaded. Check sheet sharing & CSV export permissions.")
    st.stop()

st.sidebar.header("Filters & View")
view_mode = st.sidebar.radio(
    "View mode:",
    ["Single date / Latest-in-month", "Compare two months", "Compare two dates"],
    0,
)

all_dates = sorted(df["Date"].dropna().unique())
all_months = sorted(df["Month"].dropna().unique())

def latest_date_in_month(month_str: str):
    s = df.loc[df["Month"] == month_str, "Date"]
    return s.max() if not s.empty else None

def nearest_available_date(target: pd.Timestamp):
    if not len(all_dates):
        return None
    distances = pd.Series([abs((pd.Timestamp(d) - target).days) for d in all_dates], index=all_dates)
    return distances.idxmin()

def curve_for_date(d: pd.Timestamp) -> pd.DataFrame:
    return df[df["Date"] == d].copy()

# ───────────────────────────
# 1) SINGLE-DATE / LATEST-IN-MONTH
# ───────────────────────────
if view_mode == "Single date / Latest-in-month":
    tab1, tab2 = st.tabs(["Pick a specific date", "Pick a month (latest date)"])

    with tab1:
        pick = st.selectbox("Select a date", options=all_dates, format_func=lambda x: x.strftime("%Y-%m-%d"))
        cdf = curve_for_date(pick)
        st.plotly_chart(plot_curve(cdf, f"Yield Curve – {pick.strftime('%Y-%m-%d')}"), use_container_width=True)

        st.caption("Filtered data")
        st.dataframe(
            cdf.sort_values("MaturityYears")[["Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
            use_container_width=True,
        )

    with tab2:
        pick_m = st.selectbox("Select a month (yyyy-mm)", options=all_months, index=len(all_months)-1 if all_months else 0)
        d_latest = latest_date_in_month(pick_m)
        if d_latest is None:
            st.warning("No data in that month.")
        else:
            cdf = curve_for_date(d_latest)
            st.plotly_chart(plot_curve(cdf, f"Yield Curve – latest in {pick_m} (Date: {d_latest.strftime('%Y-%m-%d')})"),
                            use_container_width=True)
            st.caption("Filtered data")
            st.dataframe(
                cdf.sort_values("MaturityYears")[["Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
                use_container_width=True,
            )

# ───────────────────────────
# 2) COMPARE TWO MONTHS (latest dates within each month)
# ───────────────────────────
elif view_mode == "Compare two months":
    left, right = st.columns(2)
    with left:
        m1 = st.selectbox("Month A", options=all_months, index=max(0, len(all_months)-2))
        d1 = latest_date_in_month(m1)
        st.caption(f"Latest date in Month A = **{d1.strftime('%Y-%m-%d') if d1 is not None else 'N/A'}**")
    with right:
        m2 = st.selectbox("Month B", options=all_months, index=len(all_months)-1)
        d2 = latest_date_in_month(m2)
        st.caption(f"Latest date in Month B = **{d2.strftime('%Y-%m-%d') if d2 is not None else 'N/A'}**")

    if d1 is None or d2 is None:
        st.warning("Select months that contain data.")
    else:
        c1 = curve_for_date(d1).assign(Which=f"{m1} (latest)")
        c2 = curve_for_date(d2).assign(Which=f"{m2} (latest)")
        comp = pd.concat([c1, c2], ignore_index=True)

        fig = go.Figure()
        for which, dfx in comp.groupby("Which"):
            dfx = dfx.sort_values("MaturityYears")
            fig.add_trace(
                go.Scatter(
                    x=dfx["MaturityYears"], y=dfx["MarketYield"],
                    mode="lines+markers", name=str(which),
                    line=dict(shape="spline"), connectgaps=True,
                    text=dfx["InstrumentText"],
                    hovertemplate=(
                        "%{name}<br>Maturity: %{x:.3f} yrs<br>Yield: %{y:.3f}%<br>"
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

# ───────────────────────────
# 3) COMPARE TWO DATES (Calendar widgets)
# ───────────────────────────
else:
    st.subheader("Compare two specific dates")
    min_d = pd.to_datetime(min(all_dates)).date() if len(all_dates) else date(2024,1,1)
    max_d = pd.to_datetime(max(all_dates)).date() if len(all_dates) else date.today()

    c1, c2 = st.columns(2)
    with c1:
        sel1 = st.date_input("Date A", value=max_d, min_value=min_d, max_value=max_d, key="dateA")
    with c2:
        sel2 = st.date_input("Date B", value=max_d, min_value=min_d, max_value=max_d, key="dateB")

    sel1_ts = pd.Timestamp(sel1)
    sel2_ts = pd.Timestamp(sel2)
    d1 = nearest_available_date(sel1_ts)
    d2 = nearest_available_date(sel2_ts)

    if d1 is None or d2 is None:
        st.warning("No data available around the selected dates.")
    else:
        if d1 != sel1_ts:
            st.info(f"Date A adjusted to nearest available: **{d1.strftime('%Y-%m-%d')}**")
        if d2 != sel2_ts:
            st.info(f"Date B adjusted to nearest available: **{d2.strftime('%Y-%m-%d')}**")

        a = curve_for_date(d1).assign(Which=f"{d1.strftime('%Y-%m-%d')}")
        b = curve_for_date(d2).assign(Which=f"{d2.strftime('%Y-%m-%d')}")
        comp = pd.concat([a, b], ignore_index=True)

        fig = go.Figure()
        for which, dfx in comp.groupby("Which"):
            dfx = dfx.sort_values("MaturityYears")
            fig.add_trace(
                go.Scatter(
                    x=dfx["MaturityYears"], y=dfx["MarketYield"],
                    mode="lines+markers", name=str(which),
                    line=dict(shape="spline"), connectgaps=True,
                    text=dfx["InstrumentText"],
                    hovertemplate=(
                        "%{name}<br>Maturity: %{x:.3f} yrs<br>Yield: %{y:.3f}%<br>"
                        "ISIN: %{text}<extra></extra>"
                    ),
                )
            )
        x_max = float(np.nanmax(comp["MaturityYears"])) if not comp["MaturityYears"].empty else 1.0
        x_padding = max(0.5, x_max * 0.05)
        fig.update_yaxes(range=[0, 16], title="Yield (%)")
        fig.update_xaxes(range=[0, x_max + x_padding], title="Remaining Maturity (years)")
        fig.update_layout(height=560, title=f"Yield Curve Comparison – {a['Which'].iloc[0]} vs {b['Which'].iloc[0]}")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Data used in comparison")
        st.dataframe(
            comp.sort_values(["Which","MaturityYears"])[["Date","Which","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
            use_container_width=True,
        )

st.markdown("---")
st.caption(
    "Sorted by maturity, spline lines, fixed Y-axis 0–16%. Bills: days→years; Bonds: years as-is."
)
