from datetime import date
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ───────────────────────────
# CONFIG: GSOM Google Sheets (secondary MTM)
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

# Global calendar bounds (show all dates even if no data)
MIN_CAL_DATE = date(2000, 1, 1)
MAX_CAL_DATE = date.today()

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
    # Bills: days → years ; Bonds: already years
    df["MaturityYears"] = np.where(
        df["Type"].str.lower().eq("bill"),
        df["RemainingMaturity"] / 365.0,
        df["RemainingMaturity"],
    )
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
# HELPERS
# ───────────────────────────
def plot_curve(cdf: pd.DataFrame, title: str) -> go.Figure:
    """Sorted, spline-smoothed, fixed axes, one line per Type."""
    cdf = cdf.copy().sort_values(["Type", "MaturityYears"])
    fig = go.Figure()
    for typ, df_typ in cdf.groupby("Type"):
        df_typ = df_typ.sort_values("MaturityYears")
        fig.add_trace(
            go.Scatter(
                x=df_typ["MaturityYears"], y=df_typ["MarketYield"],
                mode="lines+markers", name=str(typ),
                line=dict(shape="spline"), connectgaps=True,
                text=df_typ["InstrumentText"],
                hovertemplate=("Type: %{name}<br>"
                               "Maturity: %{x:.3f} yrs<br>"
                               "Yield: %{y:.3f}%<br>"
                               "ISIN: %{text}<extra></extra>"),
            )
        )
    x_max = float(np.nanmax(cdf["MaturityYears"])) if not cdf["MaturityYears"].empty else 1.0
    x_pad = max(0.5, x_max * 0.05)
    fig.update_yaxes(range=[0, 16], title="Yield (%)")
    fig.update_xaxes(range=[0, x_max + x_pad], title="Remaining Maturity (years)")
    fig.update_layout(height=520, title=title)
    return fig

def latest_date_in_month(df: pd.DataFrame, month_str: str):
    s = df.loc[df["Month"] == month_str, "Date"]
    return s.max() if not s.empty else None

def nearest_available_date(all_dates: list, target: pd.Timestamp):
    if not len(all_dates):
        return None
    distances = pd.Series([abs((pd.Timestamp(d) - target).days) for d in all_dates], index=all_dates)
    return distances.idxmin()

def curve_for_date(df: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    return df[df["Date"] == d].copy()

def month_str_from_date(d: date) -> str:
    return pd.Timestamp(d).to_period("M").strftime("%Y-%m")

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

# Calendar uses global bounds (not limited by data range)
all_dates = sorted(df["Date"].dropna().unique())

st.sidebar.header("View mode")
view_mode = st.sidebar.radio(
    "Choose a view:",
    ["Single date", "Latest-in-month", "Compare two months", "Compare two dates"],
    index=0,
)

# ───────────────────────────
# 1) SINGLE DATE — Calendar picker (any date) → nearest trading date
# ───────────────────────────
if view_mode == "Single date":
    picked = st.date_input("Pick any date", value=MAX_CAL_DATE, min_value=MIN_CAL_DATE, max_value=MAX_CAL_DATE, key="single_cal")
    nearest = nearest_available_date(all_dates, pd.Timestamp(picked))
    if nearest is None:
        st.warning("No data available around the selected date.")
    else:
        if nearest.date() != picked:
            st.info(f"Adjusted to nearest available trading date: **{nearest.strftime('%Y-%m-%d')}**")
        cdf = curve_for_date(df, nearest)
        st.plotly_chart(plot_curve(cdf, f"Yield Curve — {nearest.strftime('%Y-%m-%d')}"), use_container_width=True)
        st.dataframe(
            cdf.sort_values("MaturityYears")[["Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
            use_container_width=True,
        )

# ───────────────────────────
# 2) LATEST-IN-MONTH — Calendar picker (any date) → latest date in that month
# ───────────────────────────
elif view_mode == "Latest-in-month":
    picked = st.date_input("Pick any date (we'll use the latest date in that month)",
                           value=MAX_CAL_DATE, min_value=MIN_CAL_DATE, max_value=MAX_CAL_DATE, key="month_cal")
    m = month_str_from_date(picked)
    d_latest = latest_date_in_month(df, m)
    if d_latest is None:
        st.warning(f"No data found in {m}. Try another month.")
    else:
        cdf = curve_for_date(df, d_latest)
        st.plotly_chart(
            plot_curve(cdf, f"Yield Curve — latest in {m} (Date: {d_latest.strftime('%Y-%m-%d')})"),
            use_container_width=True,
        )
        st.dataframe(
            cdf.sort_values("MaturityYears")[["Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
            use_container_width=True,
        )

# ───────────────────────────
# 3) COMPARE TWO MONTHS — Two calendars (any dates) → latest date in each month
# ───────────────────────────
elif view_mode == "Compare two months":
    c1, c2 = st.columns(2)
    with c1:
        pickA = st.date_input("Pick any date for Month A", value=MAX_CAL_DATE, min_value=MIN_CAL_DATE, max_value=MAX_CAL_DATE, key="mA_cal")
    with c2:
        pickB = st.date_input("Pick any date for Month B", value=MAX_CAL_DATE, min_value=MIN_CAL_DATE, max_value=MAX_CAL_DATE, key="mB_cal")

    m1 = month_str_from_date(pickA)
    m2 = month_str_from_date(pickB)
    d1 = latest_date_in_month(df, m1)
    d2 = latest_date_in_month(df, m2)

    if d1 is None or d2 is None:
        st.warning("No data in one (or both) of the selected months. Try different months.")
    else:
        c1df = curve_for_date(df, d1).assign(Which=f"{m1} (latest)")
        c2df = curve_for_date(df, d2).assign(Which=f"{m2} (latest)")
        comp = pd.concat([c1df, c2df], ignore_index=True)

        fig = go.Figure()
        for which, dfx in comp.groupby("Which"):
            dfx = dfx.sort_values("MaturityYears")
            fig.add_trace(
                go.Scatter(
                    x=dfx["MaturityYears"], y=dfx["MarketYield"],
                    mode="lines+markers", name=str(which),
                    line=dict(shape="spline"), connectgaps=True,
                    text=dfx["InstrumentText"],
                    hovertemplate=("%{name}<br>Maturity: %{x:.3f} yrs<br>"
                                   "Yield: %{y:.3f}%<br>ISIN: %{text}<extra></extra>"),
                )
            )
        x_max = float(np.nanmax(comp["MaturityYears"])) if not comp["MaturityYears"].empty else 1.0
        x_pad = max(0.5, x_max * 0.05)
        fig.update_yaxes(range=[0, 16], title="Yield (%)")
        fig.update_xaxes(range=[0, x_max + x_pad], title="Remaining Maturity (years)")
        fig.update_layout(height=560, title=f"Yield Curve Comparison — {m1} vs {m2}")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            comp.sort_values(["Which","MaturityYears"])[["Date","Which","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
            use_container_width=True,
        )

# ───────────────────────────
# 4) COMPARE TWO DATES — Two calendars (any dates) → nearest trading dates
# ───────────────────────────
else:
    c1, c2 = st.columns(2)
    with c1:
        dA = st.date_input("Date A", value=MAX_CAL_DATE, min_value=MIN_CAL_DATE, max_value=MAX_CAL_DATE, key="dA_cal")
    with c2:
        dB = st.date_input("Date B", value=MAX_CAL_DATE, min_value=MIN_CAL_DATE, max_value=MAX_CAL_DATE, key="dB_cal")

    nA = nearest_available_date(all_dates, pd.Timestamp(dA))
    nB = nearest_available_date(all_dates, pd.Timestamp(dB))

    if nA is None or nB is None:
        st.warning("No data available around the selected dates.")
    else:
        if nA.date() != dA:
            st.info(f"Date A adjusted to nearest trading date: **{nA.strftime('%Y-%m-%d')}**")
        if nB.date() != dB:
            st.info(f"Date B adjusted to nearest trading date: **{nB.strftime('%Y-%m-%d')}**")

        A = curve_for_date(df, nA).assign(Which=f"{nA.strftime('%Y-%m-%d')}")
        B = curve_for_date(df, nB).assign(Which=f"{nB.strftime('%Y-%m-%d')}")
        comp = pd.concat([A, B], ignore_index=True)

        fig = go.Figure()
        for which, dfx in comp.groupby("Which"):
            dfx = dfx.sort_values("MaturityYears")
            fig.add_trace(
                go.Scatter(
                    x=dfx["MaturityYears"], y=dfx["MarketYield"],
                    mode="lines+markers", name=str(which),
                    line=dict(shape="spline"), connectgaps=True,
                    text=dfx["InstrumentText"],
                    hovertemplate=("%{name}<br>Maturity: %{x:.3f} yrs<br>"
                                   "Yield: %{y:.3f}%<br>ISIN: %{text}<extra></extra>"),
                )
            )
        x_max = float(np.nanmax(comp["MaturityYears"])) if not comp["MaturityYears"].empty else 1.0
        x_pad = max(0.5, x_max * 0.05)
        fig.update_yaxes(range=[0, 16], title="Yield (%)")
        fig.update_xaxes(range=[0, x_max + x_pad], title="Remaining Maturity (years)")
        fig.update_layout(height=560, title=f"Yield Curve Comparison — {A['Which'].iloc[0]} vs {B['Which'].iloc[0]}")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            comp.sort_values(["Which","MaturityYears"])[["Date","Which","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
            use_container_width=True,
        )
# ───────────────────────────
# PRIMARY AUCTION VIEW – Monthly comparison
# ───────────────────────────
st.header("Primary Auction: Monthly Comparison")

# Primary Auction Sheet
SHEET_PRIMARY_ID = "1O5seVugWVYfCo7M7Zkn4VW6GltC77G1w0EsmhZEwNkk"
SHEET_PRIMARY_GID = "193103690"
URL_PRIMARY = f"https://docs.google.com/spreadsheets/d/{SHEET_PRIMARY_ID}/export?format=csv&gid={SHEET_PRIMARY_GID}"

@st.cache_data(ttl=60*30)
def load_primary() -> pd.DataFrame:
    try:
        df = pd.read_csv(URL_PRIMARY)
    except Exception as e:
        st.error(f"Failed to load Primary Auction sheet: {e}")
        return pd.DataFrame()

    # Column detection
    c_date = next((c for c in ["Issue Date", "IssueDate", "Date"] if c in df.columns), None)
    c_instr = next((c for c in ["Instrument", "Security", "Securities"] if c in df.columns), None)
    c_yld = next((c for c in ["Cut-off Yield (%)", "Cutoff Yield (%)", "Cut Off Yield (%)"] if c in df.columns), None)

    if not all([c_date, c_instr, c_yld]):
        st.warning("Primary sheet missing one or more required columns.")
        return pd.DataFrame()

    df = df.rename(columns={c_date:"IssueDate", c_instr:"Instrument", c_yld:"CutoffYield"})
    df["IssueDate"] = pd.to_datetime(df["IssueDate"], errors="coerce", dayfirst=True)
    df["CutoffYield"] = pd.to_numeric(df["CutoffYield"], errors="coerce")

    # Extract Tenor (in years)
    def parse_tenor(txt):
        if not isinstance(txt,str): return np.nan
        s = txt.lower()
        import re
        m = re.search(r'(\d+(?:\.\d+)?)\s*(day|days|month|months|year|years|yr|yrs|y)\b', s)
        if not m: return np.nan
        val, unit = float(m.group(1)), m.group(2)
        if "day" in unit: return val/365
        if "month" in unit: return val*30/365
        return val
    df["TenorYears"] = df["Instrument"].apply(parse_tenor)
    df["Month"] = df["IssueDate"].dt.to_period("M").astype(str)
    df = df.dropna(subset=["TenorYears","CutoffYield","IssueDate"])
    return df

pri_df = load_primary()
if not pri_df.empty:
    months = sorted(pri_df["Month"].unique())
    if len(months) >= 1:
        latest_month = months[-1]
        prev_month = months[-2] if len(months) > 1 else months[-1]
        st.subheader("Compare two months (latest auction per tenor)")
        c1, c2 = st.columns(2)
        with c1:
            m1 = st.selectbox("Month A", months, index=months.index(prev_month) if prev_month in months else 0)
        with c2:
            m2 = st.selectbox("Month B", months, index=months.index(latest_month))

        def month_subset(m):
            sub = pri_df[pri_df["Month"] == m].copy()
            if sub.empty: return sub
            idx = sub.groupby("TenorYears")["IssueDate"].idxmax()  # latest per tenor
            return sub.loc[idx].sort_values("TenorYears")

        dfA = month_subset(m1)
        dfB = month_subset(m2)

        if dfA.empty or dfB.empty:
            st.warning("No data found for one (or both) selected months.")
        else:
            dfA["Label"], dfB["Label"] = m1, m2
            df_all = pd.concat([dfA, dfB], ignore_index=True)

            fig = go.Figure()
            for label, grp in df_all.groupby("Label"):
                grp = grp.sort_values("TenorYears")
                fig.add_trace(
                    go.Scatter(
                        x=grp["TenorYears"], y=grp["CutoffYield"],
                        mode="lines+markers", name=str(label),
                        line=dict(shape="spline"),
                        text=grp["Instrument"],
                        hovertemplate="Instrument: %{text}<br>Tenor: %{x:.2f} yrs<br>Yield: %{y:.2f}%<extra></extra>",
                    )
                )
            fig.update_yaxes(range=[0, 16], title="Cut-off Yield (%)")
            fig.update_xaxes(title="Tenor (years)")
            fig.update_layout(height=520, title=f"Primary Auction Yield Curve Comparison — {m1} vs {m2}")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                df_all[["IssueDate","Instrument","TenorYears","CutoffYield","Label"]]
                .rename(columns={
                    "IssueDate":"Issue Date","TenorYears":"Tenor (years)",
                    "CutoffYield":"Cut-off Yield (%)","Label":"Month"
                })
                .sort_values(["Month","Tenor (years)"]),
                use_container_width=True,
            )
    else:
        st.warning("No valid months detected in Primary Auction sheet.")
else:
    st.error("Primary auction data unavailable or empty.")

st.markdown("---")
st.caption("Calendars allow any date (2000–today). If data isn’t available, we use the nearest trading date (or latest within that month).")
