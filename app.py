from datetime import date
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ───────────────────────────
# CONFIG: Google Sheets (secondary GSOM + primary auction)
# ───────────────────────────
# Secondary – T-Bill
SHEET_TBILL_ID = "1rD831MnVWUGlitw1jUmdwt5jQPrkKMwrQTWBy6P9tAs"
SHEET_TBILL_GID = "1446111990"
# Secondary – T-Bond
SHEET_TBOND_ID = "1ma25T-_yMlzdrzOYxAr2P6eu1gsbjPzq3jxF4PK-xtk"
SHEET_TBOND_GID = "632609507"
# Primary Auction (your shared sheet)
SHEET_PRIMARY_ID = "1O5seVugWVYfCo7M7Zkn4VW6GltC77G1w0EsmhZEwNkk"
SHEET_PRIMARY_GID = "193103690"

def csv_url(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

URL_TBILL   = csv_url(SHEET_TBILL_ID, SHEET_TBILL_GID)
URL_TBOND   = csv_url(SHEET_TBOND_ID, SHEET_TBOND_GID)
URL_PRIMARY = csv_url(SHEET_PRIMARY_ID, SHEET_PRIMARY_GID)

# Calendars always show all dates (even if data is missing)
MIN_CAL_DATE = date(2000, 1, 1)
MAX_CAL_DATE = date.today()

# ───────────────────────────
# LOADERS (cached)
# ───────────────────────────
@st.cache_data(ttl=60 * 30)
def load_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

# ───────────────────────────
# SECONDARY (GSOM) — NORMALIZERS
# ───────────────────────────
def gsom_coerce_cols(df: pd.DataFrame) -> pd.DataFrame:
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

def gsom_add_helpers(df: pd.DataFrame, inst_type: str) -> pd.DataFrame:
    df = df.copy()
    df["Type"] = inst_type  # Bill or Bond
    # Bills: RemainingMaturity in DAYS → convert to years; Bonds already YEARS
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
def get_secondary_combined() -> pd.DataFrame:
    tbill = gsom_add_helpers(gsom_coerce_cols(load_csv(URL_TBILL)), "Bill")
    tbond = gsom_add_helpers(gsom_coerce_cols(load_csv(URL_TBOND)), "Bond")
    combined = pd.concat([tbill, tbond], ignore_index=True)
    combined = combined.sort_values(["Date", "ISIN"]).drop_duplicates(["Date", "ISIN"], keep="last")
    return combined

# ───────────────────────────
# PRIMARY AUCTION — NORMALIZER
# ───────────────────────────
@st.cache_data(ttl=60 * 30)
def get_primary() -> pd.DataFrame:
    df = load_csv(URL_PRIMARY).copy()

    # Normalize headers (support slight variations)
    col_issue = next((c for c in ["Issue Date", "IssueDate", "Date"] if c in df.columns), None)
    col_instr = next((c for c in ["Instrument", "Securities", "Security"] if c in df.columns), None)
    col_yield = next((c for c in ["Cut-off Yield (%)", "Cutoff Yield (%)", "Cut Off Yield (%)", "Cut-off Yield"] if c in df.columns), None)

    if col_issue is None or col_instr is None or col_yield is None:
        return pd.DataFrame()  # fail gracefully if columns not found

    out = pd.DataFrame({
        "IssueDate": pd.to_datetime(df[col_issue], errors="coerce"),
        "Instrument": df[col_instr].astype(str).str.strip(),
        "CutoffYield": pd.to_numeric(df[col_yield], errors="coerce"),
        "Source": df[df.columns[-1]] if df.columns.size else "Primary",
    })

    # Tenor parsing → years
    out["TenorYears"] = out["Instrument"].apply(parse_tenor_years)
    out["Month"] = out["IssueDate"].dt.to_period("M").astype(str)
    # Keep only rows with a parsed tenor and yield
    out = out[out["TenorYears"].notna() & out["CutoffYield"].notna()]
    return out

def parse_tenor_years(text: str) -> float | None:
    """
    Parse strings like:
      '91 days T.Bill', '28 days T. Bill', '2 yr T. Bill', '5 yr T.Bond', '10 year T.Bond', '18 months'
    Return tenor in years.
    """
    if not isinstance(text, str):
        return None
    s = text.lower()

    # Number can be int or decimal, unit can be day/d, month/m, year/yr/y
    m = re.search(r'(\d+(?:\.\d+)?)\s*(days?|d\b|months?|mo|m\b|years?|yrs?|y\b)', s)
    if not m:
        # fallback for patterns like '364 days T.Bill' already handled above; else None
        return None
    val = float(m.group(1))
    unit = m.group(2)

    if re.search(r'day|d\b', unit):
        return val / 365.0
    if re.search(r'month|mo|m\b', unit):
        return (val * 30.0) / 365.0
    # years
    return val

# ───────────────────────────
# PLOTTING HELPERS
# ───────────────────────────
def plot_secondary_curve(cdf: pd.DataFrame, title: str) -> go.Figure:
    """Sorted, spline-smoothed, fixed axes, one line per Type (Bill/Bond)."""
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

def plot_primary_curve(cdf: pd.DataFrame, title: str) -> go.Figure:
    """Primary auction cut-off yields vs TenorYears (single blended curve)."""
    cdf = cdf.copy().sort_values("TenorYears")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cdf["TenorYears"], y=cdf["CutoffYield"],
            mode="lines+markers", name="Primary (cut-off)",
            line=dict(shape="spline"), connectgaps=True,
            text=cdf["Instrument"],
            hovertemplate=("Instrument: %{text}<br>"
                           "Maturity: %{x:.3f} yrs<br>"
                           "Cut-off: %{y:.3f}%<extra></extra>"),
        )
    )
    x_max = float(np.nanmax(cdf["TenorYears"])) if not cdf["TenorYears"].empty else 1.0
    x_pad = max(0.5, x_max * 0.05)
    fig.update_yaxes(range=[0, 16], title="Cut-off Yield (%)")
    fig.update_xaxes(range=[0, x_max + x_pad], title="Tenor (years)")
    fig.update_layout(height=520, title=title)
    return fig

# ───────────────────────────
# APP UI
# ───────────────────────────
st.set_page_config(page_title="Bangladesh Yield Curves – GSOM & Primary", layout="wide")
st.title("Bangladesh Govt Securities – Secondary (GSOM) & Primary Auction")

view = st.sidebar.radio(
    "View:",
    [
        "Secondary: Single date",
        "Secondary: Compare two dates",
        "Primary auction: Monthly",
    ],
    index=0,
)

# Load data up-front
sec_df = get_secondary_combined()
pri_df = get_primary()

if view.startswith("Secondary"):
    if sec_df.empty:
        st.error("No secondary (GSOM) data loaded. Check sheet sharing & CSV export permissions.")
        st.stop()

    # For calendars: allow any date; snap to available
    all_dates = sorted(sec_df["Date"].dropna().unique())

    def nearest_available_date(target: pd.Timestamp):
        if not len(all_dates):
            return None
        distances = pd.Series([abs((pd.Timestamp(d) - target).days) for d in all_dates], index=all_dates)
        return distances.idxmin()

    def curve_for_date(d: pd.Timestamp) -> pd.DataFrame:
        return sec_df[sec_df["Date"] == d].copy()

    if view == "Secondary: Single date":
        picked = st.date_input("Pick any date", value=MAX_CAL_DATE, min_value=MIN_CAL_DATE, max_value=MAX_CAL_DATE, key="single_sec")
        nearest = nearest_available_date(pd.Timestamp(picked))
        if nearest is None:
            st.warning("No data available around the selected date.")
        else:
            if nearest.date() != picked:
                st.info(f"Adjusted to nearest trading date: **{nearest.strftime('%Y-%m-%d')}**")
            cdf = curve_for_date(nearest)
            st.plotly_chart(plot_secondary_curve(cdf, f"Secondary Yield Curve — {nearest.strftime('%Y-%m-%d')}"),
                            use_container_width=True)
            st.dataframe(
                cdf.sort_values("MaturityYears")[["Date","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
                use_container_width=True,
            )

    else:  # "Secondary: Compare two dates"
        c1, c2 = st.columns(2)
        with c1:
            dA = st.date_input("Date A", value=MAX_CAL_DATE, min_value=MIN_CAL_DATE, max_value=MAX_CAL_DATE, key="secA")
        with c2:
            dB = st.date_input("Date B", value=MAX_CAL_DATE, min_value=MIN_CAL_DATE, max_value=MAX_CAL_DATE, key="secB")

        nA = nearest_available_date(pd.Timestamp(dA))
        nB = nearest_available_date(pd.Timestamp(dB))

        if nA is None or nB is None:
            st.warning("No data available around the selected dates.")
        else:
            if nA.date() != dA:
                st.info(f"Date A adjusted to nearest trading date: **{nA.strftime('%Y-%m-%d')}**")
            if nB.date() != dB:
                st.info(f"Date B adjusted to nearest trading date: **{nB.strftime('%Y-%m-%d')}**")

            A = curve_for_date(nA).assign(Which=f"{nA.strftime('%Y-%m-%d')}")
            B = curve_for_date(nB).assign(Which=f"{nB.strftime('%Y-%m-%d')}")
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
            fig.update_layout(height=560, title=f"Secondary Yield Curve Comparison — {A['Which'].iloc[0]} vs {B['Which'].iloc[0]}")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                comp.sort_values(["Which","MaturityYears"])[["Date","Which","Type","ISIN","InstrumentText","MaturityYears","MarketYield","RemainingMaturity","MarketPrice","Outstanding"]],
                use_container_width=True,
            )

else:
    # Primary auction: monthly
    if pri_df.empty:
        st.error("No primary auction data loaded. Check sheet sharing & CSV export permissions.")
        st.stop()

    picked = st.date_input(
        "Pick any date (we’ll use the month of this selection)",
        value=MAX_CAL_DATE, min_value=MIN_CAL_DATE, max_value=MAX_CAL_DATE, key="pri_month",
    )
    month_sel = pd.Timestamp(picked).to_period("M").strftime("%Y-%m")
    mdf = pri_df[pri_df["Month"] == month_sel].copy()

    if mdf.empty:
        st.warning(f"No primary auctions found in {month_sel}. Try another month.")
    else:
        # Use the latest auction per tenor within the month
        mdf = mdf.sort_values("IssueDate")  # ascending
        idx = mdf.groupby("TenorYears")["IssueDate"].idxmax()
        latest_per_tenor = mdf.loc[idx].sort_values("TenorYears")

        st.plotly_chart(
            plot_primary_curve(latest_per_tenor, f"Primary Auction Yield Curve — {month_sel} (latest per tenor)"),
            use_container_width=True,
        )
        st.dataframe(
            latest_per_tenor[["IssueDate","Instrument","TenorYears","CutoffYield","Source"]]
            .sort_values("TenorYears")
            .rename(columns={"CutoffYield":"Cut-off Yield (%)", "TenorYears":"Tenor (years)"}),
            use_container_width=True,
        )

st.markdown("---")
st.caption("Calendars accept any date. Secondary uses nearest trading day(s); Primary uses latest auction per tenor within the selected month. Y-axis fixed 0–16%.")
