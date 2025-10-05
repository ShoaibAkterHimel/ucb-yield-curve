from datetime import date
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import re

# ───────────────────────────
# CONFIG: Google Sheets (Secondary GSOM + Primary Auction)
# ───────────────────────────
SHEET_TBILL_ID = "1rD831MnVWUGlitw1jUmdwt5jQPrkKMwrQTWBy6P9tAs"
SHEET_TBILL_GID = "1446111990"
SHEET_TBOND_ID = "1ma25T-_yMlzdrzOYxAr2P6eu1gsbjPzq3jxF4PK-xtk"
SHEET_TBOND_GID = "632609507"
SHEET_PRIMARY_ID = "1O5seVugWVYfCo7M7Zkn4VW6GltC77G1w0EsmhZEwNkk"
SHEET_PRIMARY_GID = "193103690"

def csv_url(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

URL_TBILL = csv_url(SHEET_TBILL_ID, SHEET_TBILL_GID)
URL_TBOND = csv_url(SHEET_TBOND_ID, SHEET_TBOND_GID)
URL_PRIMARY = csv_url(SHEET_PRIMARY_ID, SHEET_PRIMARY_GID)

MIN_CAL_DATE = date(2000, 1, 1)
MAX_CAL_DATE = date.today()

# ───────────────────────────
# LOADERS (cached)
# ───────────────────────────
@st.cache_data(ttl=60 * 30)
def load_csv(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ───────────────────────────
# SECONDARY (GSOM)
# ───────────────────────────
def coerce_cols(df: pd.DataFrame) -> pd.DataFrame:
    COLMAP = {
        "Date": ["Date"],
        "ISIN": ["ISIN"],
        "InstrumentText": ["Securities", "Securities "],
        "RemainingMaturity": ["RemainingMaturity", "Remaining Maturity"],
        "MarketYield": ["MarketYield", "Market Yield"],
        "MarketPrice": ["MarketPrice", "Market Price"],
    }
    out = pd.DataFrame()
    for new, candidates in COLMAP.items():
        col = next((c for c in candidates if c in df.columns), None)
        out[new] = df[col] if col else np.nan

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for c in ["RemainingMaturity", "MarketYield", "MarketPrice"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["InstrumentText"] = out["InstrumentText"].astype(str).str.strip()
    return out

def add_helpers(df: pd.DataFrame, inst_type: str) -> pd.DataFrame:
    df = df.copy()
    df["Type"] = inst_type
    df["MaturityYears"] = np.where(df["Type"].str.lower().eq("bill"),
                                   df["RemainingMaturity"]/365.0,
                                   df["RemainingMaturity"])
    df = df[(df["MaturityYears"] > 0) & (df["MarketYield"].notna())]
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df

@st.cache_data(ttl=60 * 30)
def get_secondary() -> pd.DataFrame:
    tbill = add_helpers(coerce_cols(load_csv(URL_TBILL)), "Bill")
    tbond = add_helpers(coerce_cols(load_csv(URL_TBOND)), "Bond")
    combined = pd.concat([tbill, tbond], ignore_index=True)
    combined = combined.sort_values(["Date", "ISIN"]).drop_duplicates(["Date", "ISIN"], keep="last")
    return combined

# ───────────────────────────
# PRIMARY
# ───────────────────────────
@st.cache_data(ttl=60 * 30)
def get_primary() -> pd.DataFrame:
    df = load_csv(URL_PRIMARY)
    if df.empty: return df
    c_date = next((c for c in ["Issue Date","IssueDate","Date"] if c in df.columns), None)
    c_instr = next((c for c in ["Instrument","Security","Securities"] if c in df.columns), None)
    c_yld = next((c for c in ["Cut-off Yield (%)","Cutoff Yield (%)","Cut Off Yield (%)"] if c in df.columns), None)
    if not all([c_date, c_instr, c_yld]): return pd.DataFrame()
    df = df.rename(columns={c_date:"IssueDate", c_instr:"Instrument", c_yld:"CutoffYield"})
    df["IssueDate"] = pd.to_datetime(df["IssueDate"], errors="coerce", dayfirst=True)
    df["CutoffYield"] = pd.to_numeric(df["CutoffYield"], errors="coerce")

    def parse_tenor(txt):
        if not isinstance(txt,str): return np.nan
        s = txt.lower()
        m = re.search(r'(\d+(?:\.\d+)?)\s*(day|days|month|months|year|years|yr|yrs|y)\b', s)
        if not m: return np.nan
        val, unit = float(m.group(1)), m.group(2)
        if "day" in unit: return val / 365
        if "month" in unit: return val * 30 / 365
        return val

    df["TenorYears"] = df["Instrument"].apply(parse_tenor)
    df["Month"] = df["IssueDate"].dt.to_period("M").astype(str)
    df = df.dropna(subset=["TenorYears","CutoffYield","IssueDate"])
    return df

# ───────────────────────────
# PLOT HELPERS
# ───────────────────────────
def plot_secondary_single(df, title):
    fig = go.Figure()
    for typ, g in df.groupby("Type"):
        g = g.sort_values("MaturityYears")
        fig.add_trace(go.Scatter(x=g["MaturityYears"], y=g["MarketYield"],
                                 mode="lines+markers", name=typ,
                                 line=dict(shape="spline")))
    fig.update_yaxes(range=[0,16], title="Yield (%)")
    fig.update_xaxes(title="Remaining Maturity (years)")
    fig.update_layout(title=title, height=520)
    return fig

def plot_secondary_compare(df, title):
    """One line per (Label × Type) so dates have different colors/styles."""
    # Assign dash styles per label to improve separation
    labels = list(dict.fromkeys(df["Label"].tolist()))
    dashes = ["solid", "dash", "dot", "dashdot", "longdash"]
    dash_map = {lab: dashes[i % len(dashes)] for i, lab in enumerate(labels)}

    fig = go.Figure()
    for (label, typ), g in df.groupby(["Label", "Type"]):
        g = g.sort_values("MaturityYears")
        fig.add_trace(
            go.Scatter(
                x=g["MaturityYears"], y=g["MarketYield"],
                mode="lines+markers",
                name=f"{label} — {typ}",
                line=dict(shape="spline", dash=dash_map[label]),
                marker=dict(size=5),
                hovertemplate=("Date: " + str(label) +
                               "<br>Type: %{text}<br>Maturity: %{x:.3f} yrs<br>"
                               "Yield: %{y:.3f}%<extra></extra>"),
                text=[typ]*len(g),
            )
        )
    fig.update_yaxes(range=[0,16], title="Yield (%)")
    fig.update_xaxes(title="Remaining Maturity (years)")
    fig.update_layout(title=title, height=560)
    return fig

def plot_primary(df, title):
    fig = go.Figure()
    for label, g in df.groupby("Label"):
        g = g.sort_values("TenorYears")
        fig.add_trace(go.Scatter(x=g["TenorYears"], y=g["CutoffYield"],
                                 mode="lines+markers", name=label,
                                 line=dict(shape="spline"),
                                 text=g["Instrument"],
                                 hovertemplate="Instrument: %{text}<br>Tenor: %{x:.2f} yrs<br>Yield: %{y:.2f}%<extra></extra>"))
    fig.update_yaxes(range=[0,16], title="Cut-off Yield (%)")
    fig.update_xaxes(title="Tenor (years)")
    fig.update_layout(title=title, height=520)
    return fig

# ───────────────────────────
# APP UI
# ───────────────────────────
st.set_page_config(page_title="UCB AML – Yield Curve Dashboard", layout="wide")
st.title("Bangladesh Govt Securities – Yield Curve Dashboard")

view_mode = st.sidebar.radio(
    "Select View Mode:",
    [
        "Secondary: Single Date",
        "Secondary: Compare Two Dates",
        "Secondary: Compare Two Months",
        "Primary Auction: Compare Two Months"
    ],
    index=0,
)

# ─────────────── SECONDARY VIEWS ───────────────
if view_mode.startswith("Secondary"):
    df = get_secondary()
    if df.empty:
        st.error("No secondary data found.")
        st.stop()
    all_dates = sorted(df["Date"].dropna().unique())

    def nearest(target):
        if not len(all_dates): return None
        dist = pd.Series([abs((pd.Timestamp(d)-target).days) for d in all_dates], index=all_dates)
        return dist.idxmin()

    def curve(d): return df[df["Date"] == d]

    # Single Date
    if "Single" in view_mode:
        d = st.date_input("Pick Date", value=MAX_CAL_DATE, min_value=MIN_CAL_DATE, max_value=MAX_CAL_DATE)
        n = nearest(pd.Timestamp(d))
        cdf = curve(n)
        st.plotly_chart(plot_secondary_single(cdf, f"Secondary Yield Curve — {n.date()}"), use_container_width=True)

    # Compare Two Dates  ❗ uses plot_secondary_compare
    elif "Two Dates" in view_mode:
        c1, c2 = st.columns(2)
        with c1: d1 = st.date_input("Date A", value=MAX_CAL_DATE)
        with c2: d2 = st.date_input("Date B", value=MAX_CAL_DATE)
        n1, n2 = nearest(pd.Timestamp(d1)), nearest(pd.Timestamp(d2))
        df1 = curve(n1).assign(Label=str(n1.date()))
        df2 = curve(n2).assign(Label=str(n2.date()))
        merged = pd.concat([df1, df2], ignore_index=True)
        st.plotly_chart(
            plot_secondary_compare(merged, f"Secondary Comparison — {n1.date()} vs {n2.date()}"),
            use_container_width=True
        )

    # Compare Two Months  ❗ also uses plot_secondary_compare
    else:
        def month_str(d): return pd.Timestamp(d).to_period("M").strftime("%Y-%m")
        def latest_in_month(m): s = df.loc[df["Month"]==m,"Date"]; return s.max() if not s.empty else None
        c1,c2 = st.columns(2)
        with c1: m1 = month_str(st.date_input("Pick Month A", value=MAX_CAL_DATE))
        with c2: m2 = month_str(st.date_input("Pick Month B", value=MAX_CAL_DATE))
        d1, d2 = latest_in_month(m1), latest_in_month(m2)
        if not d1 or not d2: st.warning("No data found."); st.stop()
        df1 = curve(d1).assign(Label=f"{m1} (latest)")
        df2 = curve(d2).assign(Label=f"{m2} (latest)")
        merged = pd.concat([df1, df2], ignore_index=True)
        st.plotly_chart(
            plot_secondary_compare(merged, f"Secondary Comparison — {m1} vs {m2}"),
            use_container_width=True
        )

# ─────────────── PRIMARY VIEW ───────────────
else:
    pri = get_primary()
    if pri.empty:
        st.error("Primary auction data unavailable or empty.")
        st.stop()
    months = sorted(pri["Month"].unique())
    if not months:
        st.warning("No valid months in primary auction data.")
        st.stop()

    latest = months[-1]
    prev = months[-2] if len(months) > 1 else months[-1]

    c1, c2 = st.columns(2)
    with c1: m1 = st.selectbox("Month A", months, index=months.index(prev) if prev in months else 0)
    with c2: m2 = st.selectbox("Month B", months, index=months.index(latest))

    def month_df(m):
        sub = pri[pri["Month"] == m].copy()
        if sub.empty: return sub
        idx = sub.groupby("TenorYears")["IssueDate"].idxmax()
        return sub.loc[idx].sort_values("TenorYears")

    dfA, dfB = month_df(m1), month_df(m2)
    if dfA.empty or dfB.empty:
        st.warning("No data found for one (or both) months.")
        st.stop()

    dfA["Label"], dfB["Label"] = m1, m2
    merged = pd.concat([dfA, dfB], ignore_index=True)
    st.plotly_chart(plot_primary(merged, f"Primary Auction Comparison — {m1} vs {m2}"), use_container_width=True)
    st.dataframe(
        merged[["IssueDate","Instrument","TenorYears","CutoffYield","Label"]]
        .rename(columns={"IssueDate":"Issue Date","TenorYears":"Tenor (years)","CutoffYield":"Cut-off Yield (%)","Label":"Month"})
        .sort_values(["Month","Tenor (years)"]),
        use_container_width=True
    )

st.markdown("---")
st.caption("Secondary = GSOM market yields (by date or month). Primary = auction cut-off yields (compare months). Each curve uses separate colors/styles; Y-axis fixed 0–16%.")
