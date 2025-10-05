from datetime import date
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ───────────────────────────
# CONFIG: Google Sheets (secondary GSOM + primary auction)
# ───────────────────────────
SHEET_TBILL_ID = "1rD831MnVWUGlitw1jUmdwt5jQPrkKMwrQTWBy6P9tAs"
SHEET_TBILL_GID = "1446111990"
SHEET_TBOND_ID = "1ma25T-_yMlzdrzOYxAr2P6eu1gsbjPzq3jxF4PK-xtk"
SHEET_TBOND_GID = "632609507"
SHEET_PRIMARY_ID = "1O5seVugWVYfCo7M7Zkn4VW6GltC77G1w0EsmhZEwNkk"
SHEET_PRIMARY_GID = "193103690"

def csv_url(sheet_id, gid):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

URL_TBILL   = csv_url(SHEET_TBILL_ID, SHEET_TBILL_GID)
URL_TBOND   = csv_url(SHEET_TBOND_ID, SHEET_TBOND_GID)
URL_PRIMARY = csv_url(SHEET_PRIMARY_ID, SHEET_PRIMARY_GID)

MIN_CAL_DATE = date(2000, 1, 1)
MAX_CAL_DATE = date.today()

# ───────────────────────────
# HELPERS / LOADERS
# ───────────────────────────
@st.cache_data(ttl=60*30)
def load_csv(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Could not load {url}: {e}")
        return pd.DataFrame()

def parse_tenor_years(text):
    if not isinstance(text,str): return None
    s=text.lower()
    m=re.search(r'(\d+(?:\.\d+)?)\s*(days?|d\b|months?|mo|m\b|years?|yrs?|y\b)',s)
    if not m: return None
    val=float(m.group(1)); unit=m.group(2)
    if re.search(r'day|d\b',unit): return val/365
    if re.search(r'month|mo|m\b',unit): return (val*30)/365
    return val

# ───────────────────────────
# SECONDARY (GSOM)
# ───────────────────────────
def gsom_coerce_cols(df):
    cmap={"Date":["Date"],"ISIN":["ISIN"],"InstrumentText":["Securities","Securities "],
          "RemainingMaturity":["RemainingMaturity","Remaining Maturity"],
          "MarketYield":["MarketYield","Market Yield"],"MarketPrice":["MarketPrice","Market Price"]}
    out=pd.DataFrame()
    for k,v in cmap.items():
        c=next((x for x in v if x in df.columns),None)
        out[k]=df[c] if c else np.nan
    out["Date"]=pd.to_datetime(out["Date"],errors="coerce")
    for c in ["RemainingMaturity","MarketYield","MarketPrice"]: out[c]=pd.to_numeric(out[c],errors="coerce")
    out["InstrumentText"]=out["InstrumentText"].astype(str)
    return out

def gsom_add_helpers(df,typ):
    df=df.copy()
    df["Type"]=typ
    df["MaturityYears"]=np.where(df["Type"].eq("Bill"),df["RemainingMaturity"]/365,df["RemainingMaturity"])
    df=df[df["MarketYield"].notna()&df["MaturityYears"].gt(0)]
    return df

@st.cache_data(ttl=60*30)
def get_secondary():
    tbill=load_csv(URL_TBILL); tbond=load_csv(URL_TBOND)
    if tbill.empty and tbond.empty: return pd.DataFrame()
    tbill=gsom_add_helpers(gsom_coerce_cols(tbill),"Bill") if not tbill.empty else pd.DataFrame()
    tbond=gsom_add_helpers(gsom_coerce_cols(tbond),"Bond") if not tbond.empty else pd.DataFrame()
    return pd.concat([tbill,tbond],ignore_index=True)

# ───────────────────────────
# PRIMARY
# ───────────────────────────
@st.cache_data(ttl=60*30)
def get_primary():
    df=load_csv(URL_PRIMARY)
    if df.empty: return pd.DataFrame()
    c_issue=next((c for c in ["Issue Date","IssueDate","Date"] if c in df.columns),None)
    c_instr=next((c for c in ["Instrument","Security","Securities"] if c in df.columns),None)
    c_yield=next((c for c in ["Cut-off Yield (%)","Cutoff Yield (%)","Cut Off Yield (%)"] if c in df.columns),None)
    if not all([c_issue,c_instr,c_yield]): return pd.DataFrame()
    out=pd.DataFrame({
        "IssueDate":pd.to_datetime(df[c_issue],errors="coerce"),
        "Instrument":df[c_instr].astype(str).str.strip(),
        "CutoffYield":pd.to_numeric(df[c_yield],errors="coerce")
    })
    out["TenorYears"]=out["Instrument"].apply(parse_tenor_years)
    out["Month"]=out["IssueDate"].dt.to_period("M").astype(str)
    return out[out["TenorYears"].notna() & out["CutoffYield"].notna()]

# ───────────────────────────
# PLOTS
# ───────────────────────────
def plot_secondary(df,title):
    fig=go.Figure()
    for typ,d in df.groupby("Type"):
        d=d.sort_values("MaturityYears")
        fig.add_trace(go.Scatter(x=d["MaturityYears"],y=d["MarketYield"],
                                 mode="lines+markers",name=typ,line=dict(shape="spline")))
    fig.update_yaxes(range=[0,16],title="Yield (%)")
    fig.update_xaxes(title="Remaining Maturity (years)")
    fig.update_layout(height=520,title=title)
    return fig

def plot_primary(df,title):
    fig=go.Figure()
    for label,d in df.groupby("Label"):
        d=d.sort_values("TenorYears")
        fig.add_trace(go.Scatter(x=d["TenorYears"],y=d["CutoffYield"],
                                 mode="lines+markers",name=label,line=dict(shape="spline"),
                                 text=d["Instrument"],
                                 hovertemplate="Instrument: %{text}<br>Tenor: %{x:.2f} yrs<br>Yield: %{y:.2f}%<extra></extra>"))
    fig.update_yaxes(range=[0,16],title="Cut-off Yield (%)")
    fig.update_xaxes(title="Tenor (years)")
    fig.update_layout(height=520,title=title)
    return fig

# ───────────────────────────
# APP UI
# ───────────────────────────
st.set_page_config(page_title="Bangladesh Yield Curves – GSOM & Primary",layout="wide")
st.title("Bangladesh Govt Securities – Yield Curves")

view=st.sidebar.radio("View:",[
    "Secondary: Single date",
    "Secondary: Compare two dates",
    "Primary auction: Compare two months"
],index=0)

sec_df=get_secondary(); pri_df=get_primary()

# ─────────────── Secondary ───────────────
if view.startswith("Secondary"):
    if sec_df.empty: st.error("Secondary data empty"); st.stop()
    all_dates=sorted(sec_df["Date"].dropna().unique())

    def nearest(target):
        dist=pd.Series([abs((pd.Timestamp(d)-target).days) for d in all_dates],index=all_dates)
        return dist.idxmin()

    def curve(d): return sec_df[sec_df["Date"]==d]

    if view=="Secondary: Single date":
        d=st.date_input("Pick date",value=MAX_CAL_DATE,min_value=MIN_CAL_DATE,max_value=MAX_CAL_DATE)
        n=nearest(pd.Timestamp(d))
        if n.date()!=d: st.info(f"Adjusted to nearest available: {n.date()}")
        df=curve(n)
        st.plotly_chart(plot_secondary(df,f"Secondary Yield Curve – {n.date()}"),use_container_width=True)
        st.dataframe(df[["Date","Type","InstrumentText","MaturityYears","MarketYield"]])
    else:
        c1,c2=st.columns(2)
        with c1: d1=st.date_input("Date A",value=MAX_CAL_DATE)
        with c2: d2=st.date_input("Date B",value=MAX_CAL_DATE)
        n1,n2=nearest(pd.Timestamp(d1)),nearest(pd.Timestamp(d2))
        A,B=curve(n1).assign(Label=str(n1.date())),curve(n2).assign(Label=str(n2.date()))
        df=pd.concat([A,B])
        st.plotly_chart(plot_secondary(df,f"Secondary Comparison – {n1.date()} vs {n2.date()}"),use_container_width=True)

# ─────────────── Primary ───────────────
else:
    if pri_df.empty:
        st.error("Primary auction data empty"); st.stop()

    months=sorted(pri_df["Month"].unique())
    if not months:
        st.warning("No valid months found."); st.stop()

    # Default = latest month and previous month (if exists)
    latest_month=months[-1]
    prev_month=months[-2] if len(months)>1 else months[-1]

    c1,c2=st.columns(2)
    with c1:
        m1=st.selectbox("Select Month A",months,index=months.index(prev_month) if prev_month in months else 0)
    with c2:
        m2=st.selectbox("Select Month B",months,index=months.index(latest_month))

    df_list=[]
    for label,m in [(m1,"Month A"),(m2,"Month B")]:
        sub=pri_df[pri_df["Month"]==m].copy()
        if sub.empty: continue
        sub=sub.sort_values("IssueDate")
        idx=sub.groupby("TenorYears")["IssueDate"].idxmax()
        sub=sub.loc[idx].sort_values("TenorYears")
        sub["Label"]=m
        df_list.append(sub)

    if not df_list:
        st.warning("No data found for selected months.")
    else:
        df=pd.concat(df_list)
        st.plotly_chart(plot_primary(df,f"Primary Auction Comparison — {m1} vs {m2}"),use_container_width=True)
        st.dataframe(
            df[["IssueDate","Instrument","TenorYears","CutoffYield","Label"]]
              .rename(columns={"CutoffYield":"Cut-off Yield (%)","TenorYears":"Tenor (years)","Label":"Month"})
              .sort_values(["Month","Tenor (years)"])
        )

st.markdown("---")
st.caption("Secondary: by trading dates. Primary: compares two months (latest per tenor). Y-axis fixed 0–16%.")
