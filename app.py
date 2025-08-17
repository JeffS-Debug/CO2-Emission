import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Streamlit page + Altair settings
st.set_page_config(page_title="Canada CO₂ Dashboard", layout="wide")
alt.data_transformers.disable_max_rows()  # avoid row-limit warnings

# ---------- RAW, COMMIT-PINNED URLs (your repo) ----------
CO2_URL = "https://raw.githubusercontent.com/JeffS-Debug/CO2-Emission/1f8f2ea65de1f821f7b0f68cb581641e710d4e43/data/processed/CO2%20Emission.csv"
TEMP_URL = "https://raw.githubusercontent.com/JeffS-Debug/CO2-Emission/1f8f2ea65de1f821f7b0f68cb581641e710d4e43/data/processed/Temp%20Canada.csv"
DIS_URL  = "https://raw.githubusercontent.com/JeffS-Debug/CO2-Emission/1f8f2ea65de1f821f7b0f68cb581641e710d4e43/data/processed/Disasters%20Canada.csv"

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def co2_long_from_url(url: str) -> pd.DataFrame:
    """Load CO₂ per-capita panel (wide or long) -> tidy long [country, year, co2_t_per_cap]."""
    df = pd.read_csv(url)
    # wide if there are year columns
    year_cols = [c for c in df.columns if str(c).isdigit()]
    if year_cols:
        cname = next((c for c in df.columns if "Country Name" in c or c.lower()=="country"), df.columns[0])
        long = df.melt(id_vars=[cname], value_vars=year_cols,
                       var_name="year", value_name="co2_t_per_cap") \
                 .rename(columns={cname:"country"})
    else:
        long = df.rename(columns=str.title).copy()
        if "Year" not in long and "Time" in long:
            long["Year"] = pd.to_datetime(long["Time"], errors="coerce").dt.year
        vcol = next(c for c in long.columns
                    if c.lower() in ["co2_t_per_cap","co2 per capita","co2 emission","value"])
        long = long.rename(columns={"Country":"country", vcol:"co2_t_per_cap", "Year":"year"})
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long["co2_t_per_cap"] = pd.to_numeric(long["co2_t_per_cap"], errors="coerce")
    long = long.dropna(subset=["year","co2_t_per_cap"]).copy()
    long["year"] = long["year"].astype(int)
    return long

@st.cache_data(show_spinner=False)
def load_all():
    co2_panel = co2_long_from_url(CO2_URL)

    # If World missing, add a simple (unweighted) mean for context (not population-weighted)
    if "World" not in co2_panel["country"].unique():
        # pandas GroupBy.mean on this runtime doesn't take skipna, so just call .mean()
        world = (co2_panel.groupby("year", as_index=False)["co2_t_per_cap"].mean())
        world["country"] = "World"
        co2_panel = pd.concat([co2_panel, world[["country","year","co2_t_per_cap"]]], ignore_index=True)

    # Canada series
    co2_can = (co2_panel[co2_panel["country"].str.lower()=="canada"]
               [["year","co2_t_per_cap"]].rename(columns={"co2_t_per_cap":"co2_t_per_cap_can"}))

    temp = pd.read_csv(TEMP_URL)
    temp.columns = [c.strip().lower() for c in temp.columns]
    temp = temp.rename(columns={"tas":"temp_c","temperature":"temp_c","temp":"temp_c"})[["year","temp_c"]]
    temp["year"] = pd.to_numeric(temp["year"], errors="coerce").astype("Int64")
    temp["temp_c"] = pd.to_numeric(temp["temp_c"], errors="coerce")

    dis = pd.read_csv(DIS_URL)
    dis.columns = [c.strip().lower() for c in dis.columns]
    dis = dis.rename(columns={"natural disasters":"disasters","disaster_count":"disasters"})[["year","disasters"]]
    dis["year"] = pd.to_numeric(dis["year"], errors="coerce").astype("Int64")
    dis["disasters"] = pd.to_numeric(dis["disasters"], errors="coerce")

    # Merge Canada tables
    can = (co2_can.merge(temp, on="year", how="outer")
                  .merge(dis,  on="year", how="outer")).sort_values("year")

    # Overlap slice (complete cases)
    can_core = (can.dropna(subset=["co2_t_per_cap_can","temp_c","disasters"])
                  .astype({"year":int})
                  .sort_values("year"))

    # Rolling + index (base = first overlap year)
    base_year = int(can_core["year"].min()) if len(can_core) else int(can["year"].dropna().min())
    can_w = can.copy()
    for col in ["co2_t_per_cap_can","temp_c","disasters"]:
        can_w[f"{col}_roll5"] = can_w[col].rolling(5, min_periods=3).mean()
        base = can_w.loc[can_w["year"]==base_year, col].squeeze()
        can_w[f"{col}_idx"] = (can_w[col] / base * 100) if pd.notna(base) and base!=0 else np.nan

    return co2_panel, can, can_core, can_w

def top10_latest(co2_panel: pd.DataFrame, force_include="Canada") -> list[str]:
    latest = int(co2_panel["year"].max())
    t10 = (co2_panel.query("year == @latest")
           .dropna(subset=["co2_t_per_cap"])
           .nlargest(10, "co2_t_per_cap")["country"].tolist())
    if force_include and force_include not in t10:
        t10.append(force_include)
    return t10

# ---------- Load data ----------
co2_panel, can, can_core, can_w = load_all()

min_year, max_year = int(co2_panel["year"].min()), int(co2_panel["year"].max())

# ---------- Sidebar ----------
st.sidebar.header("Controls")
year_range = st.sidebar.slider("Year range", min_year, max_year,
                               (max(1900, min_year), max_year), step=1)
smooth = st.sidebar.selectbox("Smoothing (some lines)", [1,3,5], index=2)
show_world = st.sidebar.checkbox("Show World overlay", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Data: commit-pinned CSVs. CO₂ is per-capita, production-based.")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Overview", "CO₂ – Global Views", "Canada – Relationships"])

# ===== Overview =====
with tab1:
    st.subheader("Key stats (overlap years)")
    if len(can_core)==0:
        st.warning("Not enough overlap across CO₂, temperature, and disasters.")
    else:
        sub = can_core[(can_core["year"]>=year_range[0]) & (can_core["year"]<=year_range[1])]
        k1, s1 = sub["co2_t_per_cap_can"].mean(), sub["co2_t_per_cap_can"].std()
        k2, s2 = sub["temp_c"].mean(), sub["temp_c"].std()
        c1, c2, c3 = st.columns(3)
        c1.metric("CO₂ per capita (mean)", f"{k1:.2f} t/person", f"SD {s1:.2f}")
        c2.metric("Temperature (mean)",     f"{k2:.2f} °C",     f"SD {s2:.2f}")
        c3.metric("Overlap years",          f"{len(sub)}")

    st.markdown("#### Indexed trends (base = first overlap year = 100)")
    base_year = int(can_core["year"].min()) if len(can_core) else None
    s = can_w[(can_w["year"]>=year_range[0]) & (can_w["year"]<=year_range[1])].copy()
    if base_year is not None:
        long_idx = s.melt(id_vars="year",
                          value_vars=["co2_t_per_cap_can_idx","temp_c_idx","disasters_idx"],
                          var_name="metric", value_name="index")
        long_idx["metric"] = long_idx["metric"].map({
            "co2_t_per_cap_can_idx":"CO₂ (index)",
            "temp_c_idx":"Temp (index)",
            "disasters_idx":"Disasters (index)"
        })
        ch = (alt.Chart(long_idx).mark_line()
              .encode(x="year:Q",
                      y=alt.Y("index:Q", title=f"Index (base = {base_year})"),
                      color=alt.Color("metric:N", legend=alt.Legend(title="Series"))))
        st.altair_chart(ch, use_container_width=True)

# ===== CO₂ – Global Views =====
with tab2:
    g = co2_panel[(co2_panel["year"]>=year_range[0]) & (co2_panel["year"]<=year_range[1])].copy()

    st.markdown("#### 1) Adding color: all countries grey; highlight Canada (and World)")
    grey = (alt.Chart(g.query("country!='Canada'"))
              .mark_line(opacity=0.15, color="lightgray")
              .encode(x="year:Q", y=alt.Y("co2_t_per_cap:Q", title="t CO₂ per person"),
                      detail="country:N"))
    can_line = (alt.Chart(g[g["country"]=="Canada"]).mark_line(size=2.5)
                .encode(x="year:Q", y="co2_t_per_cap:Q", color=alt.value("#1f77b4")))
    layers = grey + can_line
    if show_world and "World" in g["country"].unique():
        world_line = (alt.Chart(g[g["country"]=="World"]).mark_line(size=2.5, strokeDash=[4,3], color="#d62728")
                      .encode(x="year:Q", y="co2_t_per_cap:Q"))
        layers = layers + world_line
    st.altair_chart(layers.properties(height=380), use_container_width=True)

    st.markdown("#### 2) Top-10 per-capita emitters (latest year) with labels at line ends")
    t10 = top10_latest(co2_panel)
    sub = g[g["country"].isin(t10)].copy()
    last = (sub.sort_values("year").dropna(subset=["co2_t_per_cap"])
            .groupby("country").tail(1).assign(year=lambda d: d["year"] + 0.6))
    ch_lines = (alt.Chart(sub).mark_line()
                .encode(x="year:Q", y="co2_t_per_cap:Q", color=alt.Color("country:N", legend=None)))
    ch_labels = (alt.Chart(last).mark_text(align="left", baseline="middle", dx=5, size=10)
                 .encode(x="year:Q", y="co2_t_per_cap:Q", text="country:N", color=alt.Color("country:N", legend=None)))
    st.altair_chart((ch_lines + ch_labels).properties(height=420), use_container_width=True)

    st.markdown("#### 3) Tile heatmap of the same top-10")
    tile = sub.pivot(index="country", columns="year", values="co2_t_per_cap").sort_index(axis=1)
    tile = tile.loc[[c for c in t10 if c in tile.index]]
    tile_long = (tile.reset_index().melt(id_vars="country", var_name="year", value_name="co2_t_per_cap").dropna())
    heat = (alt.Chart(tile_long).mark_rect()
            .encode(x=alt.X("year:O", title="Year"),
                    y=alt.Y("country:N", sort=t10),
                    color=alt.Color("co2_t_per_cap:Q", title="t CO₂ per person",
                                    scale=alt.Scale(scheme="viridis"))))
    st.altair_chart(heat.properties(height=28*len(tile.index), width=900), use_container_width=True)

    st.markdown("#### 4) Facets (3×2): World + selected countries (incl. Canada)")
    candidates = ["World","Canada","United States","China","Russian Federation","Saudi Arabia","Australia","Qatar"]
    names = [c for c in candidates if c in g["country"].unique()][:6]
    facet = (alt.Chart(g[g["country"].isin(names)]).mark_line()
             .encode(x="year:Q", y=alt.Y("co2_t_per_cap:Q", title="t CO₂ per person"),
                     facet=alt.Facet("country:N", columns=2)))
    st.altair_chart(facet.properties(width=350, height=220), use_container_width=True)

# ===== Canada – Relationships =====
with tab3:
    st.markdown("#### Scatter: CO₂ per capita vs Temperature (overlap years)")
    core = can_core[(can_core["year"]>=year_range[0]) & (can_core["year"]<=year_range[1])]
    if len(core)==0:
        st.info("No overlap in selected range.")
    else:
        x, y = core["co2_t_per_cap_can"].values, core["temp_c"].values
        m, b = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0,1]
        reg = pd.DataFrame({"x":[x.min(), x.max()], "y":[m*x.min()+b, m*x.max()+b]})
        ch_sc = (alt.Chart(core).mark_point(opacity=0.7)
                 .encode(x=alt.X("co2_t_per_cap_can:Q", title="t CO₂ per person"),
                         y=alt.Y("temp_c:Q", title="°C (annual mean)")))
        ch_reg = alt.Chart(reg).mark_line().encode(x="x:Q", y="y:Q")
        st.altair_chart((ch_sc + ch_reg).properties(height=360), use_container_width=True)
        st.caption(f"Pearson r = {r:.3f}   |   OLS: temp = {m:.3f}·CO₂ + {b:.3f}")

    st.markdown("#### Bubble-scatter: CO₂ per capita vs Natural disasters")
    core2 = core.dropna(subset=["disasters"])
    if len(core2):
        d = core2["disasters"].astype(float)
        if d.max() > d.min():
            size = 50 + 350*(d - d.min())/(d.max() - d.min())
        else:
            size = pd.Series(150.0, index=core2.index)
        core2 = core2.assign(size=size)
        x2, y2 = core2["co2_t_per_cap_can"].values, core2["disasters"].values
        m2, b2 = np.polyfit(x2, y2, 1)
        reg2 = pd.DataFrame({"x":[x2.min(), x2.max()], "y":[m2*x2.min()+b2, m2*x2.max()+b2]})
        ch_b = (alt.Chart(core2).mark_point(opacity=0.6, stroke="black", strokeWidth=0.3)
                .encode(x=alt.X("co2_t_per_cap_can:Q", title="t CO₂ per person"),
                        y=alt.Y("disasters:Q", title="Events per year"),
                        size=alt.Size("size:Q", legend=None)))
        ch_r2 = alt.Chart(reg2).mark_line().encode(x="x:Q", y="y:Q")
        st.altair_chart((ch_b + ch_r2).properties(height=360), use_container_width=True)
        st.caption("Bubble size encodes disaster counts (EM-DAT). Trend line is descriptive.")
