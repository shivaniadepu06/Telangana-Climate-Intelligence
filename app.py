import os
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Telangana Climate Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Selectable visual themes
THEMES = {
    "Cloud Blue": {
        "navy": "#10243A", "blue": "#58B8E8", "cyan": "#75D6D0",
        "gold": "#F6C66A", "coral": "#F28B82", "purple": "#A99BFF",
        "green": "#9BD6B5", "bg1": "#234765", "bg2": "#0b1b2c",
        "sidebar1": "#102a42", "sidebar2": "#0b1b2c", "hero1": "#27597d", "hero2": "#1a3956"
    },
    "Sunrise Sand": {
        "navy": "#382B2A", "blue": "#D98C5F", "cyan": "#E7BFA3",
        "gold": "#F2C66D", "coral": "#C96B58", "purple": "#A68A9D",
        "green": "#A8B89A", "bg1": "#6E5145", "bg2": "#2B2428",
        "sidebar1": "#493633", "sidebar2": "#2B2428", "hero1": "#9A6650", "hero2": "#55404A"
    },
    "Arctic Mint": {
        "navy": "#142E35", "blue": "#53B7C5", "cyan": "#8ADBC8",
        "gold": "#E9C46A", "coral": "#E58C78", "purple": "#9C9FE8",
        "green": "#91D3A7", "bg1": "#24545A", "bg2": "#102A32",
        "sidebar1": "#173D43", "sidebar2": "#102A32", "hero1": "#286A70", "hero2": "#214A55"
    }
}
selected_theme = st.sidebar.selectbox("🎨 Dashboard colour theme", list(THEMES), index=0)
THEME = THEMES[selected_theme]
NAVY = THEME["navy"]
BLUE = THEME["blue"]
CYAN = THEME["cyan"]
GOLD = THEME["gold"]
CORAL = THEME["coral"]
PURPLE = THEME["purple"]
GREEN = THEME["green"]
PALETTE = [BLUE, CYAN, GOLD, CORAL, PURPLE, GREEN, "#91A7FF"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#eaf4ff; --muted:#a9bfd2; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(circle at 10% 0%,__THEME_BG1__ 0%,__THEME_BG2__ 100%); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(0,0,0,0); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,__THEME_SIDEBAR1__,__THEME_SIDEBAR2__); border-right:1px solid rgba(155,205,235,.13); }
[data-testid="stSidebar"] * { color:#eaf4ff; }
h1,h2,h3 { font-family:'Manrope',sans-serif; color:#f2f8ff; letter-spacing:-.4px; }
.block-container { padding-top:1.2rem; padding-bottom:3rem; max-width:1550px; }
.hero { position:relative; overflow:hidden; padding:30px 34px; border:1px solid rgba(193,226,248,.18); border-radius:26px; background:linear-gradient(115deg,__THEME_HERO1__f0,__THEME_HERO2__e8 65%,#3f4e6fee); box-shadow:0 18px 50px rgba(0,0,0,.20); margin-bottom:22px; }
.hero:before,.hero:after { content:''; position:absolute; border-radius:100px; background:rgba(230,246,255,.10); filter:blur(1px); }
.hero:before { width:230px;height:78px;right:8%;top:24px;box-shadow:35px -22px 0 -7px rgba(230,246,255,.10),75px 5px 0 -12px rgba(230,246,255,.10); animation:drift 16s ease-in-out infinite alternate; }
.hero:after { width:150px;height:52px;right:30%;bottom:18px;box-shadow:25px -16px 0 -6px rgba(230,246,255,.08); animation:drift 21s ease-in-out infinite alternate-reverse; }
@keyframes drift { from{transform:translateX(-15px)} to{transform:translateX(35px)} }
.hero-kicker { color:#a9e6f6; font-size:12px; font-weight:800; letter-spacing:2.4px; text-transform:uppercase; }
.hero-title { color:#fff; font-family:'Manrope',sans-serif; font-size:clamp(27px,4vw,43px); font-weight:800; line-height:1.12; margin:8px 0; }
.hero-sub { color:#d2e4f2; font-size:15px; max-width:760px; }
.hero-cloud { position:absolute; right:5%; bottom:0; font-size:82px; opacity:.23; animation:floatcloud 7s ease-in-out infinite alternate; }
@keyframes floatcloud { from{transform:translateY(2px)} to{transform:translateY(-13px)} }
.section-label { color:#8ed7ec; font-size:11px; font-weight:800; letter-spacing:2px; text-transform:uppercase; margin:14px 0 5px; }
.section-heading { display:block; clear:both; position:relative; z-index:1; line-height:1.35; padding-top:7px; padding-bottom:5px; font-family:'Manrope',sans-serif; font-size:23px; font-weight:800; color:#f2f8ff; margin:0 0 14px; }
.metric-card { background:linear-gradient(145deg,rgba(36,69,96,.92),rgba(20,43,66,.92)); border:1px solid rgba(176,218,244,.16); border-radius:18px; padding:17px 19px; min-height:112px; box-shadow:0 10px 25px rgba(0,0,0,.12); }
.metric-label { color:#aec7d9; font-size:12px; font-weight:700; letter-spacing:.3px; }
.metric-value { color:#fff; font-family:'Manrope',sans-serif; font-size:27px; font-weight:800; margin-top:8px; }
.metric-note { color:#8fd8d0; font-size:11px; margin-top:3px; }
.weather-scene { position:relative; overflow:hidden; min-height:250px; border-radius:22px; padding:25px; color:#fff; border:1px solid rgba(255,255,255,.18); }
.weather-scene.rain { background:linear-gradient(125deg,#203c59,#536e86); }
.weather-scene.sun { background:linear-gradient(125deg,#d18c3c,#edb85d 55%,#e88961); }
.weather-scene.cloud { background:linear-gradient(125deg,#526b83,#8ca4b3); }
.weather-scene.wind { background:linear-gradient(125deg,#176b88,#64b8ba); }
.weather-scene.storm { background:linear-gradient(125deg,#171f3d,#45516e); }
.weather-scene .weather-emoji { font-size:55px; position:relative; z-index:2; }
.weather-scene .scene-title { font-size:24px; font-weight:800; position:relative; z-index:2; }
.weather-scene .scene-detail { color:rgba(255,255,255,.9); position:relative; z-index:2; }
.scene-cloud { position:absolute; font-size:75px; opacity:.28; animation:drift 12s ease-in-out infinite alternate; }
.scene-cloud.c1 {right:12%;top:5px}.scene-cloud.c2 {right:38%;bottom:-22px;font-size:62px;animation-duration:17s}
.scene-rain {position:absolute;inset:-30px;pointer-events:none;background-image:repeating-linear-gradient(105deg,transparent 0 22px,rgba(215,240,255,.48) 23px 25px,transparent 26px 48px);background-size:90px 180px;animation:rainmove .65s linear infinite;opacity:.48}
@keyframes rainmove {to{background-position:0 180px}}
.scene-sun {position:absolute;right:12%;top:16px;font-size:92px;filter:drop-shadow(0 0 22px rgba(255,230,150,.7));animation:sunpulse 3s ease-in-out infinite alternate}
@keyframes sunpulse {to{transform:scale(1.12);filter:drop-shadow(0 0 34px rgba(255,230,150,.9))}}
.scene-wind {position:absolute;inset:0;opacity:.4;background:repeating-linear-gradient(0deg,transparent 0 25px,rgba(230,255,255,.6) 26px 28px,transparent 29px 52px);animation:windmove 3s linear infinite}
@keyframes windmove {to{background-position:160px 0}}
.scene-lightning {position:absolute;right:17%;top:72px;font-size:75px;animation:flash 1.7s infinite;}
@keyframes flash {0%,76%,100%{opacity:.15}80%,85%{opacity:1}}
.stTabs [data-baseweb="tab-list"] { gap:8px; background:rgba(14,34,53,.55); padding:7px; border-radius:15px; }
.stTabs [data-baseweb="tab"] { color:#c8dbea; border-radius:10px; padding:9px 16px; }
.stTabs [aria-selected="true"] { background:#2c6385 !important; color:#fff !important; }
div[data-testid="stPlotlyChart"] { margin-top:10px; background:rgba(18,42,64,.60); border:1px solid rgba(176,218,244,.12); border-radius:18px; padding:8px; }
[data-testid="stMetric"] { background:rgba(24,51,75,.7); border:1px solid rgba(176,218,244,.13); padding:12px 15px; border-radius:15px; }
hr { border-color:rgba(176,218,244,.16); }
footer {visibility:hidden;}
</style>
""".replace("__THEME_BG1__", THEME["bg1"]).replace("__THEME_BG2__", THEME["bg2"]).replace("__THEME_SIDEBAR1__", THEME["sidebar1"]).replace("__THEME_SIDEBAR2__", THEME["sidebar2"]).replace("__THEME_HERO1__", THEME["hero1"]).replace("__THEME_HERO2__", THEME["hero2"]), unsafe_allow_html=True)

# ============================================================
# DATA STANDARDISATION
# ============================================================
REQUIRED = [
    "district", "mandal", "date", "rainfall", "temp_min", "temp_max",
    "humidity_min", "humidity_max", "wind_speed_min", "wind_speed_max"
]
NUMERIC = ["rainfall", "temp_min", "temp_max", "humidity_min", "humidity_max", "wind_speed_min", "wind_speed_max"]

# Canonical district labels. Generic 'Warangal' is intentionally kept separate
# until its Urban/Rural origin can be verified from source records.
DISTRICT_MAP = {
    "Hanumakkonda": "Hanumakonda",
    "Hanumakonda": "Hanumakonda",
    "Jagitial": "Jagtial",
    "Jagtial": "Jagtial",
    "Bhadradri-Kothagudem": "Bhadradri Kothagudem",
    "Bhadradri Kothagudem": "Bhadradri Kothagudem",
    "Jayashankar": "Jayashankar Bhupalpally",
    "Jayashankar-Bhupalpally": "Jayashankar Bhupalpally",
    "Jayashankar Bhupalpally": "Jayashankar Bhupalpally",
    "Jogulamba-Gadwal": "Jogulamba Gadwal",
    "Jogulamba Gadwal": "Jogulamba Gadwal",
    "Komaram Bheem-Asifabad": "Komaram Bheem Asifabad",
    "Kumuram Bheem": "Komaram Bheem Asifabad",
    "Kumuram Bheem - Asifabad": "Komaram Bheem Asifabad",
    "Kumuram Bheem-Asifabad": "Komaram Bheem Asifabad",
    "Medchal-Malkajgiri": "Medchal Malkajgiri",
    "Mulug": "Mulugu",
    "Mulugu": "Mulugu",
    "Rajanna-Siricilla": "Rajanna Sircilla",
    "Rajanna Sircilla": "Rajanna Sircilla",
    "Yadadri-Bhongir": "Yadadri Bhuvanagiri",
    "Yadadri Bhongir": "Yadadri Bhuvanagiri",
    "Yadadri Bhuvanagiri": "Yadadri Bhuvanagiri",
    "Warangal (R)": "Warangal Rural",
    "Warangal Rural": "Warangal Rural",
    "Warangal (U)": "Warangal Urban",
    "Warangal Urban": "Warangal Urban",
    "Warangal": "Warangal",
}


def standardize_district(value):
    if pd.isna(value):
        return "Unknown"
    clean = " ".join(str(value).strip().split())
    return DISTRICT_MAP.get(clean, clean)


@st.cache_data(show_spinner="Loading, cleaning and standardising the climate dataset...")
def prepare_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(file_bytes))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = df[REQUIRED].copy()
    df["district"] = df["district"].apply(standardize_district)
    df["mandal"] = df["mandal"].astype("string").str.strip()
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates().reset_index(drop=True)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col in ["rainfall", "humidity_min", "humidity_max", "wind_speed_min", "wind_speed_max"]:
            df.loc[df[col] < 0, col] = np.nan
        else:
            df.loc[df[col] == -1, col] = np.nan
        medians = df.groupby("district")[col].transform("median")
        df[col] = df[col].fillna(medians).fillna(df[col].median())
    df["mandal"] = df["mandal"].fillna("Unknown")
    df = df.sort_values(["date", "district", "mandal"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["season"] = np.select(
        [df["month"].isin([12, 1, 2]), df["month"].isin([3, 4, 5]), df["month"].isin([6, 7, 8, 9])],
        ["Winter", "Summer", "Monsoon"], default="Post-Monsoon"
    )
    return df

# ============================================================
# SIDEBAR: DATA SOURCE + FILTERS
# ============================================================
st.sidebar.markdown("## ☁️ Dashboard Controls")
uploaded = st.sidebar.file_uploader("Upload Telangana climate CSV", type=["csv"])
possible_files = [
    Path("telangana climate (1).csv"), Path("telangana climate.csv"),
    Path("telangana_climate.csv"), Path("telangana_dashboard_data.csv")
]
local_csv = next((p for p in possible_files if p.exists()), None)
if uploaded is not None:
    raw_bytes = uploaded.getvalue()
elif local_csv is not None:
    raw_bytes = local_csv.read_bytes()
else:
    st.markdown("<div class='hero'><div class='hero-kicker'>DATA SOURCE REQUIRED</div><div class='hero-title'>Upload your Telangana climate dataset</div><div class='hero-sub'>Upload the cleaned CSV with district, mandal, date, rainfall, temperature, humidity and wind-speed columns.</div><div class='hero-cloud'>☁️</div></div>", unsafe_allow_html=True)
    st.stop()

try:
    df = prepare_data(raw_bytes)
except Exception as exc:
    st.error(f"Dataset could not be prepared: {exc}")
    st.stop()
if df.empty:
    st.error("No valid records were found in the uploaded dataset.")
    st.stop()

all_districts = sorted(df["district"].dropna().unique().tolist())
selected_district = st.sidebar.selectbox("District", ["All districts"] + all_districts)
years = sorted(df["year"].unique().tolist())
selected_years = st.sidebar.multiselect("Years", years, default=years)
if not selected_years:
    selected_years = years
filtered = df[df["year"].isin(selected_years)].copy()
if selected_district != "All districts":
    filtered = filtered[filtered["district"] == selected_district]
if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# ============================================================
# HELPERS
# ============================================================
def style_fig(fig, height=340):
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#DCEBFA", size=12),
        margin=dict(l=24, r=24, t=105, b=65),
        title=dict(font=dict(size=16, color="#F2F8FF")),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5, title_text=""),
        xaxis=dict(gridcolor="rgba(190,220,240,.10)"),
        yaxis=dict(gridcolor="rgba(190,220,240,.10)"),
    )
    return fig


def section(title, kicker=None):
    if kicker:
        st.markdown(f"<div class='section-label'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-heading'>{title}</div>", unsafe_allow_html=True)


def metric_card(label, value, note=""):
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )

# ============================================================
# MACHINE LEARNING: 3 MODELS, TIME-ORDERED 80/20 SPLIT
# Predicts next day's statewide average maximum temperature.
# ============================================================
@st.cache_resource(show_spinner="Training Linear Regression, Random Forest and Gradient Boosting...")
def train_models(data: pd.DataFrame):
    daily = data.groupby("date", as_index=False)[NUMERIC].mean().sort_values("date").reset_index(drop=True)
    daily["temp_max_lag1"] = daily["temp_max"].shift(1)
    daily["temp_max_lag2"] = daily["temp_max"].shift(2)
    daily["temp_max_lag3"] = daily["temp_max"].shift(3)
    daily["rainfall_lag1"] = daily["rainfall"].shift(1)
    daily["humidity_max_lag1"] = daily["humidity_max"].shift(1)
    daily["wind_speed_max_lag1"] = daily["wind_speed_max"].shift(1)
    daily["month_sin"] = np.sin(2 * np.pi * daily["date"].dt.month / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["date"].dt.month / 12)
    features = [
        "temp_max_lag1", "temp_max_lag2", "temp_max_lag3", "rainfall_lag1",
        "humidity_max_lag1", "wind_speed_max_lag1", "month_sin", "month_cos"
    ]
    model_data = daily.dropna(subset=features + ["temp_max"]).reset_index(drop=True)
    if len(model_data) < 40:
        raise ValueError("At least 40 daily observations are required for the chronological model evaluation.")
    split_at = int(len(model_data) * 0.80)
    X = model_data[features]
    y = model_data["temp_max"]
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=160, max_depth=12, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=160, learning_rate=0.04, max_depth=2,
            loss="huber", random_state=42
        ),
    }

    metric_rows = []
    predictions_by_model = {}
    fitted_models = {}
    for name, estimator in models.items():
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        err = y_test.to_numpy() - pred
        metric_rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred),
        })
        predictions_by_model[name] = pred
        fitted_models[name] = estimator

    metrics = pd.DataFrame(metric_rows).sort_values(["RMSE", "MAE"], ascending=True).reset_index(drop=True)
    # Winner is chosen by lowest held-out chronological RMSE; MAE breaks ties.
    best_name = metrics.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_predictions = predictions_by_model[best_name]
    test_results = pd.DataFrame({
        "date": model_data["date"].iloc[split_at:].to_numpy(),
        "Actual": y_test.to_numpy(),
        "Predicted": best_predictions,
        "Error": y_test.to_numpy() - best_predictions,
    })
    all_test_predictions = pd.DataFrame({"date": model_data["date"].iloc[split_at:].to_numpy(), "Actual": y_test.to_numpy()})
    for name, pred in predictions_by_model.items():
        all_test_predictions[name] = pred

    split_data = pd.DataFrame({"Dataset": ["Training", "Testing"], "Records": [len(X_train), len(X_test)]})
    timeline = model_data[["date", "temp_max"]].copy()
    timeline["Split"] = np.where(np.arange(len(timeline)) < split_at, "Training", "Testing")

    # Model-specific importance: tree models use impurity importance;
    # Linear Regression is represented using absolute standardized coefficient magnitude.
    importance_frames = []
    for name, estimator in fitted_models.items():
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
        else:
            linear_step = estimator[-1] if hasattr(estimator, "steps") else estimator
            values = np.abs(np.asarray(linear_step.coef_))
            values = values / values.sum() if values.sum() else values
        importance_frames.append(pd.DataFrame({"Feature": features, "Importance": values, "Model": name}))
    importance = pd.concat(importance_frames, ignore_index=True)

    # Build next-day feature row from the latest available historical observations.
    latest = daily.sort_values("date").reset_index(drop=True)
    next_date = latest["date"].iloc[-1] + pd.Timedelta(days=1)
    next_month = next_date.month
    next_input = pd.DataFrame([{
        "temp_max_lag1": latest["temp_max"].iloc[-1],
        "temp_max_lag2": latest["temp_max"].iloc[-2],
        "temp_max_lag3": latest["temp_max"].iloc[-3],
        "rainfall_lag1": latest["rainfall"].iloc[-1],
        "humidity_max_lag1": latest["humidity_max"].iloc[-1],
        "wind_speed_max_lag1": latest["wind_speed_max"].iloc[-1],
        "month_sin": np.sin(2 * np.pi * next_month / 12),
        "month_cos": np.cos(2 * np.pi * next_month / 12),
    }], columns=features)
    next_predictions = {name: float(est.predict(next_input)[0]) for name, est in fitted_models.items()}
    return {
        "daily": daily, "model_data": model_data, "models": fitted_models,
        "metrics": metrics, "test_results": test_results,
        "all_test_predictions": all_test_predictions, "split_data": split_data,
        "timeline": timeline, "importance": importance,
        "best_name": best_name, "next_predictions": next_predictions,
        "next_date": next_date,
    }

try:
    ml = train_models(df)
except Exception as exc:
    ml = None
    ml_error = str(exc)

# ============================================================
# HERO BANNER
# ============================================================
period = f"{min(selected_years)}–{max(selected_years)}" if len(selected_years) > 1 else str(selected_years[0])
where = selected_district if selected_district != "All districts" else "Across Telangana"
st.markdown(f"""
<div class='hero'>
  <div class='hero-kicker'>TELANGANA • CLIMATE INTELLIGENCE • {period}</div>
  <div class='hero-title'>Climate & Weather<br>Trend Analytics</div>
  <div class='hero-sub'>Explore historical climate patterns, district-level differences, seasonal rainfall and three machine-learning models for temperature prediction.</div>
  <div style='margin-top:15px;color:#b9e8f4;font-weight:700'>📍 {where} &nbsp; • &nbsp; {len(filtered):,} records in current view</div>
  <div class='hero-cloud'>☁️</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# DASHBOARD TABS
# ============================================================
t_overview, t_climate, t_district, t_ml, t_weather, t_quality = st.tabs([
    "☀️ Overview", "🌦️ Climate Insights", "🗺️ District Explorer",
    "🤖 ML Studio", "🌤️ Live Weather", "🧪 Data Quality"
])

# ---------------- OVERVIEW ----------------
with t_overview:
    section("Climate at a glance", "01 / OVERVIEW")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: metric_card("AVG. MAX TEMPERATURE", f"{filtered['temp_max'].mean():.1f}°C", "Historical average")
    with k2: metric_card("HIGHEST TEMPERATURE", f"{filtered['temp_max'].max():.1f}°C", "Maximum in selected view")
    with k3: metric_card("AVG. RAINFALL", f"{filtered['rainfall'].mean():.2f} mm", "Mean per observation")
    with k4: metric_card("AVG. MAX HUMIDITY", f"{filtered['humidity_max'].mean():.1f}%", "Moisture level")
    with k5: metric_card("DISTRICTS COVERED", f"{filtered['district'].nunique():,}", "Standardised labels")

    left, right = st.columns([1.55, 1])
    with left:
        section("Temperature through the years", "TREND WATCH")
        monthly = filtered.groupby(["year", "month"], as_index=False).agg(temp_max=("temp_max", "mean"))
        monthly["month_label"] = monthly["month"].map({i: datetime(2000, i, 1).strftime("%b") for i in range(1, 13)})
        fig = px.line(monthly, x="month_label", y="temp_max", color="year", markers=True, color_discrete_sequence=PALETTE, title="Monthly average maximum temperature")
        fig.update_traces(line=dict(width=3), marker=dict(size=7))
        st.plotly_chart(style_fig(fig, 370), use_container_width=True)
    with right:
        section("Seasonal rainfall share", "SEASONAL SNAPSHOT")
        season_rain = filtered.groupby("season", as_index=False)["rainfall"].sum()
        fig = px.pie(season_rain, names="season", values="rainfall", hole=.62, color="season", color_discrete_map={"Winter":BLUE,"Summer":GOLD,"Monsoon":CYAN,"Post-Monsoon":PURPLE}, title="Rainfall contribution by season")
        fig.update_traces(textposition="inside", textinfo="percent", insidetextorientation="radial", marker=dict(line=dict(color="#122b43", width=3)))
        st.plotly_chart(style_fig(fig, 370), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        section("Temperature range by season", "DISTRIBUTION")
        fig = px.box(filtered, x="season", y="temp_max", color="season", color_discrete_sequence=PALETTE, points=False, title="Maximum temperature distribution")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with c2:
        section("Weather variable relationship", "CORRELATION")
        corr_cols = NUMERIC
        corr = filtered[corr_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Weather-variable correlation heatmap")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

# ---------------- CLIMATE INSIGHTS ----------------
with t_climate:
    section("Climate patterns & extremes", "02 / CLIMATE INSIGHTS")
    a, b = st.columns(2)
    season_summary = filtered.groupby("season", as_index=False).agg(avg_temp=("temp_max", "mean"), avg_rain=("rainfall", "mean"), total_rain=("rainfall", "sum"))
    with a:
        fig = px.bar(season_summary, x="season", y="avg_temp", color="season", text_auto=".1f", color_discrete_sequence=PALETTE, title="Average maximum temperature by season")
        st.plotly_chart(style_fig(fig, 350), use_container_width=True)
    with b:
        fig = px.bar(season_summary, x="season", y="avg_rain", color="season", text_auto=".2f", color_discrete_sequence=PALETTE, title="Average rainfall by season")
        st.plotly_chart(style_fig(fig, 350), use_container_width=True)

    a, b = st.columns(2)
    with a:
        month_rain = filtered.groupby("month", as_index=False).agg(rainfall=("rainfall", "mean"))
        month_rain["Month"] = month_rain["month"].map({i: datetime(2000, i, 1).strftime("%B") for i in range(1, 13)})
        fig = px.area(month_rain, x="Month", y="rainfall", markers=True, title="Average rainfall by month", color_discrete_sequence=[CYAN])
        fig.update_traces(line=dict(width=3), fillcolor="rgba(117,214,208,.20)")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with b:
        sample = filtered.sample(min(5000, len(filtered)), random_state=42)
        fig = px.scatter(sample, x="humidity_max", y="rainfall", color="season", size="temp_max", opacity=.65, color_discrete_sequence=PALETTE, title="Humidity, rainfall & temperature relationship")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    district_summary = filtered.groupby("district", as_index=False).agg(avg_temp=("temp_max", "mean"), avg_rain=("rainfall", "mean"))
    a, b = st.columns(2)
    with a:
        hottest = district_summary.nlargest(10, "avg_temp").sort_values("avg_temp")
        fig = px.bar(hottest, x="avg_temp", y="district", orientation="h", text_auto=".2f", color="avg_temp", color_continuous_scale="OrRd", title="Top 10 hottest districts")
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)
    with b:
        wettest = district_summary.nlargest(10, "avg_rain").sort_values("avg_rain")
        fig = px.bar(wettest, x="avg_rain", y="district", orientation="h", text_auto=".2f", color="avg_rain", color_continuous_scale="Blues", title="Top 10 rainfall districts")
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)

# ---------------- DISTRICT EXPLORER ----------------
with t_district:
    section("Explore district-level weather", "03 / DISTRICT EXPLORER")
    district_for_explorer = st.selectbox("Choose a district to inspect", all_districts, key="district_explorer")
    ddf = df[(df["district"] == district_for_explorer) & (df["year"].isin(selected_years))].copy()
    if ddf.empty:
        ddf = df[df["district"] == district_for_explorer].copy()
    d1, d2, d3, d4 = st.columns(4)
    with d1: metric_card("AVG. MAX TEMP", f"{ddf['temp_max'].mean():.1f}°C", district_for_explorer)
    with d2: metric_card("AVG. RAINFALL", f"{ddf['rainfall'].mean():.2f} mm", "Per observation")
    with d3: metric_card("AVG. HUMIDITY", f"{ddf['humidity_max'].mean():.1f}%", "Maximum humidity")
    with d4: metric_card("AVG. WIND SPEED", f"{ddf['wind_speed_max'].mean():.2f} m/s", "Maximum wind speed")
    district_daily = ddf.groupby("date", as_index=False).agg(temp_max=("temp_max", "mean"), rainfall=("rainfall", "mean"), humidity=("humidity_max", "mean"), wind=("wind_speed_max", "mean"))
    x, y = st.columns(2)
    with x:
        fig = px.line(district_daily, x="date", y="temp_max", title=f"{district_for_explorer}: temperature timeline", color_discrete_sequence=[CORAL])
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with y:
        fig = px.area(district_daily, x="date", y="rainfall", title=f"{district_for_explorer}: rainfall timeline", color_discrete_sequence=[BLUE])
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    x, y = st.columns(2)
    with x:
        fig = px.line(district_daily, x="date", y="humidity", title="Humidity trend", color_discrete_sequence=[CYAN])
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    with y:
        fig = px.line(district_daily, x="date", y="wind", title="Wind-speed trend", color_discrete_sequence=[PURPLE])
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

# ---------------- MACHINE LEARNING STUDIO ----------------
with t_ml:
    section("Three-model training, testing & prediction laboratory", "04 / MACHINE LEARNING STUDIO")
    if ml is None:
        st.error(f"Model training could not be completed: {ml_error}")
    else:
        metrics = ml["metrics"]
        best_name = ml["best_name"]
        best_row = metrics.iloc[0]
        st.success(f"Best model on the chronological test set: {best_name} (lowest RMSE)")
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_card("TRAINING RECORDS", f"{ml['split_data'].iloc[0]['Records']:,}", "Earlier 80% of dates")
        with m2: metric_card("TESTING RECORDS", f"{ml['split_data'].iloc[1]['Records']:,}", "Later 20% of dates")
        with m3: metric_card("BEST MODEL TEST RMSE", f"{best_row['RMSE']:.3f}°C", best_name)
        with m4: metric_card("BEST MODEL TEST R²", f"{best_row['R²']:.3f}", "Held-out test period")

        st.markdown("### Model performance comparison")
        st.dataframe(metrics.style.format({"MAE":"{:.3f}", "RMSE":"{:.3f}", "R²":"{:.3f}"}), use_container_width=True, hide_index=True)
        a, b = st.columns(2)
        with a:
            fig = px.bar(metrics.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Error"), x="Model", y="Error", color="Metric", barmode="group", text_auto=".3f", color_discrete_map={"MAE":CYAN,"RMSE":CORAL}, title="Test error comparison (lower is better)")
            st.plotly_chart(style_fig(fig, 370), use_container_width=True)
        with b:
            fig = px.bar(metrics, x="Model", y="R²", color="Model", color_discrete_sequence=PALETTE, text_auto=".3f", title="Test R² comparison")
            st.plotly_chart(style_fig(fig, 370), use_container_width=True)

        st.markdown("### Training and testing dataset")
        a, b = st.columns(2)
        with a:
            fig = px.pie(ml["split_data"], names="Dataset", values="Records", hole=.62, color="Dataset", color_discrete_map={"Training":BLUE,"Testing":GOLD}, title="Chronological 80:20 split")
            fig.update_traces(textposition="inside", textinfo="percent+value", insidetextorientation="radial")
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
        with b:
            fig = px.scatter(ml["timeline"], x="date", y="temp_max", color="Split", color_discrete_map={"Training":BLUE,"Testing":GOLD}, title="Training period vs testing period")
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)

        st.markdown("### Actual vs predicted — all three models")
        selected_model_view = st.selectbox("Choose model for detailed prediction graph", metrics["Model"].tolist(), index=metrics["Model"].tolist().index(best_name))
        test_view = ml["all_test_predictions"].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_view["date"], y=test_view["Actual"], mode="lines", name="Actual", line=dict(color=CYAN, width=2.5)))
        fig.add_trace(go.Scatter(x=test_view["date"], y=test_view[selected_model_view], mode="lines", name=f"Predicted — {selected_model_view}", line=dict(color=GOLD, width=2, dash="dot")))
        fig.update_layout(title=f"Actual vs predicted maximum temperature — {selected_model_view}", xaxis_title="Date", yaxis_title="Temperature (°C)")
        st.plotly_chart(style_fig(fig, 390), use_container_width=True)

        a, b = st.columns(2)
        with a:
            pred_scatter = test_view[["Actual", selected_model_view]].rename(columns={selected_model_view:"Predicted"})
            fig = px.scatter(pred_scatter, x="Actual", y="Predicted", opacity=.65, color_discrete_sequence=[PURPLE], title=f"Actual vs predicted scatter — {selected_model_view}")
            low = float(min(pred_scatter.min().min(), pred_scatter.max().min()))
            high = float(max(pred_scatter.max().max(), pred_scatter.min().max()))
            fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(color=GOLD, dash="dash"))
            st.plotly_chart(style_fig(fig, 350), use_container_width=True)
        with b:
            residuals = test_view["Actual"] - test_view[selected_model_view]
            residual_df = pd.DataFrame({"Prediction error (°C)": residuals})
            fig = px.histogram(residual_df, x="Prediction error (°C)", nbins=35, title=f"Prediction error distribution — {selected_model_view}", color_discrete_sequence=[CORAL])
            st.plotly_chart(style_fig(fig, 350), use_container_width=True)

        st.markdown("### Feature importance / model coefficients")
        importance_model = st.selectbox("Select model for feature contribution", metrics["Model"].tolist(), key="importance_model")
        imp = ml["importance"].query("Model == @importance_model").sort_values("Importance")
        title_text = "Absolute standardised coefficient magnitude" if importance_model == "Linear Regression" else "Tree-based feature importance"
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Teal", title=f"{title_text} — {importance_model}")
        st.plotly_chart(style_fig(fig, 350), use_container_width=True)

        st.markdown("### Next-day maximum temperature estimate")
        next_value = ml["next_predictions"][best_name]
        st.markdown(f"""
        <div class='weather-scene cloud'>
          <div class='scene-cloud c1'>☁️</div><div class='scene-cloud c2'>☁️</div>
          <div class='weather-emoji'>🔮</div>
          <div class='scene-title'>Selected model: {best_name}</div>
          <div style='font-size:40px;font-weight:800;position:relative;z-index:2'>{next_value:.2f} °C</div>
          <div class='scene-detail'>Forecast date: {ml['next_date']:%d %B %Y} • Selected using lowest held-out test RMSE</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Evaluation uses a chronological 80/20 split. Earlier dates train the models; later dates are held out for testing. No random shuffle is used. The next-day estimate is a model-based estimate, not a live weather forecast.")

# ---------------- LIVE WEATHER + ANIMATED WEATHER EFFECTS ----------------
with t_weather:
    section("Live Weather", "05 / LIVE WEATHER")
    st.write("Choose a Telangana district or city to request current weather from OpenWeather. Configure your API key as an environment variable; never paste the key into this source file.")
    weather_places = sorted(set(all_districts + ["Warangal", "Hyderabad", "Karimnagar", "Nizamabad", "Khammam"]))
    selected_place = st.selectbox("Select district / city", weather_places, key="weather_place")
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    if not api_key:
        st.info("Live API is not configured. Set OPENWEATHER_API_KEY in PowerShell, then restart Streamlit. All historical dashboard sections remain available.")
    else:
        @st.cache_data(ttl=600, show_spinner=False)
        def fetch_live_weather(place: str, key: str):
            response = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": f"{place},IN", "appid": key, "units": "metric"},
                timeout=15,
            )
            response.raise_for_status()
            return response.json()
        try:
            API_PLACE_MAP = {
                "Warangal Rural": "Warangal", "Warangal Urban": "Warangal",
                "Warangal": "Warangal", "Hanumakonda": "Hanamkonda",
                "Medchal Malkajgiri": "Hyderabad", "Rangareddy": "Hyderabad",
                "Jayashankar Bhupalpally": "Bhupalpally",
                "Komaram Bheem Asifabad": "Asifabad",
                "Bhadradri Kothagudem": "Kothagudem",
                "Jogulamba Gadwal": "Gadwal", "Rajanna Sircilla": "Sircilla",
                "Yadadri Bhuvanagiri": "Bhongir"
            }
            api_place = API_PLACE_MAP.get(selected_place, selected_place)
            live = fetch_live_weather(api_place, api_key)
            condition = str(live["weather"][0]["main"])
            description = str(live["weather"][0]["description"]).title()
            temp = float(live["main"]["temp"])
            humidity = int(live["main"]["humidity"])
            wind = float(live.get("wind", {}).get("speed", 0))
            low_condition = condition.lower()
            if "thunder" in low_condition:
                scene, emoji, css_class, effect = "Thunderstorm", "⛈️", "storm", '<div class="scene-rain"></div><div class="scene-lightning">⚡</div><div class="scene-cloud c1">☁️</div>'
            elif "rain" in low_condition or "drizzle" in low_condition:
                scene, emoji, css_class, effect = "Rainy", "🌧️", "rain", '<div class="scene-rain"></div><div class="scene-cloud c1">☁️</div>'
            elif "clear" in low_condition:
                scene, emoji, css_class, effect = "Sunny", "☀️", "sun", '<div class="scene-sun">☀️</div>'
            elif wind >= 8:
                scene, emoji, css_class, effect = "Windy", "💨", "wind", '<div class="scene-wind"></div><div class="scene-cloud c1">☁️</div>'
            elif "cloud" in low_condition or "mist" in low_condition or "fog" in low_condition or "haze" in low_condition:
                scene, emoji, css_class, effect = "Cloudy", "☁️", "cloud", '<div class="scene-cloud c1">☁️</div><div class="scene-cloud c2">☁️</div>'
            else:
                scene, emoji, css_class, effect = "Current conditions", "🌤️", "cloud", '<div class="scene-cloud c1">☁️</div>'
            st.markdown(f"""
            <div class='weather-scene {css_class}'>{effect}
              <div class='weather-emoji'>{emoji}</div>
              <div class='scene-title'>{selected_place} • {scene}</div>
              <div style='font-size:40px;font-weight:800;position:relative;z-index:2'>{temp:.1f}°C</div>
              <div class='scene-detail'>{description} &nbsp; | &nbsp; Humidity {humidity}% &nbsp; | &nbsp; Wind {wind:.1f} m/s</div>
            </div>
            """, unsafe_allow_html=True)
            w1, w2, w3 = st.columns(3)
            with w1: metric_card("TEMPERATURE", f"{temp:.1f}°C", selected_place)
            with w2: metric_card("HUMIDITY", f"{humidity}%", "Current relative humidity")
            with w3: metric_card("WIND SPEED", f"{wind:.1f} m/s", "Current wind speed")
            st.caption("Live conditions are provided by OpenWeather and cached for 10 minutes. Some district names may not be recognised by the weather provider; try the district headquarters/city name if lookup fails.")
        except requests.HTTPError as exc:
            st.error(f"Weather provider could not find or return data for {selected_place} (lookup: {api_place}). Check the place name and API key. Details: {exc}")
        except requests.RequestException as exc:
            st.error(f"Could not retrieve live weather: {exc}")
        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"The weather response was incomplete: {exc}")

# ---------------- DATA QUALITY / DISTRICT AUDIT ----------------
with t_quality:
    section("Data quality & district-name audit", "06 / DATA QUALITY")
    st.write("District variants are standardised before filters, charts and model training. Generic 'Warangal' records remain labelled Warangal because the source does not identify them as Urban or Rural.")
    q1, q2, q3 = st.columns(3)
    with q1: metric_card("STANDARDISED DISTRICT LABELS", f"{df['district'].nunique():,}", "Uses source district labels; generic Warangal remains separate")
    with q2: metric_card("DUPLICATE ROWS REMOVED", "Applied", "Exact duplicate records removed during cleaning")
    with q3: metric_card("DATE RANGE", f"{df['date'].min():%Y}–{df['date'].max():%Y}", "Historical records")
    st.markdown("### Districts in the cleaned dataset")
    district_counts = df.groupby("district", as_index=False).size().rename(columns={"size":"Records"}).sort_values("district")
    st.dataframe(district_counts, use_container_width=True, hide_index=True)
    st.download_button(
        "Download standardised dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="telangana_climate_standardised.csv",
        mime="text/csv",
    )

st.markdown("<hr><div style='text-align:center;color:#8faec5;font-size:12px'>TELANGANA CLIMATE INTELLIGENCE • Historical trends • District analytics • Three-model machine learning • Live weather</div>", unsafe_allow_html=True)
