from __future__ import annotations

import datetime as dt
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_generator import generate_dataset
from train_model import FEATURE_COLUMNS, TARGET_COLUMN, train_and_select_model
from utils import classify_risk, suggestions, water_components

st.set_page_config(page_title="Water Footprint Estimator", page_icon="W", layout="wide")

st.markdown(
    """
    <style>
      :root {
        --bg: #020817;
        --sidebar: #111827;
        --panel: #0b1224;
        --text: #f8fafc;
        --muted: #9ca3af;
        --accent: #ef4444;
      }
      .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      }
      .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1300px;
      }
      .hero {
        background: transparent;
        color: var(--text);
        border-radius: 10px;
        padding: 4px 0 8px 0;
        margin-bottom: 14px;
      }
      .hero h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: 0.2px;
      }
      .hero p {
        margin: 8px 0 0;
        color: var(--muted);
        font-size: 1.15rem;
      }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f2433 0%, #1a1f2f 100%);
      }
      [data-testid="stSidebar"] * {
        color: #f8fafc !important;
      }
      [data-testid="stMetricLabel"] {
        color: #f1f5f9 !important;
      }
      [data-testid="stMetricValue"] {
        color: #ffffff !important;
      }
      [data-testid="stAlert"] {
        border-radius: 10px;
      }
      .section-card {
        background: linear-gradient(180deg, #0b1224 0%, #0a1020 100%);
        border: 1px solid #1f2a44;
        border-radius: 14px;
        padding: 14px 14px 10px 14px;
        box-shadow: 0 8px 24px rgba(2, 6, 23, 0.35);
        margin-bottom: 12px;
      }
      .section-title {
        font-size: 0.78rem;
        color: #93c5fd;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 8px;
      }
      div[data-testid="stVerticalBlock"] div.stButton > button {
        background: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
      }
      div[data-testid="stVerticalBlock"] div.stButton > button:hover {
        background: #dc2626 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Water Footprint Estimator</h1>
      <p>Estimate daily household water footprint and identify reduction opportunities.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/water_data.csv"


@st.cache_resource
def ensure_bundle(path: str):
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(path):
        if not os.path.exists(DATA_PATH):
            df = generate_dataset()
            df.to_csv(DATA_PATH, index=False)

        train_df = pd.read_csv(DATA_PATH)
        model, metrics = train_and_select_model(train_df)
        bundle = {
            "model": model,
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "metrics": {
                "model_name": metrics.model_name,
                "mae": metrics.mae,
                "r2": metrics.r2,
            },
        }
        joblib.dump(bundle, path)

    return joblib.load(path)


bundle = ensure_bundle(MODEL_PATH)
model = bundle["model"]
metrics = bundle.get("metrics", {})

with st.sidebar:
    st.header("Input Parameters")
    daily_water = st.slider("Daily Water Usage (liters)", 50, 800, 250)
    rice = st.slider("Rice Consumption (kg/day)", 0.0, 2.5, 0.4, 0.05)
    meat = st.slider("Meat Consumption (kg/day)", 0.0, 1.5, 0.2, 0.05)
    electricity = st.slider("Electricity Usage (kWh/day)", 0, 30, 8)
    household_size = st.slider("Household Size", 1, 10, 4)
    run_pred = st.button("Estimate Water Footprint", type="primary", use_container_width=True)

left, right = st.columns([1.2, 1])

if run_pred:
    input_df = pd.DataFrame(
        [
            {
                "daily_water_usage": daily_water,
                "rice_consumption_kg": rice,
                "meat_consumption_kg": meat,
                "electricity_usage_kwh": electricity,
                "household_size": household_size,
            }
        ]
    )

    prediction = float(model.predict(input_df)[0])
    risk = classify_risk(prediction)
    parts = water_components(daily_water, rice, meat, electricity, household_size)

    with left:
        st.markdown("<div class='section-card'><div class='section-title'>Overview</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated Footprint", f"{prediction:,.0f} L/day")
        m2.metric("Risk Level", risk)
        m3.metric("Household Size", household_size)

        if risk == "High":
            st.error("High water footprint detected. Immediate reduction actions are recommended.")
        elif risk == "Moderate":
            st.warning("Moderate water footprint. There is room for optimization.")
        else:
            st.success("Low water footprint. Keep sustaining efficient habits.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'><div class='section-title'>Analysis</div>", unsafe_allow_html=True)
        st.subheader("Water Footprint Breakdown")
        comp_df = pd.DataFrame({"Component": list(parts.keys()), "Liters": list(parts.values())})
        st.bar_chart(comp_df.set_index("Component"))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'><div class='section-title'>Guidance</div>", unsafe_allow_html=True)
        st.subheader("Actionable Suggestions")
        for tip in suggestions(risk):
            st.write(f"- {tip}")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-card'><div class='section-title'>Composition</div>", unsafe_allow_html=True)
        st.subheader("Component Share")
        fig, ax = plt.subplots()
        ax.pie(parts.values(), labels=parts.keys(), autopct="%1.1f%%", startangle=140)
        ax.axis("equal")
        st.pyplot(fig, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'><div class='section-title'>Forecast</div>", unsafe_allow_html=True)
        st.subheader("30-Day Trend Simulation")
        today = dt.date.today()
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 180, size=30)
        trend = np.clip(prediction + noise, a_min=0, a_max=None)
        trend_df = pd.DataFrame(
            {
                "date": pd.date_range(today - dt.timedelta(days=29), periods=30),
                "estimated_liters": trend,
            }
        )
        st.line_chart(trend_df.set_index("date"))
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Set the input values in the sidebar and click 'Estimate Water Footprint'.")

st.divider()
col_a, col_b = st.columns(2)
with col_a:
    st.write("**Model Details**")
    st.write(f"- Selected Model: `{metrics.get('model_name', 'Unknown')}`")
with col_b:
    st.write("**Evaluation Metrics**")
    st.write(
        f"- MAE: `{metrics.get('mae', 'N/A'):.2f}`"
        if isinstance(metrics.get("mae"), (int, float))
        else "- MAE: `N/A`"
    )
    st.write(
        f"- R2: `{metrics.get('r2', 'N/A'):.4f}`"
        if isinstance(metrics.get("r2"), (int, float))
        else "- R2: `N/A`"
    )
