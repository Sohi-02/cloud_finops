from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="FinOps Forecast Dashboard", layout="wide")

API_BASE_URL = st.sidebar.text_input(
    "FinOps API URL",
    value=os.getenv("FINOPS_API_URL", "http://127.0.0.1:8000"),
)


def safe_request(method: str, endpoint: str, payload: dict | None = None, timeout: int = 10):
    """Request helper with a graceful fallback for local API startup issues."""
    url = f"{API_BASE_URL.rstrip('/')}{endpoint}"
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=timeout)
        else:
            response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


st.title("FinOps Cloud Cost Forecast Dashboard")

with st.sidebar:
    st.header("Controls")
    health = safe_request("GET", "/health")
    model_info = safe_request("GET", "/model-info")
    model_label = model_info.get("alias", "Persistence baseline") if model_info else "Persistence baseline"
    st.metric("Current model", model_label)

    horizon = st.selectbox("Forecast horizon", ["1_hour"])
    current_cost = st.number_input(
        "Current estimated cost index",
        min_value=0.0,
        value=24.4,
        step=0.1,
    )
    st.caption("This dashboard reads from the existing FastAPI forecast service.")

    if health:
        st.success(f"API status: {health.get('status', 'healthy')}")
    else:
        st.warning("API is not reachable yet. Start the FastAPI service on the configured URL.")


prediction_result = None
if st.button("Predict now"):
    prediction_result = safe_request(
        "POST",
        "/predict",
        payload={"estimated_cost_index": float(current_cost)},
    )

if prediction_result:
    predicted_cost = float(prediction_result.get("predicted_next_hour_cost", current_cost))
else:
    predicted_cost = float(current_cost)

col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted next-hour cost", f"{predicted_cost:.4f}")
with col2:
    st.metric("Forecast horizon", horizon)

st.subheader("Cost history")
recent_history = pd.DataFrame(
    {
        "time_bucket": pd.date_range("2026-08-01 00:00:00", periods=24, freq="H"),
        "estimated_cost_index": [
            22.1, 22.4, 22.8, 23.0, 23.6, 24.0, 24.2, 24.8, 25.1, 24.9,
            25.6, 25.7, 26.1, 26.4, 26.8, 26.5, 27.0, 27.1, 27.4, 27.6,
            27.8, 28.1, 28.0, 28.4,
        ],
    }
)
if prediction_result:
    recent_history = pd.concat(
        [
            recent_history,
            pd.DataFrame(
                {
                    "time_bucket": [pd.Timestamp.now().tz_localize("UTC")],
                    "estimated_cost_index": [predicted_cost],
                }
            ),
        ],
        ignore_index=True,
    )

st.line_chart(recent_history.set_index("time_bucket"))

st.subheader("Monitoring and retraining")
monitoring = {
    "drift": safe_request("GET", "/drift"),
    "performance": safe_request("GET", "/performance"),
    "retraining_status": safe_request("GET", "/retraining-status"),
}

st.json(monitoring)

st.subheader("Model information")
if model_info:
    st.json(model_info)
else:
    st.write("Champion: persistence baseline")
    st.write("Forecast contract: next-hour cost equals current-hour estimated cost index")
