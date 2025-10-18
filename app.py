# app.py
import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

from src.utils import (
    load_drift_summary,
    load_drift_history,
    load_metrics,
    load_logs,
)

# -----------------------------------------------------------
# 🎨 Streamlit Page Setup
# -----------------------------------------------------------
st.set_page_config(
    page_title="Machine Failure MLOps Dashboard",
    layout="wide",
    page_icon="⚙️",
)

st.title("⚙️ Machine Failure – MLOps Monitoring Dashboard")
st.caption("Real-time Data Drift Detection, CI/CD Retraining, and Model Metrics")

# Auto-refresh every 10 seconds
st_autorefresh(interval=10000, key="drift_monitor")

# -----------------------------------------------------------
# 📊 Load Data
# -----------------------------------------------------------
drift_summary = load_drift_summary()
drift_history = load_drift_history()
metrics = load_metrics()
logs = load_logs()

drift_share = drift_summary.get("drift_share", 0)
n_drifted = drift_summary.get("drifted_columns", 0)
n_total = drift_summary.get("total_columns", 1)
timestamp = drift_summary.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# -----------------------------------------------------------
# 🧭 Layout: 3 Columns
# -----------------------------------------------------------
col1, col2, col3 = st.columns(3)

# Drift Gauge
with col1:
    st.markdown("### 🌡️ Drift Status")
    if drift_share < 0.3:
        st.success(f"Stable — {drift_share*100:.1f}% drifted")
    elif drift_share < 0.5:
        st.warning(f"Moderate — {drift_share*100:.1f}% drifted")
    else:
        st.error(f"⚠️ High Drift — {drift_share*100:.1f}% drifted")

    st.metric("Drifted Features", f"{n_drifted}/{n_total}")
    st.metric("Last Checked", timestamp)

# Model Metrics
with col2:
    st.markdown("### 📈 Model Performance")
    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2f}")
    st.metric("F1 Score", f"{metrics.get('f1_score', 0):.2f}")
    st.metric("Precision", f"{metrics.get('precision', 0):.2f}")
    st.metric("Recall", f"{metrics.get('recall', 0):.2f}")

# Manual retrain option
with col3:
    st.markdown("### 🔁 Actions")
    if st.button("Trigger Retrain (Manual Demo)"):
        with st.spinner("Running DVC repro..."):
            st.info("Executing retraining pipeline ⏳")
            import os
            os.system("dvc repro train")
            st.success("Retraining complete ✅")
            time.sleep(2)
            st.experimental_rerun()

# -----------------------------------------------------------
# 📈 Drift History Chart
# -----------------------------------------------------------
st.markdown("### 📉 Drift Trend Over Time")

if drift_history:
    df_hist = pd.DataFrame(drift_history)
    df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
    st.line_chart(df_hist.set_index("timestamp")["drift_share"])
else:
    st.info("No drift history yet. Run the pipeline once to generate reports.")

# -----------------------------------------------------------
# 🧾 Logs
# -----------------------------------------------------------
st.markdown("### 🧾 CI/CD Pipeline Logs")
st.text_area("Live Log Output", "\n".join(logs), height=300)

# -----------------------------------------------------------
# 📊 Drifted Feature Summary (Optional)
# -----------------------------------------------------------
if "drifted_columns" in drift_summary:
    st.markdown("### 🔍 Latest Drift Summary (JSON View)")
    st.json(drift_summary)
