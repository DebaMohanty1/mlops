# ============================================================
# 🧠 Machine Downtime Drift Simulation (Low / Mid / High)
# ============================================================

import os
import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
import json

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
RAW_PATH = "data/raw/Machine_downtime.csv"
OUT_DIR = "data/live"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load and prepare data
# ------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
df = df.drop(columns=["Date"], errors="ignore")
reference = df.copy()
print(f"✅ Loaded: {df.shape}")

# ------------------------------------------------------------
# Simulate drift levels
# ------------------------------------------------------------
np.random.seed(42)

# 🟢 Low drift (~0.1)
df_low = df.copy()
df_low["Hydraulic_Pressure(bar)"] *= np.random.normal(1.05, 0.01, len(df_low))
df_low["Coolant_Temperature(°C)"] += np.random.normal(0.5, 0.2, len(df_low))
df_low["Tool_Vibration(µm)"] *= np.random.normal(1.03, 0.02, len(df_low))

# 🟡 Mid drift (~0.2)
df_mid = df.copy()
df_mid["Hydraulic_Pressure(bar)"] *= np.random.normal(1.10, 0.03, len(df_mid))
df_mid["Coolant_Temperature(°C)"] += np.random.normal(1.5, 0.4, len(df_mid))
df_mid["Spindle_Speed(RPM)"] *= np.random.normal(0.98, 0.03, len(df_mid))
df_mid["Tool_Vibration(µm)"] *= np.random.normal(1.08, 0.03, len(df_mid))
df_mid["Torque"] *= np.random.normal(1.05, 0.02, len(df_mid))

# 🔴 High drift (~0.4)
df_high = df.copy()
df_high["Hydraulic_Pressure(bar)"] *= np.random.normal(1.25, 0.05, len(df_high))
df_high["Air_System_Pressure(bar)"] += np.random.normal(0.3, 0.3, len(df_high))
df_high["Coolant_Temperature(°C)"] += np.random.normal(3.0, 0.5, len(df_high))
df_high["Tool_Vibration(µm)"] *= np.random.normal(1.15, 0.05, len(df_high))
df_high["Cutting_Force(kN)"] *= np.random.normal(0.90, 0.03, len(df_high))
df_high["Torque"] *= np.random.normal(1.10, 0.02, len(df_high))
df_high["Spindle_Speed(RPM)"] *= np.random.normal(0.95, 0.03, len(df_high))

# ------------------------------------------------------------
# Helper: run Evidently drift summary
# ------------------------------------------------------------
def drift_summary(df_new, label):
    r = Report([DataDriftPreset()])
    res = r.run(df_new, reference)
    res_dict = json.loads(res.json())
    for m in res_dict["metrics"]:
        if m["metric_id"].startswith("DriftedColumnsCount"):
            count, share = m["value"]["count"], m["value"]["share"]
            if share < 0.1:
                color = "🟢 Stable  "
            elif share < 0.25:
                color = "🟡 Moderate"
            else:
                color = "🔴 High    "
            print(f"{color} | {label:<10} → {count} cols drifted ({share:.2%})")
            return share

# ------------------------------------------------------------
# Run all drift levels
# ------------------------------------------------------------
print("\n📊 Drift Summaries:")
drift_summary(df_low, "Low Drift")
drift_summary(df_mid, "Mid Drift")
drift_summary(df_high, "High Drift")

# ------------------------------------------------------------
# Save CSVs
# ------------------------------------------------------------
df_low.to_csv(os.path.join(OUT_DIR, "current_low.csv"), index=False)
df_mid.to_csv(os.path.join(OUT_DIR, "current_mid.csv"), index=False)
df_high.to_csv(os.path.join(OUT_DIR, "current_high.csv"), index=False)
print("\n💾 Saved all drift datasets → data/live/")
