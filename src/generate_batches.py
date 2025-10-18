# ============================================================
# 🧠 Machine Downtime Dataset Generator (with Realistic Drift)
# ============================================================

import os
import pandas as pd
import numpy as np
import sqlalchemy
from dotenv import load_dotenv

# ----------------------------
# Configuration
# ----------------------------
RAW_PATH = "data/raw/Machine_downtime.csv"
FULL_PATH = "data/raw/Machine_downtime_full.csv"
PROCESSED_PATH = "data/processed/train.csv"
LIVE_DIR = "data/live"
np.random.seed(42)

EXPANSION_FACTOR = 10
N_BATCHES = 3  # low / mid / high

# ----------------------------
# Drift configuration
# ----------------------------
DRIFT_LEVELS = ["low", "mid", "high"]  # label-based
DRIFT_MAP = {
    "low": {
        "Hydraulic_Pressure(bar)": (1.05, 0.01, "mul"),
        "Coolant_Temperature(°C)": (0.5, 0.2, "add"),
        "Tool_Vibration(µm)": (1.03, 0.02, "mul"),
    },
    "mid": {
        "Hydraulic_Pressure(bar)": (1.12, 0.03, "mul"),
        "Coolant_Temperature(°C)": (2.0, 0.5, "add"),
        "Spindle_Speed(RPM)": (0.97, 0.03, "mul"),
        "Tool_Vibration(µm)": (1.10, 0.03, "mul"),
        "Torque": (1.06, 0.02, "mul"),
    },
    "high": {
        "Hydraulic_Pressure(bar)": (1.35, 0.08, "mul"),
        "Air_System_Pressure(bar)": (0.7, 0.4, "add"),
        "Coolant_Temperature(°C)": (5.0, 0.8, "add"),
        "Tool_Vibration(µm)": (1.25, 0.07, "mul"),
        "Cutting_Force(kN)": (0.88, 0.05, "mul"),
        "Torque": (1.15, 0.04, "mul"),
        "Spindle_Speed(RPM)": (0.92, 0.05, "mul"),
        "Load_cells": (1.10, 0.05, "mul"),
        "Hydraulic_Oil_Temperature(°C)": (4.0, 0.7, "add"),
        "Proximity_sensors": (1.12, 0.05, "mul"),
    },
}


# ----------------------------
# Helper Functions
# ----------------------------
def load_env():
    """Load MySQL credentials if available"""
    load_dotenv()
    creds = {
        "host": os.getenv("MYSQL_HOST"),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DB"),
    }
    if all(creds.values()):
        print("🔐 Loaded MySQL credentials from .env")
        return creds
    else:
        print("⚠️ MySQL credentials not found. Skipping SQL upload.")
        return None


def save_to_mysql(df, table_name, creds):
    """Save dataframe to MySQL"""
    try:
        import pymysql  # lazy import
        engine = sqlalchemy.create_engine(
            f"mysql+pymysql://{creds['user']}:{creds['password']}@{creds['host']}/{creds['database']}"
        )
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"🗄️ Saved {table_name} → MySQL successfully.")
    except Exception as e:
        print(f"❌ Failed to save {table_name} to MySQL: {e}")


def load_data():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"❌ Raw file not found: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"✅ Loaded base dataset: {df.shape}")
    return df


def expand_data(df, factor=10):
    """Expand dataset with Gaussian noise for realism"""
    dfs = []
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for i in range(factor):
        df_copy = df.copy()
        df_copy[numeric_cols] = df_copy[numeric_cols].apply(
            lambda x: x + np.random.normal(0, 0.02 * x.std(), size=len(x))
        )
        dfs.append(df_copy)
    df_expanded = pd.concat(dfs, ignore_index=True)
    print(f"📈 Expanded dataset to: {df_expanded.shape}")
    return df_expanded


def apply_drift(df, drift_label):
    """Apply realistic drift based on drift level"""
    df_drifted = df.copy()
    config = DRIFT_MAP[drift_label]

    for col, (mean_shift, std_shift, mode) in config.items():
        if col not in df_drifted.columns:
            continue
        if mode == "mul":
            df_drifted[col] *= np.random.normal(mean_shift, std_shift, len(df_drifted))
        elif mode == "add":
            df_drifted[col] += np.random.normal(mean_shift, std_shift, len(df_drifted))
    return df_drifted


def save_batches(df, creds=None):
    """Split data into baseline + drifted live batches"""
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(LIVE_DIR, exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    # Save full expanded dataset
    df.to_csv(FULL_PATH, index=False)
    print(f"💾 Full dataset saved → {FULL_PATH}")

    total_rows = len(df)
    batch_size = total_rows // (N_BATCHES + 1)
    print(f"📦 Using {batch_size} rows per batch")

    # Baseline
    train_df = df.iloc[:batch_size].reset_index(drop=True)
    train_df.to_csv(PROCESSED_PATH, index=False)
    print(f"💾 Baseline saved → {PROCESSED_PATH}")
    if creds:
        save_to_mysql(train_df, "train_data", creds)

    # Drifted batches
    for i, label in enumerate(DRIFT_LEVELS, start=1):
        start = i * batch_size
        end = (i + 1) * batch_size if i < N_BATCHES else total_rows
        batch_df = df.iloc[start:end].reset_index(drop=True)

        drifted = apply_drift(batch_df, label)
        path = os.path.join(LIVE_DIR, f"batch_{i}.csv")
        drifted.to_csv(path, index=False)
        print(f"💾 Saved {path} (drift type = {label})")

        if creds:
            save_to_mysql(drifted, f"batch_{i}", creds)


def summarize_batches():
    print("\n📊 Drift Summary (expected separation):")
    print(" - batch_1.csv → Low drift ≈ 0–10%")
    print(" - batch_2.csv → Moderate drift ≈ 25–35%")
    print(" - batch_3.csv → High drift ≈ 60–70%+ (retraining likely)")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    creds = load_env()
    df = load_data()
    df_expanded = expand_data(df, EXPANSION_FACTOR)
    save_batches(df_expanded, creds)
    summarize_batches()
    print("\n✅ All synthetic batches + full dataset generated successfully!")
