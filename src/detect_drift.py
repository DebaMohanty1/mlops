# ======================================================
# 🧠 Machine Failure Drift Detection + DVC Integration
# ======================================================

import os
import json
import pandas as pd
import webbrowser
from datetime import datetime
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

# ---------------------------------------------------------------------
# Paths
REF_PATH = "data/processed/train.csv"
LIVE_DIR = "data/live"
REPORTS_DIR = "reports"
LOG_FILE = "logs.txt"
DRIFT_HISTORY = "drift_history.json"
DRIFT_THRESHOLD = 0.5
# ---------------------------------------------------------------------

def log_event(msg: str):
    """Append timestamped message to logs.txt"""
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)

def prepare_dataset(df: pd.DataFrame):
    """Clean and convert dataframe to Evidently Dataset"""
    drop_cols = ["Date", "Downtime", "Machine_ID"]
    df = df.drop(columns=drop_cols, errors="ignore")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    schema = DataDefinition(numerical_columns=numeric_cols, categorical_columns=[])
    return Dataset.from_pandas(df, data_definition=schema)

def detect_drift(ref_df, cur_df, tag):
    """Run Evidently drift detection and return summary"""
    ref_data, cur_data = prepare_dataset(ref_df), prepare_dataset(cur_df)
    report = Report([DataDriftPreset()])
    results = report.run(cur_data, ref_data)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    html_path = f"{REPORTS_DIR}/drift_report_{tag}.html"
    results.save_html(html_path)
    print(f"✅ Drift report saved to {html_path}")

    # Extract drift summary from results
    result_dict = json.loads(results.json())
    drifted_columns = 0
    total_columns = 0
    drift_share = 0

    for m in result_dict["metrics"]:
        if m["metric_id"].startswith("DriftedColumnsCount"):
            drifted_columns = m["value"]["count"]
            drift_share = m["value"]["share"]
            total_columns = len(ref_df.columns)
            break

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": tag,
        "drift_share": round(float(drift_share), 3),
        "drifted_columns": int(drifted_columns),
        "total_columns": total_columns
    }

    with open(f"reports/drift_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary

def update_history(summary):
    """Append summary to drift_history.json"""
    history = []
    if os.path.exists(DRIFT_HISTORY):
        with open(DRIFT_HISTORY, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append(summary)
    with open(DRIFT_HISTORY, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

def main():
    ref_df = pd.read_csv(REF_PATH)
    live_files = sorted([f for f in os.listdir(LIVE_DIR) if f.startswith("current_")])

    for file in live_files:
        tag = file.replace("current_", "").replace(".csv", "")
        cur_df = pd.read_csv(os.path.join(LIVE_DIR, file))
        print(f"\n🚀 Running drift check for {file} ...")

        summary = detect_drift(ref_df, cur_df, tag)
        update_history(summary)

        drift = summary["drift_share"]
        drifted = summary["drifted_columns"]
        total = summary["total_columns"]

        if drift > DRIFT_THRESHOLD:
            log_event(f"⚠️ Drift detected for {tag} ({drift:.2f}) → Triggering DVC retraining...")
            os.system("dvc repro train")
            log_event(f"✅ Retraining complete for {tag}. Model updated.")
        else:
            log_event(f"✅ No significant drift for {tag} ({drift:.2f}).")

        webbrowser.open(f"reports/drift_report_{tag}.html")

    print("\n✅ All drift checks complete.")

if __name__ == "__main__":
    main()
