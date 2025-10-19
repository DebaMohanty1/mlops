import os
import json
import pandas as pd
import argparse
import webbrowser
from datetime import datetime
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ------------------------------------------------------
# Paths
REF_PATH = "data/processed/train.csv"
LIVE_DIR = "data/live"
REPORTS_DIR = "reports"
LOG_FILE = "logs.txt"
DRIFT_HISTORY = "drift_history.json"
DRIFT_THRESHOLD = 0.4

# ------------------------------------------------------
ORDERED_BATCHES = ["batch_1", "batch_2", "batch_3"]


# ------------------------------------------------------
def log_event(msg: str):
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def load_env():
    load_dotenv()
    creds = {
        "host": os.getenv("MYSQL_HOST"),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DB"),
    }
    if all(creds.values()):
        print("🔐 MySQL credentials loaded.")
        return creds
    else:
        print("⚠️ SQL credentials not found, defaulting to CSV mode.")
        return None


def get_sql_engine(creds):
    return create_engine(
        f"mysql+pymysql://{creds['user']}:{creds['password']}@{creds['host']}/{creds['database']}"
    )


def load_data(table_or_file, source="file", creds=None):
    """Loads dataset either from SQL or CSV"""
    if source == "sql" and creds:
        try:
            engine = get_sql_engine(creds)
            df = pd.read_sql_table(table_or_file, engine)
            print(f"✅ Loaded SQL table: {table_or_file} ({df.shape})")
            return df
        except Exception as e:
            print(f"❌ Failed to load SQL table {table_or_file}: {e}")

    # Fallback to CSV
    if table_or_file == "train_data":
        path = REF_PATH
    else:
        path = os.path.join(LIVE_DIR, f"{table_or_file}.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"⚠️ Missing file: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded CSV file: {path} ({df.shape})")
    return df


def prepare_dataset(df: pd.DataFrame):
    """
    Prepare dataset for Evidently drift analysis:
    - Drop non-feature columns like Date, Machine_ID, Downtime
    - Ensure all numeric columns are valid
    """
    drop_cols = ["Date", "Machine_ID", "Downtime"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Convert any categorical/object columns to numeric where possible
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace infinite values with NaN
    df = df.replace([float("inf"), float("-inf")], pd.NA)

    # Only keep numeric columns for drift detection
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    schema = DataDefinition(numerical_columns=numeric_cols, categorical_columns=[])

    return Dataset.from_pandas(df, data_definition=schema)



def detect_drift(ref_df, cur_df, tag):
    ref_data, cur_data = prepare_dataset(ref_df), prepare_dataset(cur_df)
    report = Report([DataDriftPreset()])
    results = report.run(cur_data, ref_data)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    html_path = f"{REPORTS_DIR}/drift_report_{tag}.html"
    results.save_html(html_path)

    result_dict = json.loads(results.json())
    drift_share = 0.0
    drifted_columns = 0
    total_columns = len(ref_df.columns)

    for m in result_dict["metrics"]:
        if m["metric_id"].startswith("DriftedColumnsCount"):
            drift_share = m["value"]["share"]
            drifted_columns = m["value"]["count"]
            break

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": tag,
        "drift_share": round(float(drift_share), 3),
        "drifted_columns": int(drifted_columns),
        "total_columns": total_columns,
    }

    json_path = f"{REPORTS_DIR}/drift_{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary


def update_history(summary):
    history = []
    if os.path.exists(DRIFT_HISTORY):
        with open(DRIFT_HISTORY, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append(summary)
    with open(DRIFT_HISTORY, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


# ------------------------------------------------------
def main(source="file"):
    creds = load_env() if source == "sql" else None
    ref_df = load_data("train_data", source, creds)

    for batch in ORDERED_BATCHES:
        print(f"\n🚀 Running drift check for {batch} ...")
        cur_df = load_data(batch, source, creds)

        summary = detect_drift(ref_df, cur_df, batch)
        update_history(summary)

        drift = summary["drift_share"]

        if drift < 0.1:
            level = "🟢 Low (Stable)"
            log_event(f"{level}: {drift:.2f} — no action needed.")
        elif drift < 0.3:
            level = "🟡 Moderate (Monitor)"
            log_event(f"{level}: {drift:.2f} → monitor for trend.")
        else:
            level = "🔴 High (Retrain)"
            log_event(f"{level}: {drift:.2f} → triggering retraining...")

            # ✅ Directly call training script to avoid recursive DVC locking
            os.system(f"{os.getenv('PYTHON', 'python')} src/train.py")
            log_event(f"✅ Retraining complete for {batch}. Model updated.")

            # ✅ Promote current dataset as baseline
            cur_df.to_csv(REF_PATH, index=False)
            log_event(f"🆕 Baseline updated → {REF_PATH}")
            break


        # open report in browser
        webbrowser.open(f"file:///{os.path.abspath(REPORTS_DIR)}/drift_report_{batch}.html")

    print("\n✅ All drift checks complete.")

    # ------------------------------------------------------
    # Persist drift logs for DVC outs
    os.makedirs(REPORTS_DIR, exist_ok=True)
    drift_summary_path = os.path.join(REPORTS_DIR, "drift_log.json")
    history_path = os.path.join(REPORTS_DIR, "drift_history.json")

    # Write summaries so DVC outs always exist
    with open(drift_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    if os.path.exists(DRIFT_HISTORY):
        with open(DRIFT_HISTORY, "r", encoding="utf-8") as src:
            hist = json.load(src)
        with open(history_path, "w", encoding="utf-8") as dst:
            json.dump(hist, dst, indent=4)

# ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drift detection")
    parser.add_argument("--source", choices=["file", "sql"], default="file", help="Data source mode")
    args = parser.parse_args()
    main(args.source)
