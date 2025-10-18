# src/utils.py
import os
import json
from datetime import datetime

LOG_FILE = "logs.txt"


# ---------------------------------------------------------------------
# 🧾 Logging Utilities
# ---------------------------------------------------------------------
def log_event(message: str, show=True):
    """Append timestamped messages to logs.txt and print optionally."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

    if show:
        print(line)


# ---------------------------------------------------------------------
# 📁 JSON Helpers
# ---------------------------------------------------------------------
def read_json(path: str, default=None):
    """Safely read JSON file, return default if not found."""
    if not os.path.exists(path):
        return default or {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default or {}


def write_json(data: dict, path: str):
    """Write dictionary data to JSON file with indentation."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------
# 🧩 Directory and File Utilities
# ---------------------------------------------------------------------
def ensure_dir(path: str):
    """Create directory if it doesn’t exist."""
    os.makedirs(path, exist_ok=True)


def timestamp():
    """Return current timestamp (human-readable)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------
# 📊 Loaders for Streamlit Dashboard
# ---------------------------------------------------------------------
def load_drift_summary():
    """Load latest drift summary from reports/drift.json."""
    return read_json("reports/drift.json", default={"drift_share": 0, "drifted_columns": 0, "total_columns": 0})


def load_drift_history():
    """Load drift history for line chart."""
    return read_json("drift_history.json", default=[])


def load_metrics():
    """Load current metrics (accuracy, F1, etc.)"""
    return read_json("reports/metrics.json", default={"accuracy": 0, "f1_score": 0, "precision": 0, "recall": 0})


def load_logs():
    """Read logs.txt for live console display."""
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    return lines[-25:]  # last 25 lines for display
