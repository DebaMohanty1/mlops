# ============================================================
# 🚀 Train & Register Model (with MLflow + GitHub Auto Push)
# ============================================================

import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import subprocess
from datetime import datetime

# ------------------------------------------------------------
# 📂 Paths
# ------------------------------------------------------------
DATA_PATH = "data/processed/train.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "reports/metrics.json"
DRIFT_LOG = "reports/drift_log.json"

# ------------------------------------------------------------
# 🔧 Helper: GitHub Auto Push
# ------------------------------------------------------------
def auto_push_to_github(message="Auto update after retrain"):
    load_dotenv()
    user = os.getenv("GITHUB_USER")
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")

    if not all([user, token, repo]):
        print("⚠️ Missing GitHub credentials in .env — skipping push.")
        return

    try:
        remote_url = f"https://{user}:{token}@github.com/{user}/{repo}.git"
        subprocess.run(["git", "config", "user.name", user], check=False)
        subprocess.run(["git", "config", "user.email", f"{user}@users.noreply.github.com"], check=False)
        subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=False)
        subprocess.run(["git", "add", "-A"], check=False)
        subprocess.run(["git", "commit", "-m", message], check=False)
        subprocess.run(["git", "push", "origin", "main"], check=False)
        print("🚀 Auto-pushed latest artifacts to GitHub.")
    except Exception as e:
        print(f"❌ GitHub push failed: {e}")

# ------------------------------------------------------------
# 🧩 Data Load & Split
# ------------------------------------------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Processed data not found at {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded processed data: {df.shape}")
    return df

def split_data(df):
    X = df.drop(columns=["Downtime"], errors="ignore")
    X = X.select_dtypes(include=["number"])
    y = df["Downtime"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------------------------------------
# ⚖️ Balance + Train + Evaluate
# ------------------------------------------------------------
def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"⚖️ After SMOTE: {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["NON_FAILURE", "FAILURE"]
    )
    print("\n📊 Classification Report:\n", report)
    return {"accuracy": acc, "f1_score": f1}

# ------------------------------------------------------------
# 💾 Save + MLflow Log + GitHub Push
# ------------------------------------------------------------
def save_artifacts(model, metrics):
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"📈 Metrics saved → {METRICS_PATH}")

def update_drift_log(df, drift_score):
    os.makedirs("reports", exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
        "drift_score": drift_score,
    }

    log = []
    if os.path.exists(DRIFT_LOG):
        with open(DRIFT_LOG, "r") as f:
            log = json.load(f)
    log.append(entry)

    with open(DRIFT_LOG, "w") as f:
        json.dump(log, f, indent=4)

    print(f"🪄 Drift log updated → {DRIFT_LOG}")

# ------------------------------------------------------------
# 🚀 Main Training Routine
# ------------------------------------------------------------
def main():
    load_dotenv()

    # 1️⃣ Load & Clean
    df = load_data(DATA_PATH)
    if df.empty or len(df) == 0:
        raise RuntimeError(f"{DATA_PATH} is empty. Aborting retrain to avoid invalid model.")

    if "Downtime" not in df.columns:
        raise ValueError("❌ Missing 'Downtime' column in dataset.")

    df["Downtime"] = pd.to_numeric(df["Downtime"], errors="coerce")
    df = df.dropna(subset=["Downtime"])
    df["Downtime"] = df["Downtime"].astype(int)

    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 2️⃣ Split + Train
    X_train, X_test, y_train, y_test = split_data(df)
    X_res, y_res = balance_data(X_train, y_train)
    model = train_model(X_res, y_res)
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, metrics)

    # 3️⃣ 🔬 MLflow Tracking + Registry
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("machine_downtime_monitoring")

    with mlflow.start_run(run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params({"model_type": "RandomForest", "n_estimators": 200})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", registered_model_name="MachineDowntimeModel")

    print("📦 Model registered in MLflow → MachineDowntimeModel")

    # 4️⃣ Update Drift Log + Push
    update_drift_log(df, metrics.get("f1_score"))
    auto_push_to_github("Auto: retrained model + updated metrics + drift log")

    print("\n✅ Retraining complete! Model, metrics, and drift logs updated.")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
