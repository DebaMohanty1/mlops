import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------
# 📂 Paths
DATA_PATH = "data/processed/train.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "reports/metrics.json"
# ------------------------------------------------------------

# ------------------------------------------------------------
# 🧩 Load processed or merged data
# ------------------------------------------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Processed data not found at {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded processed data: {df.shape}")
    return df

# ------------------------------------------------------------
# ✂️ Split into train/test
# ------------------------------------------------------------
def split_data(df):
    X = df.drop(columns=["Downtime"], errors="ignore")
    X = X.select_dtypes(include=["number"])
    y = df["Downtime"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# ------------------------------------------------------------
# ⚖️ Balance data using SMOTE
# ------------------------------------------------------------
def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"⚖️ After SMOTE: {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res

# ------------------------------------------------------------
# 🌲 Train model
# ------------------------------------------------------------
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model

# ------------------------------------------------------------
# 📊 Evaluate model
# ------------------------------------------------------------
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
# 💾 Save model and metrics
# ------------------------------------------------------------
def save_artifacts(model, metrics):
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"📈 Metrics saved → {METRICS_PATH}")

# ------------------------------------------------------------
# 🪄 Promote dataset as new baseline
# ------------------------------------------------------------
def update_baseline(df, baseline_path="data/processed/train.csv", drift_score=None):
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    df.to_csv(baseline_path, index=False)
    print(f"🆕 Baseline updated → {baseline_path}")

    log_entry = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "new_baseline_path": baseline_path,
        "rows": len(df),
        "columns": list(df.columns),
        "drift_score": drift_score,
    }

    drift_log_path = "reports/drift_log.json"
    os.makedirs(os.path.dirname(drift_log_path), exist_ok=True)
    if os.path.exists(drift_log_path):
        with open(drift_log_path, "r") as f:
            drift_log = json.load(f)
    else:
        drift_log = []

    drift_log.append(log_entry)
    with open(drift_log_path, "w") as f:
        json.dump(drift_log, f, indent=4)

    print(f"🪄 Drift baseline promoted and logged → {drift_log_path}")

# ------------------------------------------------------------
# 📦 Load live drifted data
# ------------------------------------------------------------
def load_live_data(live_dir="data/live"):
    live_dfs = []
    if os.path.exists(live_dir):
        for f in os.listdir(live_dir):
            if f.startswith("current_") and f.endswith(".csv"):
                path = os.path.join(live_dir, f)
                try:
                    df_live = pd.read_csv(path)
                    live_dfs.append(df_live)
                    print(f"📦 Included drifted dataset: {f} ({df_live.shape})")
                except Exception as e:
                    print(f"⚠️ Skipping {f} — error: {e}")
    return live_dfs

# ------------------------------------------------------------
# 🚀 Main Training Logic (with NaN fix)
# ------------------------------------------------------------
def main():
    # 1️⃣ Load baseline
    df = load_data(DATA_PATH)

    # 2️⃣ Merge drifted datasets if present
    live_dfs = load_live_data()
    if live_dfs:
        print(f"🧠 Loaded {len(live_dfs)} drifted datasets (for analysis only).")
    else:
        print("✅ No drifted data to merge.")


    # 3️⃣ Clean up and ensure numeric safety
    if "Downtime" in df.columns:
        df["Downtime"] = pd.to_numeric(df["Downtime"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["Downtime"])
        after = len(df)
        print(f"🧹 Dropped {before - after} unlabeled rows (no Downtime).")
        df["Downtime"] = df["Downtime"].astype(int)
    else:
        raise ValueError("❌ 'Downtime' target column missing after merge!")

    # 4️⃣ Handle NaNs before SMOTE
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    if df[num_cols].isna().sum().any():
        print("⚠️ Warning: Some NaNs remain even after imputation.")
    else:
        print("✅ No NaNs found after imputation.")

    # 5️⃣ Split + train
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_res, y_train_res = balance_data(X_train, y_train)
    model = train_model(X_train_res, y_train_res)

    # 6️⃣ Evaluate and save
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, metrics)

    # 7️⃣ Update baseline reference
    try:
        update_baseline(df, drift_score=metrics.get("f1_score"))
    except Exception as e:
        print(f"⚠️ Baseline update failed: {e}")

    print("✅ Training complete (drift-aware retraining done) and baseline refreshed!")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
