# src/train.py
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

DATA_PATH = "data/processed/train.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "reports/metrics.json"

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Processed data not found at {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded processed data: {df.shape}")
    return df

def split_data(df):
    # Drop non-numeric or timestamp columns (like Date)
    X = df.drop(columns=["Downtime"], errors="ignore")
    # Keep only numeric columns for training
    X = X.select_dtypes(include=["number"])
    y = df["Downtime"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

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
    report = classification_report(y_test, y_pred, target_names=["NON_FAILURE", "FAILURE"])
    print("\n📊 Classification Report:\n", report)
    return {"accuracy": acc, "f1_score": f1}

def save_artifacts(model, metrics):
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"📈 Metrics saved → {METRICS_PATH}")

def main():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_res, y_train_res = balance_data(X_train, y_train)
    model = train_model(X_train_res, y_train_res)
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, metrics)
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
