# src/evaluate.py
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

DATA_PATH = "data/processed/train.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "reports/metrics.json"

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Data not found at {path}")
    df = pd.read_csv(path)
    return df

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model not found at {path}")
    model = joblib.load(path)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print("\n📊 Evaluation Report:\n")
    print(classification_report(y_test, y_pred, target_names=["NON_FAILURE", "FAILURE"]))

    return {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4)
    }

def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"📈 Metrics saved → {path}")

def main():
    df = load_data(DATA_PATH)

    if "Downtime" not in df.columns:
        raise ValueError("❌ Missing 'Downtime' column in dataset.")

    # Drop non-numeric or timestamp columns (like Date, Machine_ID)
    X = df.drop(columns=["Downtime"], errors="ignore")
    X = X.select_dtypes(include=["number"])
    y = df["Downtime"]


    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = load_model(MODEL_PATH)
    metrics = evaluate(model, X_test, y_test)
    save_metrics(metrics, METRICS_PATH)

    print("✅ Evaluation complete!")

if __name__ == "__main__":
    main()
