# src/preprocess.py
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

RAW_PATH = "data/raw/Machine_downtime.csv"
PROCESSED_PATH = "data/processed/train.csv"

def load_raw_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Raw file not found: {path}")
    print(f"✅ Loaded raw data: {path}")
    return pd.read_csv(path)

def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing numeric values with median
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with mode
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def encode_labels(df):
    if "Downtime" in df.columns:
        le = LabelEncoder()
        df["Downtime"] = le.fit_transform(df["Downtime"])
    return df

def main():
    os.makedirs("data/processed", exist_ok=True)

    df = load_raw_data(RAW_PATH)
    df = clean_data(df)
    df = encode_labels(df)

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"💾 Processed data saved to {PROCESSED_PATH}")
    print(f"✅ Shape: {df.shape}")

if __name__ == "__main__":
    main()
