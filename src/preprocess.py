import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
from dotenv import load_dotenv

RAW_PATH = "data/raw/Machine_downtime.csv"
PROCESSED_PATH = "data/processed/train.csv"

# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def load_env():
    """Load SQL credentials from .env if present"""
    load_dotenv()
    creds = {
        "host": os.getenv("MYSQL_HOST"),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DB"),
    }
    if all(creds.values()):
        print("🔐 SQL credentials loaded.")
        return creds
    else:
        print("⚠️ SQL credentials not found. Using CSV fallback.")
        return None


def get_sql_engine(creds):
    """Return SQLAlchemy engine"""
    return create_engine(
        f"mysql+pymysql://{creds['user']}:{creds['password']}@{creds['host']}/{creds['database']}"
    )


# --------------------------------------------------
# Core preprocessing functions
# --------------------------------------------------
def load_data(creds=None, table_name="train_data"):
    """Load from SQL if creds exist, else from CSV"""
    if creds:
        try:
            engine = get_sql_engine(creds)
            df = pd.read_sql_table(table_name, engine)
            print(f"✅ Loaded data from SQL table: {table_name} ({df.shape})")
            return df
        except Exception as e:
            print(f"❌ SQL load failed, falling back to CSV: {e}")
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"❌ Raw file not found: {RAW_PATH}")
    print(f"✅ Loaded raw data: {RAW_PATH}")
    return pd.read_csv(RAW_PATH)


def clean_data(df):
    """Drop duplicates and handle missing values"""
    df = df.drop_duplicates()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    print("🧹 Data cleaned.")
    return df


def encode_labels(df):
    """Encode target label 'Downtime'"""
    if "Downtime" in df.columns:
        le = LabelEncoder()
        df["Downtime"] = le.fit_transform(df["Downtime"])
        print("🔢 Encoded target column 'Downtime'.")
    return df


def save_outputs(df, creds=None, table_name="train_data"):
    """Save cleaned data to CSV and optionally SQL"""
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"💾 Saved processed data → {PROCESSED_PATH}")

    if creds:
        try:
            engine = get_sql_engine(creds)
            df.to_sql(table_name, engine, if_exists="replace", index=False)
            print(f"🗄️ Saved processed data to SQL table: {table_name}")
        except Exception as e:
            print(f"❌ Failed to save to SQL: {e}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    creds = load_env()
    df = load_data(creds)
    df = clean_data(df)
    df = encode_labels(df)
    save_outputs(df, creds)
    print(f"✅ Preprocessing complete! Final shape: {df.shape}")


if __name__ == "__main__":
    main()
