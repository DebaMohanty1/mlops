import pandas as pd, numpy as np, os

os.makedirs("data/live", exist_ok=True)
df = pd.read_csv("data/raw/Machine_downtime.csv")

df_low = df.copy()
df_mid = df.copy()
df_high = df.copy()

# ------------- simulate drift -------------
np.random.seed(42)
df_low["Hydraulic_Pressure(bar)"] *= np.random.normal(1.05, 0.02, len(df))
df_mid["Hydraulic_Pressure(bar)"] *= np.random.normal(1.10, 0.03, len(df))
df_high["Hydraulic_Pressure(bar)"] *= np.random.normal(1.25, 0.05, len(df))

# Add more drift features for realism
for d in [df_low, df_mid, df_high]:
    d["Coolant_Temperature(°C)"] += np.random.normal(0.5, 0.3, len(d))

# ------------- save files -------------
df_low.to_csv("data/live/current_low.csv", index=False)
df_mid.to_csv("data/live/current_mid.csv", index=False)
df_high.to_csv("data/live/current_high.csv", index=False)
print("✅ Simulated drift datasets saved to data/live/")
