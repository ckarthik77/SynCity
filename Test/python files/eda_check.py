import pandas as pd

df = pd.read_csv("telemetry.csv")

print("Total rows:", len(df))
print("Unique vehicles:", df["veh_id"].nunique())
print("Simulation time range:", df["time"].min(), "to", df["time"].max())
print("Edges observed:", df["edge"].unique()[:10])
print("TLS observed:", df["nearest_tls"].unique())
print("Missing values:\n", df.isna().sum())
