import pandas as pd
INPUT="telemetry.csv"
OUT="edge_aggregates.csv"

chunksize=200000
first=True
for c in pd.read_csv(INPUT, chunksize=chunksize):
    agg = c.groupby(["time","edge"]).agg(
        vehicles=("veh_id","nunique"),
        mean_speed=("speed","mean"),
        max_speed=("speed","max")
    ).reset_index()
    agg.to_csv(OUT, mode='a', header=first, index=False)
    first=False
print("Aggregates saved:", OUT)
