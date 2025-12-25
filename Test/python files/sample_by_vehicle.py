import pandas as pd
import numpy as np

INPUT="telemetry.csv"
OUT="telemetry_vehicle_sampled.csv"
K = 1000   # target number of unique vehicles

df = pd.read_csv(INPUT, usecols=["veh_id"])
unique = df["veh_id"].unique()
sample_ids = np.random.choice(unique, min(K, len(unique)), replace=False)
chunksize = 200000
first=True
for c in pd.read_csv(INPUT, chunksize=chunksize):
    sel = c[c["veh_id"].isin(sample_ids)]
    sel.to_csv(OUT, mode='a', header=first, index=False)
    first=False
print("Sample saved:", OUT)
