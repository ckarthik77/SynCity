import pandas as pd
N = 5               # keep every 5th timestep (0.1s -> 0.5s)
INPUT="telemetry.csv"
OUT="telemetry_time_sampled.csv"

chunks = pd.read_csv(INPUT, chunksize=2_000_00)
first = True
cnt = 0
for c in chunks:
    c = c[c.index % N == 0]   # simple downsample within chunk
    c.to_csv(OUT, mode='a', header=first, index=False)
    first = False
    cnt += len(c)
print("Wrote", cnt, "rows to", OUT)
