import pandas as pd
df = pd.read_csv(r"E:\Desktop\Jobs\projects\Syncity\Test\Olddataset_2.csv")
print(f"Speed range: {df['speed'].min():.2f} to {df['speed'].max():.2f} m/s")
print(f"Speed mean: {df['speed'].mean():.2f} m/s")
print(f"Speed std: {df['speed'].std():.2f} m/s")