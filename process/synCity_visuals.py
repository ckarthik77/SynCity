import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("synCity_dataset.csv")

# Plot speed over time for each vehicle
for vid, group in df.groupby("vehicle_id"):
    plt.plot(group["time"], group["speed"], label=f"Vehicle {vid}")

plt.xlabel("Time")
plt.ylabel("Speed (km/h)")
plt.title("Vehicle Speeds Over Time")
plt.legend()
plt.show()
