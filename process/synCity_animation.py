import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load dataset
df = pd.read_excel("synCity_realtime_dataset.xlsx")

# Setup plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(df["x"].min() - 0.001, df["x"].max() + 0.001)
ax.set_ylim(df["y"].min() - 0.001, df["y"].max() + 0.001)
ax.set_title("SynCity Traffic Simulation")

# Scatter plot for vehicles
scat = ax.scatter([], [], c="blue")

def update(frame):
    data = df[df["time"] == frame]
    scat.set_offsets(data[["x", "y"]])
    ax.set_xlabel(f"Time = {frame}s | Signal: {data['signal_state'].iloc[0]}")
    return scat,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=df["time"].max(), interval=200, blit=True)

# Save as MP4
ani.save("synCity_traffic_demo.gif", writer="pillow")

print("Video saved as synCity_traffic_demo.mp4")
