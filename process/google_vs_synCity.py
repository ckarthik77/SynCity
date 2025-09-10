import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load your dataset
df = pd.read_excel("synCity_realtime_dataset.xlsx")

# Setup figure with two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# --------------------------
# Google Maps (Static)
# --------------------------
ax1.set_title("Google Maps (Static)")
ax1.set_xlim(df["x"].min() - 0.001, df["x"].max() + 0.001)
ax1.set_ylim(df["y"].min() - 0.001, df["y"].max() + 0.001)
gm_scat = ax1.scatter([], [], c="red")

# --------------------------
# SynCity (Dynamic)
# --------------------------
ax2.set_title("SynCity (Dynamic, Real-Time)")
ax2.set_xlim(df["x"].min() - 0.001, df["x"].max() + 0.001)
ax2.set_ylim(df["y"].min() - 0.001, df["y"].max() + 0.001)
sc_scat = ax2.scatter([], [], c="blue")

def update(frame):
    data = df[df["time"] == frame]

    # Google Maps side: pretend traffic is static (all red stuck cars)
    gm_scat.set_offsets([[77.595, 12.972]] * len(data))  

    # SynCity side: plot actual dynamic positions
    sc_scat.set_offsets(data[["x", "y"]])

    # Update title with current signal state
    ax2.set_xlabel(f"Time = {frame}s | Signal: {data['signal_state'].iloc[0]}")

    return gm_scat, sc_scat

ani = animation.FuncAnimation(fig, update, frames=df["time"].max(),
                              interval=200, blit=True)

# Save as GIF (works in PowerPoint and browsers)
ani.save("google_vs_synCity.gif", writer="pillow")
print("✅ Animation saved as google_vs_synCity.gif")
