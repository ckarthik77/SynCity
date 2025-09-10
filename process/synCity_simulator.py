import random
import pandas as pd

# Signal cycle (G=30s, Y=5s, R=45s)
def get_signal_state(t):
    cycle = (t % 80)  # 30+5+45 = 80 seconds
    if cycle < 30:
        return "GREEN"
    elif cycle < 35:
        return "YELLOW"
    else:
        return "RED"

vehicles = [{"id": i, "x": 77.5946, "y": 12.9716, "speed": random.randint(20, 40)} for i in range(10)]
log = []

for t in range(200):  # 200 seconds simulation
    signal = get_signal_state(t)
    congestion = random.randint(5, 25)  # number of vehicles in same area
    
    for v in vehicles:
        # Adjust speed depending on signal
        if signal == "RED":
            v["speed"] = max(0, v["speed"] - random.randint(5, 15))
        elif signal == "GREEN":
            v["speed"] = min(80, v["speed"] + random.randint(1, 5))
        else:  # YELLOW
            v["speed"] = max(5, v["speed"] - random.randint(1, 3))
        
        # Random GPS drift
        v["x"] += random.uniform(-0.0005, 0.0005)
        v["y"] += random.uniform(-0.0005, 0.0005)
        
        log.append({
            "time": t,
            "vehicle_id": v["id"],
            "x": round(v["x"], 6),
            "y": round(v["y"], 6),
            "speed": v["speed"],
            "acceleration": random.uniform(-3, 3),
            "signal_state": signal,
            "congestion_level": congestion
        })

df = pd.DataFrame(log)
df.to_excel("synCity_realtime_dataset.xlsx", index=False)
print("Dataset saved as synCity_realtime_dataset.xlsx")
