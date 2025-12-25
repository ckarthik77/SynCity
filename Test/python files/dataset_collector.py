import traci
import csv

# Start SUMO (non-GUI for speed)
traci.start(["sumo", "-c", "city.sumocfg"])

fields = [
    "time", "veh_id",
    "x", "y",
    "speed", "acceleration",
    "edge", "lane", "angle",
    "nearest_tls", "tls_state", "dist_to_tls",
    "congestion"
]

f = open("dataset.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(fields)

def get_congestion(edge):
    try:
        lane = edge + "_0"
        vehs = traci.lane.getLastStepVehicleNumber(lane)
        length = traci.lane.getLength(lane)
        return round(vehs / (length / 1000 + 1e-6), 3)
    except:
        return 0

step = 0

# Run until all vehicles finish
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    for vid in traci.vehicle.getIDList():
        # Basic data
        x, y = traci.vehicle.getPosition(vid)
        speed = traci.vehicle.getSpeed(vid)
        accel = traci.vehicle.getAcceleration(vid)
        edge = traci.vehicle.getRoadID(vid)
        lane = traci.vehicle.getLaneID(vid)
        angle = traci.vehicle.getAngle(vid)

        # Traffic light interaction
        tls_info = traci.vehicle.getNextTLS(vid)
        if tls_info:
            tls_id, dist, state, _ = tls_info[0]
        else:
            tls_id, dist, state = "none", -1, "none"

        # Congestion level
        cong = get_congestion(edge)

        writer.writerow([
            step, vid,
            x, y,
            speed, accel,
            edge, lane, angle,
            tls_id, state, dist,
            cong
        ])

    step += 1

traci.close()
f.close()
print(" Dataset saved → dataset.csv ")
