#!/usr/bin/env python3
"""
traci_logger.py
Causal telemetry logger for SynCity (FIXED TLS LOGIC)

Outputs:
time, veh_id,
x, y,
speed, accel, heading, delta_speed,
junction_id,
tls_id, tls_state, time_to_phase_change,
front_vehicle_dist, front_vehicle_speed,
lane_density, avg_lane_speed
"""

import csv
import os
import traci

# ================= CONFIG =================
SUMO_CFG = "city.sumocfg"
OUTPUT_CSV = "telemetry.csv"
SIM_STEP_DURATION = 0.1
MAX_STEPS = 2000

# =========================================
def main():
    # ---- Start SUMO ----
    SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
    sumoBinary = os.path.join(SUMO_HOME, "bin", "sumo.exe")

    traci.start([
        sumoBinary,
        "-c", SUMO_CFG,
        "--step-length", str(SIM_STEP_DURATION)
    ])
    print("DEBUG: Checking TLS visibility")
    for vid in traci.vehicle.getIDList():
        print("Vehicle:", vid)
        print("Next TLS:", traci.vehicle.getNextTLS(vid))
        break


    prev_speeds = {}

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # ================= CSV HEADER =================
        writer.writerow([
            "time", "veh_id",
            "x", "y",
            "speed", "accel", "heading", "delta_speed",
            "junction_id",
            "tls_id", "tls_state", "time_to_phase_change",
            "front_vehicle_dist", "front_vehicle_speed",
            "lane_density", "avg_lane_speed"
        ])

        step = 0
        while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_STEPS:
            sim_time = round(traci.simulation.getTime(), 3)

            for vid in traci.vehicle.getIDList():
                try:
                    # ---------- Vehicle kinematics ----------
                    x, y = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)

                    try:
                        accel = traci.vehicle.getAcceleration(vid)
                    except:
                        accel = 0.0

                    try:
                        heading = traci.vehicle.getAngle(vid)
                    except:
                        heading = 0.0

                    # ---------- Delta speed ----------
                    prev_speed = prev_speeds.get(vid, speed)
                    delta_speed = speed - prev_speed
                    prev_speeds[vid] = speed

                    # ---------- Junction ----------
                    junction_id = traci.vehicle.getRoadID(vid)

                    # ---------- VEHICLE-SPECIFIC TLS STATE (CRITICAL FIX) ----------
                    tls_id = ""
                    tls_state = -1.0
                    time_to_phase_change = -1.0

                    try:
                        tls_info = traci.vehicle.getNextTLS(vid)
                        if tls_info:
                            tls_id, link_index, dist = tls_info[0]

                            full_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                            signal = full_state[link_index]

                            if signal == "r":
                                tls_state = 0.0
                            elif signal == "y":
                                tls_state = 0.5
                            elif signal in ("G", "g"):
                                tls_state = 1.0

                            time_to_phase_change = (
                                traci.trafficlight.getNextSwitch(tls_id)
                                - traci.simulation.getTime()
                            )

                    except traci.exceptions.TraCIException:
                        # Vehicle is not approaching a TLS-controlled junction
                        pass


                    # ---------- Interaction (leader vehicle) ----------
                    leader = traci.vehicle.getLeader(vid)
                    front_vehicle_dist = -1.0
                    front_vehicle_speed = -1.0

                    if leader:
                        front_vehicle_dist = leader[1]
                        front_vehicle_speed = traci.vehicle.getSpeed(leader[0])

                    # ---------- Lane context ----------
                    lane_id = traci.vehicle.getLaneID(vid)
                    lane_density = traci.lane.getLastStepVehicleNumber(lane_id)
                    avg_lane_speed = traci.lane.getLastStepMeanSpeed(lane_id)

                    # ---------- Write row ----------
                    writer.writerow([
                        sim_time, vid,
                        round(x, 3), round(y, 3),
                        round(speed, 3), round(accel, 3),
                        round(heading, 3), round(delta_speed, 3),
                        junction_id,
                        tls_id, tls_state, round(time_to_phase_change, 3),
                        round(front_vehicle_dist, 3),
                        round(front_vehicle_speed, 3),
                        lane_density, round(avg_lane_speed, 3)
                    ])

                except Exception:
                    continue

            traci.simulationStep()
            step += 1

    traci.close()
    print("✅ Finished. Output saved to", OUTPUT_CSV)

# =========================================
if __name__ == "__main__":
    main()
