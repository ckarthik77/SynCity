#!/usr/bin/env python3
"""
traci_logger_v2.py
Clean telemetry logger for SynCity - Optimized for Multi-Horizon Prediction

Outputs CLEAN dataset with only essential features:
- time, veh_id, speed, accel
- front_vehicle_dist, front_vehicle_speed
- lane_density, avg_lane_speed
"""

import csv
import os
import traci
import numpy as np

# ================= CONFIG =================
SUMO_CFG = "city.sumocfg"
OUTPUT_CSV = "syncity_clean_dataset.csv"
SIM_STEP_DURATION = 0.5  # 0.5 seconds for proper 30s/90s/150s predictions
MAX_STEPS = 10000  # Longer simulation = more training data
MIN_VEHICLE_LIFETIME = 350  # Only track vehicles that live >175 seconds (for 150s predictions)

print("="*80)
print("SynCity Data Generation - Clean Multi-Horizon Dataset")
print("="*80)
print(f"Configuration:")
print(f"  Timestep:     {SIM_STEP_DURATION}s")
print(f"  Max steps:    {MAX_STEPS}")
print(f"  Min lifetime: {MIN_VEHICLE_LIFETIME} steps ({MIN_VEHICLE_LIFETIME * SIM_STEP_DURATION}s)")
print(f"  Output:       {OUTPUT_CSV}")
print("="*80)

def main():
    # ---- Start SUMO ----
    SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
    sumoBinary = os.path.join(SUMO_HOME, "bin", "sumo.exe")

    traci.start([
        sumoBinary,
        "-c", SUMO_CFG,
        "--step-length", str(SIM_STEP_DURATION),
        "--no-warnings",
        "--no-step-log"
    ])
    
    print("\n✅ SUMO simulation started\n")

    # Track vehicle data over time
    vehicle_trajectories = {}  # {veh_id: [data_rows]}
    
    step = 0
    rows_collected = 0
    
    while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_STEPS:
        sim_time = round(traci.simulation.getTime(), 3)

        for vid in traci.vehicle.getIDList():
            try:
                # ========== ESSENTIAL FEATURES ONLY ==========
                
                # Speed and acceleration
                speed = traci.vehicle.getSpeed(vid)
                
                try:
                    accel = traci.vehicle.getAcceleration(vid)
                except:
                    accel = 0.0

                # Leader vehicle (critical for prediction)
                leader = traci.vehicle.getLeader(vid)
                front_vehicle_dist = -1.0
                front_vehicle_speed = -1.0

                if leader:
                    front_vehicle_dist = leader[1]
                    front_vehicle_speed = traci.vehicle.getSpeed(leader[0])

                # Lane context (traffic density)
                lane_id = traci.vehicle.getLaneID(vid)
                lane_density = traci.lane.getLastStepVehicleNumber(lane_id)
                avg_lane_speed = traci.lane.getLastStepMeanSpeed(lane_id)

                # Store data
                row = [
                    sim_time, 
                    vid,
                    round(speed, 4),
                    round(accel, 4),
                    round(front_vehicle_dist, 4),
                    round(front_vehicle_speed, 4),
                    lane_density,
                    round(avg_lane_speed, 4)
                ]
                
                if vid not in vehicle_trajectories:
                    vehicle_trajectories[vid] = []
                
                vehicle_trajectories[vid].append(row)
                rows_collected += 1

            except Exception as e:
                continue

        traci.simulationStep()
        step += 1
        
        # Progress update every 500 steps
        if step % 500 == 0:
            print(f"  Step {step}/{MAX_STEPS} | Active vehicles: {len(traci.vehicle.getIDList())} | Rows: {rows_collected}")

    traci.close()
    
    print(f"\n✅ Simulation complete!")
    print(f"  Total steps: {step}")
    print(f"  Total vehicles tracked: {len(vehicle_trajectories)}")
    print(f"  Total data rows: {rows_collected}")
    
    # ========== FILTER & SAVE CLEAN DATA ==========
    print(f"\n📊 Filtering vehicles with lifetime >= {MIN_VEHICLE_LIFETIME} steps...")
    
    valid_vehicles = 0
    valid_rows = 0
    
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow([
            "time", "veh_id",
            "speed", "accel",
            "front_vehicle_dist", "front_vehicle_speed",
            "lane_density", "avg_lane_speed"
        ])
        
        # Write only vehicles with sufficient lifetime
        for vid, trajectory in vehicle_trajectories.items():
            if len(trajectory) >= MIN_VEHICLE_LIFETIME:
                for row in trajectory:
                    writer.writerow(row)
                valid_vehicles += 1
                valid_rows += len(trajectory)
    
    print(f"\n✅ Dataset saved to {OUTPUT_CSV}")
    print(f"\n📈 Final Statistics:")
    print(f"  Valid vehicles:  {valid_vehicles}")
    print(f"  Valid data rows: {valid_rows}")
    print(f"  Avg trajectory:  {valid_rows/max(valid_vehicles,1):.0f} timesteps")
    print(f"  Sequence length: 30 timesteps (15 seconds)")
    print(f"  Prediction horizons: 60/180/300 steps (30s/90s/150s)")
    print(f"  Expected sequences: ~{max(0, valid_rows - 330 * valid_vehicles)}")
    print("="*80)

if __name__ == "__main__":
    main()