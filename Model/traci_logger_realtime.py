import traci
import numpy as np
import csv
import sys
import os
from collections import defaultdict

# Import predictor
from inference_realtime import TrafficPredictor

# SUMO Configuration
SUMO_CONFIG = "../Test/city.sumocfg"  # Adjust path to your .sumocfg file
STEP_LENGTH = 1.0  # seconds per simulation step

# Initialize predictor
print("Initializing AI Traffic Predictor...")
predictor = TrafficPredictor(
    model_path='multihorizon_lstm.pth',
    scaler_path='multihorizon_scaler.pkl'
)

# CSV output
output_file = "realtime_predictions.csv"
csv_file = open(output_file, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'time', 'vehicle_id', 'current_speed', 
    'predicted_30s', 'predicted_90s', 'predicted_150s',
    'delta_30s', 'delta_90s', 'delta_150s'
])

def get_lane_metrics(lane_id):
    """Calculate lane-level metrics"""
    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
    
    if len(vehicle_ids) == 0:
        return 0.0, 0.0
    
    speeds = [traci.vehicle.getSpeed(vid) for vid in vehicle_ids]
    lane_length = traci.lane.getLength(lane_id)
    
    density = len(vehicle_ids) / lane_length if lane_length > 0 else 0
    avg_speed = np.mean(speeds) if speeds else 0
    
    return density, avg_speed

def run_simulation():
    """Run SUMO with real-time AI predictions"""
    
    # Start SUMO
    sumo_binary = "sumo-gui"  # Use "sumo" for headless
    sumo_cmd = [sumo_binary, "-c", SUMO_CONFIG, "--step-length", str(STEP_LENGTH)]
    
    traci.start(sumo_cmd)
    
    step = 0
    prediction_count = 0
    
    print("\n[START] Starting real-time traffic prediction...")
    print("=" * 70)
    
    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step < 3600:  # Run for 1 hour
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            
            # Get all active vehicles
            vehicle_ids = traci.vehicle.getIDList()
            
            for vehicle_id in vehicle_ids:
                try:
                    # Get vehicle state
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    accel = traci.vehicle.getAcceleration(vehicle_id)
                    lane_id = traci.vehicle.getLaneID(vehicle_id)
                    
                    # Front vehicle
                    leader = traci.vehicle.getLeader(vehicle_id)
                    if leader:
                        front_vehicle_dist = leader[1]
                        front_vehicle_speed = traci.vehicle.getSpeed(leader[0])
                    else:
                        front_vehicle_dist = 100.0  # No leader
                        front_vehicle_speed = speed
                    
                    # Lane metrics
                    lane_density, avg_lane_speed = get_lane_metrics(lane_id)
                    
                    # Update predictor
                    predictor.update_vehicle_data(
                        vehicle_id=vehicle_id,
                        speed=speed,
                        accel=accel,
                        front_vehicle_dist=front_vehicle_dist,
                        front_vehicle_speed=front_vehicle_speed,
                        lane_density=lane_density,
                        avg_lane_speed=avg_lane_speed
                    )
                    
                    # Get predictions (only every 5 seconds to reduce overhead)
                    if step % 5 == 0:
                        result = predictor.get_predicted_speed(vehicle_id)
                        
                        if result:
                            # Log to CSV
                            csv_writer.writerow([
                                current_time,
                                vehicle_id,
                                result['current_speed'],
                                result['predicted_30s'],
                                result['predicted_90s'],
                                result['predicted_150s'],
                                result['delta_30s'],
                                result['delta_90s'],
                                result['delta_150s']
                            ])
                            
                            prediction_count += 1
                            
                            # Print sample predictions
                            if prediction_count % 50 == 0:
                                print(f"[{current_time:.0f}s] {vehicle_id}: "
                                      f"Current={result['current_speed']:.1f} | "
                                      f"30s={result['predicted_30s']:.1f} | "
                                      f"90s={result['predicted_90s']:.1f} | "
                                      f"150s={result['predicted_150s']:.1f}")
                
                except traci.exceptions.TraCIException:
                    continue
            
            step += 1
            
            # Progress indicator
            if step % 100 == 0:
                print(f"[TIME] Simulation time: {current_time:.0f}s | Predictions made: {prediction_count}")
    
    except KeyboardInterrupt:
        print("\n[WARNING] Simulation interrupted by user")
    
    finally:
        print("\n" + "=" * 70)
        print(f"[DONE] Simulation complete!")
        print(f"   Total predictions: {prediction_count}")
        print(f"   Output file: {output_file}")
        print("=" * 70)
        
        traci.close()
        csv_file.close()

if __name__ == "__main__":
    run_simulation()