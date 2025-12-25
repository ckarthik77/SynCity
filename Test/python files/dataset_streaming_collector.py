#!/usr/bin/env python3
# dataset_streaming_collector.py
# Run with: python dataset_streaming_collector.py
# Streams WebSocket at ws://localhost:8765/

import traci
import csv
import json
import threading
import time
from typing import Dict, Any, List
import asyncio
import websockets

SUMO_CMD = ["sumo", "-c", "city.sumocfg"]  # change to sumo-gui if you want GUI
WS_HOST = "localhost"
WS_PORT = 8765
WS_SEND_INTERVAL = 0.1  # seconds between sends to connected clients

# Shared snapshot for websockets (protected by lock)
latest_snapshot = None
snapshot_lock = threading.Lock()

def detect_stop_reason(veh_id):
    """
    Heuristic for stop reason:
    - red_light: stopped and nearest TLS is red and close
    - yield: stopped because leader vehicle is very close ahead
    - congestion: waitingTime > threshold
    - stopped: generic
    """
    try:
        speed = traci.vehicle.getSpeed(veh_id)
        if speed > 0.1:
            return ""  # moving

        # moving is near 0: determine cause
        # next TLS
        tls_info = traci.vehicle.getNextTLS(veh_id)
        if tls_info:
            tls_id, dist, state, _ = tls_info[0]
            # state is like 'r', 'G' or combination for multi-lane; test for 'r' or 'R'
            if dist >= 0 and dist < 7 and ('r' in state.lower()):
                return "red_light"

        # leader (vehicle ahead in same lane)
        leader = traci.vehicle.getLeader(veh_id)
        if leader is not None:
            leader_id, leader_dist = leader
            if leader_dist is not None and leader_dist < 5.0:
                return "yield"

        # waiting time - if waitingTime large -> congestion
        wait = traci.vehicle.getWaitingTime(veh_id)
        if wait is not None and wait > 5.0:
            return "congestion"

        # default
        return "stopped"

    except Exception:
        return "unknown"

def get_congestion(edge_id):
    """Simple congestion score: vehicles per km on first lane of edge."""
    try:
        lane_id = f"{edge_id}_0"
        vehs = traci.lane.getLastStepVehicleNumber(lane_id)
        length_m = traci.lane.getLength(lane_id)
        if length_m <= 0:
            return 0.0
        return round((vehs / (length_m / 1000.0 + 1e-6)), 3)
    except Exception:
        return 0.0

def simulation_loop(csv_path="dataset.csv"):
    global latest_snapshot
    # Start SUMO
    print("Starting SUMO...")
    traci.start(SUMO_CMD)
    print("SUMO started.")

    # Open CSV writer
    fields = [
        "time", "veh_id",
        "x", "y",
        "speed", "acceleration",
        "edge", "lane", "angle",
        "nearest_tls", "tls_state", "dist_to_tls",
        "congestion",
        "veh_type", "route_id", "stop_reason", "waiting_time"
    ]
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(fields)

    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            vehicles = traci.vehicle.getIDList()
            snapshot: List[Dict[str, Any]] = []

            for vid in vehicles:
                try:
                    x, y = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    accel = traci.vehicle.getAcceleration(vid)
                    edge = traci.vehicle.getRoadID(vid)
                    lane = traci.vehicle.getLaneID(vid)
                    angle = traci.vehicle.getAngle(vid)

                    # tls data
                    tls_info = traci.vehicle.getNextTLS(vid)
                    if tls_info:
                        tls_id, dist_to_tls, state, _ = tls_info[0]
                    else:
                        tls_id, dist_to_tls, state = "none", -1.0, "none"

                    cong = get_congestion(edge) if edge not in ["", "None"] else 0.0

                    # vehicle type + route id
                    try:
                        veh_type = traci.vehicle.getTypeID(vid)
                    except Exception:
                        veh_type = "unknown"
                    try:
                        route_id = traci.vehicle.getRouteID(vid)
                    except Exception:
                        route_id = "unknown"

                    waiting_time = traci.vehicle.getWaitingTime(vid)
                    stop_reason = detect_stop_reason(vid)

                    # CSV row
                    writer.writerow([
                        step, vid,
                        x, y,
                        speed, accel,
                        edge, lane, angle,
                        tls_id, state, dist_to_tls,
                        cong,
                        veh_type, route_id, stop_reason, waiting_time
                    ])

                    # Build snapshot item for websocket
                    snapshot.append({
                        "time": step,
                        "veh_id": vid,
                        "x": x, "y": y,
                        "speed": speed,
                        "acceleration": accel,
                        "edge": edge, "lane": lane, "angle": angle,
                        "nearest_tls": tls_id, "tls_state": state, "dist_to_tls": dist_to_tls,
                        "congestion": cong,
                        "veh_type": veh_type,
                        "route_id": route_id,
                        "stop_reason": stop_reason,
                        "waiting_time": waiting_time
                    })
                except Exception as ex_v:
                    # protect per-vehicle errors
                    # print(f"Vehicle read error {vid}: {ex_v}")
                    continue

            # update shared snapshot for websockets
            with snapshot_lock:
                latest_snapshot = {"time": step, "vehicles": snapshot}

            step += 1

    finally:
        print("Simulation finished or interrupted.")
        try:
            traci.close()
        except Exception:
            pass
        f.close()
        # final snapshot push
        with snapshot_lock:
            latest_snapshot = {"time": step, "vehicles": []}
        print("CSV saved to", csv_path)

async def ws_handler(websocket, path):
    """
    Sends the latest_snapshot periodically to any connected client.
    Simple: clients receive full snapshot each interval.
    """
    print("WS client connected:", websocket.remote_address)
    try:
        while True:
            await asyncio.sleep(WS_SEND_INTERVAL)
            with snapshot_lock:
                snap = latest_snapshot
            if snap is None:
                # nothing yet
                continue
            try:
                await websocket.send(json.dumps(snap))
            except websockets.ConnectionClosed:
                break
    except asyncio.CancelledError:
        pass
    finally:
        print("WS client disconnected:", websocket.remote_address)

def start_websocket_server():
    # run an asyncio event loop for websockets
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(ws_handler, WS_HOST, WS_PORT)
    ws_server = loop.run_until_complete(start_server)
    print(f"WebSocket server running on ws://{WS_HOST}:{WS_PORT}/")
    try:
        loop.run_forever()
    finally:
        ws_server.close()
        loop.run_until_complete(ws_server.wait_closed())
        loop.close()

if __name__ == "__main__":
    # Start websocket server in separate thread
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()

    # Run simulation loop (blocking) in main thread
    simulation_loop(csv_path="dataset.csv")

    # On finish, keep server alive for a short grace, then exit
    print("Waiting 3s for any final websocket sends...")
    time.sleep(3)
    print("Exiting.")
