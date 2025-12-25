"""
generate_routes.py
Creates vehicle routes for city_tls.net.xml using SUMO's randomTrips tool
"""

import subprocess
import os
import sys

SUMO_TOOLS = r"C:\Program Files (x86)\Eclipse\Sumo\tools\randomTrips.py"
NET_FILE = "city_tls.net.xml"
ROUTE_FILE = "city_tls.rou.xml"

def main():
    if not os.path.exists(NET_FILE):
        print("❌ Network file not found:", NET_FILE)
        sys.exit(1)

    print("🚗 Generating vehicle routes...")

    cmd = [
        sys.executable,
        SUMO_TOOLS,
        "-n", NET_FILE,
        "-o", ROUTE_FILE,
        "--seed", "42",
        "-e", "3600",
        "--period", "2",
        "--fringe-factor", "5"
    ]

    subprocess.run(cmd, check=True)

    if os.path.exists(ROUTE_FILE):
        print("✅ Route file generated:", ROUTE_FILE)
    else:
        print("❌ Route generation failed")

if __name__ == "__main__":
    main()
