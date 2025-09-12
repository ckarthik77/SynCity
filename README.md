🚦 SynCity: Synchronized Urban Traffic via AV-Infrastructure Synergy

SynCity is a smart urban traffic simulation and dataset generation framework. It integrates SUMO (Simulation of Urban Mobility) with Python-based data processing, visualization, and real-time analytics to model intelligent traffic management systems.

This repository contains:

Traffic simulation setup (network, routes, configs)

Dataset generation and logging (CSV/Excel)

Visualizations & animations

Comparative analysis with real-world datasets

Presentation material for research dissemination

📂 Repository Structure
...
.
├── Test/                              # SUMO configuration files
│   ├── test.net.xml                   # Road network (junctions, signals, roads)
│   ├── test.rou.xml                   # Vehicle trips & flows
│   ├── test.sumocfg                   # Simulation configuration
│
├── process/                           # Data processing (future scripts go here)
│
├── Presentation/                      # Presentation & research dissemination
│   ├── google_vs_synCity.py           # Google traffic vs SynCity dataset comparison
│   ├── synCity_animation.py           # Traffic animation & visualization
│   ├── synCity_data.csv               # Generated dataset (CSV format)
│   ├── synCity_realtime_dataset.xlsx  # Real-time structured dataset
│   ├── synCity_simulator.py           # Main simulation runner (TraCI integration)
│   ├── synCity_visuals.py             # Data visualization utilities
│
├── README.md                          # Project documentation

...

✅ Current Progress

SUMO Setup

Network (test.net.xml) with multiple intersections and traffic lights.

Vehicle flows (test.rou.xml) with defined trips and continuous inflows.

Config (test.sumocfg) to run simulation.

Dataset Generation

Real-time vehicle logging (CSV & Excel).

Attributes: time, vehicle_id, x, y, speed, signal_state, congestion_level, hazard.

SynCity dataset stored in synCity_data.csv and synCity_realtime_dataset.xlsx.

Visualization

synCity_visuals.py: Plots traffic density, flow over time, and signal interaction.

synCity_animation.py: Animates vehicle movement across the network.

Comparative Analysis

google_vs_synCity.py: Benchmarks SynCity dataset against real-world traffic data (Google/Uber).

Presentation Ready

Scripts and structured datasets for research presentations and demonstrations.

⚙️ How to Run
Run SUMO Simulation
sumo-gui -c Test/test.sumocfg


or headless mode:

sumo -c Test/test.sumocfg

Run Python Scripts

Generate dataset (CSV/Excel):

python synCity_simulator.py


Visualize results:

python synCity_visuals.py


Animate traffic flow:

python synCity_animation.py


Compare with Google/Uber data:

python google_vs_synCity.py

🌍 Research Applications

Dynamic traffic signal optimization

Predictive routing & congestion forecasting

Violation detection (wrong-way driving, overtaking, etc.)

Real-time urban traffic dashboards

Autonomous Vehicle (AV) – Infrastructure synergy

🔮 Future Work

Expand road network to city-scale environments

Integrate real-world traffic datasets (India-specific, sensor/IoT feeds)

Develop deep learning models for congestion prediction & anomaly detection

Add driver behavior modeling (lane changing, overtaking, wrong-way detection)

Build a live data visualization dashboard for smart cities

Prepare journal paper submission with novel dataset contributions

✨ Authors & Credits

Lead Developer & Researcher: Ch. Karthikeya

Framework: SUMO
 + Python (TraCI, matplotlib, pandas)

Vision: Create a novel Indian traffic dataset for research in AI-driven urban mobility

🔥 With SynCity, we are building a bridge between simulation and real-world urban traffic management, paving the way for smarter, safer, and more efficient cities.
