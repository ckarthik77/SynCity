# 🚦 SynCity: Synchronized Urban Traffic via AV-Infrastructure Synergy

**SynCity** is a **smart city traffic simulation and dataset generation framework** built on **SUMO** and Python.
It produces **novel traffic datasets** by simulating vehicle flows, logging real-time telemetry, and comparing results with real-world data sources (e.g., Google traffic trends).

The project integrates **simulation, dataset creation, visualization, and presentation** into a unified workflow — aimed at both **research publication** and **practical smart city applications**.

---

## 📂 Repository Structure

```
.
├── Test/                               # SUMO simulation setup
│   ├── test.net.xml                    # Road network (junctions, signals, lanes)
│   ├── test.rou.xml                    # Vehicle routes & flows
│   ├── test.sumocfg                    # SUMO configuration file
│
├── process/                            # Data processing & presentation assets
│   ├── Presentation/                   # Presentation-ready outputs
│   │   ├── Figure_1.png                # Research figure (pipeline/diagram)
│   │   ├── google_vs_synCity.gif       # Comparative traffic flow (SynCity vs Google)
│   │   ├── google_vs_synCity.colored.gif
│   │   ├── synCity_traffic_demo.gif    # Demo animation of traffic simulation
│   │
│   ├── google_vs_synCity.py            # Benchmark SynCity vs real-world (Google/Uber)
│   ├── synCity_animation.py            # Animate vehicle movement across the network
│   ├── synCity_data.csv                # Generated dataset (CSV format)
│   ├── synCity_realtime_dataset.xlsx   # Real-time structured dataset
│   ├── synCity_simulator.py            # Simulation runner + TraCI logging
│   ├── synCity_visuals.py              # Data visualization (plots, graphs)
│
├── README.md                           # Project documentation
```

---

## ✅ Current Progress

### 🔧 Simulation (SUMO)

* Configured **urban road network** (`test.net.xml`) with intersections, priority rules, and a traffic-light-controlled junction.
* Defined **vehicle trips and flows** (`test.rou.xml`) injecting up to 500 vehicles/hour.
* Configured **simulation settings** (`test.sumocfg`) for GUI and headless runs.

### 📊 Dataset Generation

* Logged **vehicle telemetry**:

  ```
  time, vehicle_id, x, y, speed, signal_state, congestion_level, hazard
  ```
* Exported datasets:

  * `synCity_data.csv` (raw log)
  * `synCity_realtime_dataset.xlsx` (structured dataset for ML analysis)

### 🎨 Visualization & Animation

* `synCity_visuals.py`: Plots traffic density, congestion levels, and signal behavior.
* `synCity_animation.py`: Animates vehicle movement across the network.
* `synCity_traffic_demo.gif`: Animated demo of traffic flow.

### 🔬 Comparative Analysis

* `google_vs_synCity.py`: Benchmarks SynCity-generated data against **Google traffic trends**.
* Output GIFs (`google_vs_synCity.gif`, `google_vs_synCity_colored.gif`) show real vs simulated traffic patterns.

### 📑 Presentation Assets

* Research figure (`Figure_1.png`) explaining pipeline.
* GIFs and visuals integrated for conference/journal presentations.

---

## ⚙️ How to Run

### Run SUMO Simulation

```bash
sumo-gui -c Test/test.sumocfg
```

or headless mode:

```bash
sumo -c Test/test.sumocfg
```

### Run Python Scripts

* **Generate dataset (CSV/Excel):**

  ```bash
  python synCity_simulator.py
  ```
* **Visualize traffic data:**

  ```bash
  python synCity_visuals.py
  ```
* **Animate traffic flow:**

  ```bash
  python synCity_animation.py
  ```
* **Compare with Google traffic trends:**

  ```bash
  python google_vs_synCity.py
  ```

---

## 🌍 Research Applications

* **Dynamic Signal Control** – optimize timing based on congestion.
* **Predictive Routing** – machine learning models for route forecasting.
* **Traffic Violation Detection** – overtaking, wrong-way driving, lane misuse.
* **Smart City Dashboards** – real-time IoT-driven traffic visualization.
* **Autonomous Vehicles (AV) – Infrastructure Synergy** – enabling intelligent V2I systems.

---

## 🔮 Future Work

* Expand simulation to **city-scale road networks**.
* Integrate **India-specific real-world datasets** (IoT sensors, OpenTraffic, Uber Movement).
* Enhance **driver behavior modeling** (lane changes, overtaking, violations).
* Apply **deep learning** for:

  * Congestion prediction
  * Traffic anomaly detection
  * Violation classification
* Build a **live visualization & alerting dashboard** for smart city traffic management.
* Prepare for **journal/conference submission** with novel dataset contributions.

---

## ✨ Authors & Credits

* **Lead Developer & Researcher**: Ch. Karthikeya
* **Frameworks**: [SUMO](https://www.eclipse.org/sumo/), Python (TraCI, matplotlib, pandas)
* **Vision**: To create a **novel Indian traffic dataset** combining **simulation + real-world data** for next-gen urban mobility research.

---

🔥 SynCity is not just a simulation — it’s a **research platform** to bridge the gap between **synthetic simulation data** and **real-world urban traffic intelligence**.

---
