# 🚦 **SynCity**  
**Synchronized Urban Traffic via AV-Infrastructure Synergy**



## 📂 Repository Structure

```
SynCity/
│
├── docs/                     # Documentation, diagrams, pitch deck
├── datasets/                 # Collected CSV/JSON traffic data
├── simulator/                # SUMO/CARLA configs, road networks, routes
│   ├── sumo/
│   └── carla/
├── controller/               # Cloud traffic management logic
│   ├── api/                  # Flask/FastAPI endpoints for V2I comms
│   ├── rules_engine/         # Dynamic signal control, routing logic
│   └── models/               # ML models for congestion prediction
├── dashboard/                # React.js or Dash visualization UI
├── scripts/                  # Utility scripts (data logging, dataset cleaning)
├── tests/                    # Unit/integration tests
├── requirements.txt          # Python dependencies
├── package.json              # Frontend dependencies
├── README.md                 # Project overview, setup guide
└── LICENSE                   # License file
```

---

## ⚙️ CI/CD Pipeline – GitHub Actions

**Workflow File:** `.github/workflows/syncity.yml`

```yaml
name: SynCity CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install Backend Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run Backend Tests
      run: |
        pytest tests/

    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'

    - name: Install Frontend Dependencies
      run: |
        cd dashboard
        npm install

    - name: Build Frontend
      run: |
        cd dashboard
        npm run build

    - name: Archive Production Build
      uses: actions/upload-artifact@v3
      with:
        name: dashboard-build
        path: dashboard/build
```


## 🤝 Contribution Workflow

1. **Fork & Clone** the repository  
2. Create a feature branch:  
   `git checkout -b feature/<feature-name>`  
3. Add your code and commit with clear messages  
4. Push the branch and open a Pull Request  
5. GitHub Actions will automatically run tests  
6. Maintainers review and merge

---

## 📘 README Structure

- **Project Title + Logo**
- **Abstract / Elevator Pitch**
- **Key Features**
- **System Architecture Diagram**
- **Tech Stack**
- **Setup Instructions**
- **Usage Guide**  
  (Run simulation, launch controller, start dashboard)
- **Demo Screenshots / GIFs**
- **Contribution Guidelines**
- **License**

---

## ⚔️ Google Maps vs SynCity

| Feature/Aspect              | Google Maps                                         | SynCity                                                  |
|----------------------------|-----------------------------------------------------|----------------------------------------------------------|
| **Data Source**            | Historical + crowd-sourced GPS                     | Real-time AV + Smart Infrastructure                      |
| **Traffic Updates**        | Estimated, delayed                                 | Live, predictive, proactive                              |
| **Routing Engine**         | Static routing with congestion avoidance           | Adaptive routing via V2I + ML congestion prediction      |
| **Vehicle Communication**  | None                                               | Bidirectional vehicle ↔ infrastructure                   |
| **Signal Optimization**    | Not involved                                       | Dynamic signal control                                   |
| **Intersection Management**| Passive (driver decision)                          | Cooperative slot-based AV crossing                       |
| **Environmental Impact**   | Neutral                                            | Minimizes emissions and idle time                        |
| **City-Level Integration** | Standalone app                                     | Integrated smart city ecosystem                          |
| **AV Support**             | Not designed for AV-native systems                 | AV-first, V2I-native architecture                        |

---

