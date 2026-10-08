# AirFlow AI: Smart Air Quality & Energy Management System

## 📌 Project Overview

Conventional HVAC and air purification systems rely on static, rule-based thresholds (such as simple ON/OFF timers or fixed setpoints). These traditional systems consume high amounts of electricity and cannot adapt to time-of-use (TOU) pricing structures or sudden spikes in air pollution.

**AirFlow AI** introduces a dynamic, multi-variable control environment that continuously evaluates indoor environmental metrics alongside variable energy rates. Built on top of a custom **Gymnasium** environment (`SmartAirEnv`) and paired with an interactive **Streamlit** dashboard, the system balances energy efficiency with indoor comfort and safety.

---

## 🏗️ System Architecture & Formulation

The indoor environmental dynamics and energy grid pricing are modeled as a **Markov Decision Process (MDP)**:

* **State Space ($\mathcal{S}$):** 4D continuous state vector $\mathbf{s}_t = [\text{AQI}_t, \text{Temp}_t, \text{Humidity}_t, \text{Tariff}_t] \in \mathbb{R}^4$.
* **Action Space ($\mathcal{A}$):** Discrete control modes:
  $$\mathcal{A} = \{0: \text{OFF}, 1: \text{ECO}, 2: \text{MED}, 3: \text{TURBO}\}$$
* **Multi-Objective Reward Function:**
  $$R_t = -\Big[ \alpha_1 \cdot \max(0, \text{AQI}_t - 50)^2 + \alpha_2 \cdot \text{ThermalPenalty}_t + \alpha_3 \cdot \text{EnergyCost}_t \Big]$$

---

## 🚀 Key Features

* **Custom Gymnasium Environment (`air_quality_env.py`):** Simulates real-world thermal exchange, pollutant buildup/dissipation, and variable electricity tariffs.
* **Interactive Web Dashboard (`app.py`):** Live Streamlit interface with Plotly data visualization to run simulations, tweak environmental parameters, and monitor control metrics.
* **Proactive Load Shifting:** Automatically curtails energy usage during peak-rate hours while preserving healthy Air Quality Index ($\text{AQI} < 50$) levels.

---

## 📂 Repository Structure

```text
AirFlow-AI/
├── air_quality_env.py    # Custom Gymnasium RL environment (SmartAirEnv)
├── app.py                # Streamlit UI & interactive dashboard
├── requirements.txt      # Project dependencies for local & cloud execution
└── README.md             # Project documentation
