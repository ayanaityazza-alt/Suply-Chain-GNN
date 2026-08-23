# NEXUS Control Tower – Supply Chain GNN

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Multi-Relational GNN (RGCN) for supply chain demand forecasting with an interactive dashboard.**  
Achieves **R² = 0.7364** on the SCG dataset.

---

## 📖 Overview

Products in a supply chain are connected: they share warehouses, plants, and product categories. Our RGCN model captures these relationships using **4 edge types** (Storage, Plant, Group, SubGroup) and an **ensemble of 5 models**.

The **NEXUS Control Tower** dashboard provides:
- 📊 Real-time KPIs (Total Demand, Confidence, Products at Risk)
- 🧠 Explainability (why a prediction was made)
- ⚙️ What-If Simulator (test production changes)
- 🎯 Recommender Engine (Top 3 actions)

---

## 📂 Dataset Setup

The SCG dataset is **publicly available** on Zenodo:  
👉 [https://doi.org/10.5281/zenodo.13652826](https://doi.org/10.5281/zenodo.13652826)

**1. Download** the dataset from the link above.  
**2. Create** a folder called `trade_data` in this repository.  
**3. Place** these 9 CSV files inside `trade_data/`:

- `NodesIndex.csv`
- `Edges (Storage Location).csv`
- `Edges (Plant).csv`
- `Edges (Product Group).csv`
- `Edges (Product Sub-Group).csv`
- `Sales Order.csv`
- `Production.csv`
- `Factory Issue.csv`
- `Delivery To distributor.csv`

---

## ▶️ Quick Start (Run the Dashboard)

```bash
# 1. Clone (using your username)
git clone https://github.com/ayanaityazza-alt/Supply-Chain-GNN.git
cd Supply-Chain-GNN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
python DashBoard.py

# 4. Open browser
http://127.0.0.1:8000

⏱️ Runtime
First Run	Subsequent Runs
~20-40 min (trains 5 models)	< 1 min (loads cached models)
📊 Results
Model	R²
Hybrid GNN	0.6403
RGCN (Single)	0.7200
RGCN Ensemble	0.7364

📁 Project Structure
Supply-Chain-GNN/
├── DashBoard.py        # Main application
├── requirements.txt    # Dependencies
├── README.md           # This file
└── trade_data/         # ⚠️ Place CSV files here (empty folder)

📝 Citation
@article{wasi2026nexus,
  title={A Visual Analytics Framework for Supply Chain Demand Forecasting Using Multi-Relational Graph Neural Networks},
  author={Wasi, Azmine Toushik and [Supervisor Name]},
  journal={Malaysian Journal of Computing},
  year={2026}
}

License
MIT License. See the LICENSE file.
