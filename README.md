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
# 1. Clone (replace with your username)
git clone https://github.com/YOUR_USERNAME/Supply-Chain-GNN.git
cd Supply-Chain-GNN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app_final.py

# 4. Open browser
http://127.0.0.1:8000
