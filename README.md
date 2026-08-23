
# Hybrid GNN for Supply Chain Demand Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3+-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![Dataset: SCG](https://img.shields.io/badge/Dataset-SCG-important.svg)](https://doi.org/10.5281/zenodo.13652826)

**A Multi-Relational Graph Neural Network for supply chain demand forecasting with an interactive Control Tower dashboard.**

Achieves **R² = 0.7364** on the SCG benchmark dataset.

---

## 📖 Overview

Products in a supply chain are connected. They share warehouses, production plants, and product categories. Our model captures these relationships using an **RGCN with 4 edge types** and an **ensemble of 5 models**, outperforming standard GNN architectures.

The **NEXUS Control Tower** dashboard provides:
- Real-time KPIs (Total Demand, Confidence, Products at Risk)
- Explainability (why a prediction was made)
- What-If Simulator (test production changes)
- Recommender Engine (Top 3 actions)

---

## 📊 Results

| Model | R² |
| :--- | :---: |
| Hybrid GNN (GATv2 + Transformer) | 0.6403 |
| RGCN (Single Seed) | 0.7200 |
| **RGCN Ensemble(The model implemented in DashBoard.py)** | **0.7364** |

---

## 📂 Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  221 days × 40 products × 4 features                           │
│  (Production, Factory Issue, Delivery, Sales Order)            │
│                                                                 │
│  ▼                                                              │
│  Remove 12 dead products (>80% zero sales)                     │
│                                                                 │
│  ▼                                                              │
│  221 days × 28 active products × 4 features                    │
│                                                                 │
│  ▼                                                              │
│  Feature Engineering (27 features/day):                        │
│  └─ Rolling stats (mean + std over 7 days)                     │
│  └─ Lags (day -7 and day -14)                                  │
│  └─ Day-of-week encoding (7 one-hot features)                  │
│                                                                 │
│  ▼                                                              │
│  14-day sliding window                                         │
│                                                                 │
│  ▼                                                              │
│  213 samples × 28 products × 378 input channels                │
│                                                                 │
│  ▼                                                              │
│  Temporal split: 80% train / 20% test (no shuffling)          │
│                                                                 │
│  ▼                                                              │
│  Train: 170 snapshots  |  Test: 43 snapshots                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Dataset source:** SCG dataset (Wasi et al., 2024)  
👉 [https://doi.org/10.5281/zenodo.13652826](https://doi.org/10.5281/zenodo.13652826)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Input: 28 products × 378 features (14 days × 27 features)    │
│                                                                 │
│  ▼                                                              │
│  [Input Projection] Linear(378 → 256)                          │
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────────────┐   │
│  │   RGCN Layer 1      │    │   RGCN Layer 1              │   │
│  │   (5 relations)     │    │   (5 relations)             │   │
│  └─────────────────────┘    └─────────────────────────────┘   │
│  │                           │                                  │
│  ▼                           ▼                                  │
│  [LayerNorm + Dropout]      [LayerNorm + Dropout]              │
│  │                           │                                  │
│  ▼                           ▼                                  │
│  ┌─────────────────────┐    ┌─────────────────────────────┐   │
│  │   RGCN Layer 2      │    │   RGCN Layer 2              │   │
│  │   (5 relations)     │    │   (5 relations)             │   │
│  └─────────────────────┘    └─────────────────────────────┘   │
│  │                           │                                  │
│  ▼                           ▼                                  │
│  [LayerNorm + Dropout]      [LayerNorm + Dropout]              │
│  │                           │                                  │
│  └─────────────┬─────────────┘                                  │
│                ▼                                                │
│  [Output MLP] Linear(256 → 64 → 1)                            │
│                                                                 │
│  ▼                                                              │
│  Prediction: Demand for each product                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

5-model ensemble with seeds: 42, 123, 456, 789, 1011
```

---

## 📂 Dataset Setup

The SCG dataset is **publicly available** on Zenodo:  
👉 [https://doi.org/10.5281/zenodo.13652826](https://doi.org/10.5281/zenodo.13652826)

**1. Download** the dataset from the link above.  
**2. Place** these 9 CSV files inside `trade_data/`:

```
trade_data/
├── NodesIndex.csv
├── Edges (Storage Location).csv
├── Edges (Plant).csv
├── Edges (Product Group).csv
├── Edges (Product Sub-Group).csv
├── Sales Order.csv
├── Production.csv
├── Factory Issue.csv
└── Delivery To distributor.csv
```

---

## ▶️ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ayanaityazza-alt/Supply-Chain-GNN.git
cd Supply-Chain-GNN

# 2. Create virtual environment (recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
python DashBoard.py

# 5. Open browser
http://127.0.0.1:8000
```

---

## ⏱️ Runtime

| First Run | Subsequent Runs |
| :--- | :--- |
| ~20-40 min (trains 5 models) | **< 1 min** (loads cached models) |

---

## 📁 Project Structure

```
Supply-Chain-GNN/
├── DashBoard.py        # Main application
├── requirements.txt    # Dependencies
├── README.md           # This file
├── LICENSE   
└── trade_data/         # Place CSV files here (empty folder)
```

---

## 📝 Citation

```bibtex
@article{wasi2026nexus,
  title={A Visual Analytics Framework for Supply Chain Demand Forecasting Using Multi-Relational Graph Neural Networks},
  author={Aya Nait Yazza, Ahmad Faiz Ghazali},
  journal={Malaysian Journal of Computing},
  year={2026}
}
```

**Dataset**:
```bibtex
@article{wasi2024graph,
  title={Graph Neural Networks in Supply Chain Analytics and Optimization: Concepts, Perspectives, Dataset and Benchmarks},
  author={Wasi, Azmine Toushik and Islam, MD Shafikul and Akib, Adipto Raihan and Bappy, Mahathir Mohammad},
  journal={arXiv preprint arXiv:2411.08550},
  year={2024}
}
```

---

## 📄 License

MIT License. See the [LICENSE](LICENSE) file.
