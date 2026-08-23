# Suply-Chain-GNN
# NEXUS Control Tower – Supply Chain Demand Forecasting

**Multi-Relational Graph Neural Network (RGCN) for end-to-end supply chain demand forecasting with an interactive decision-support dashboard.**

This repository contains the official implementation for the paper:  
*"A Visual Analytics Framework for Supply Chain Demand Forecasting Using Multi-Relational Graph Neural Networks"*.

---

## 🚀 Overview

Supply chain products are highly interconnected. They share warehouses, production plants, and product categories. Traditional forecasting treats products independently and misses these dependencies.

We build a **Multi-Relational Graph Convolutional Network (RGCN)** to capture these relationships. The model uses 4 edge types (Storage, Plant, Product Group, Sub-Group) and an ensemble of 5 models to achieve **R² = 0.7364** on the SCG benchmark dataset.

The system is wrapped in the **NEXUS Control Tower** – a professional interactive dashboard that empowers managers with:

- 📈 **Real-time KPIs**: Total Demand, Average Confidence, Products at Risk, Total Inventory.
- 🧠 **Explainability (XAI)**: Visual breakdown of why a prediction was made (Production, Factory Issue, Delivery, Sales History).
- ⚙️ **What-If Simulator**: Adjust production volumes and instantly see the cascading effect across all 28 products.
- 🎯 **Intelligent Recommender**: Get Top-3 actionable actions (Top Seller Boost, Growth Capturer, Risk Mitigator).
- 📊 **Network View**: Click on any product tag to analyze its specific demand profile.

---

## 🏗️ Architecture

| Component | Specification |
| :--- | :--- |
| **Model** | Relational Graph Convolutional Network (RGCN) |
| **Graph Nodes** | 28 Active Products |
| **Edge Types** | 4 (Storage Location, Plant, Product Group, Product Sub-Group) + Self-loops |
| **Input Features** | 27 engineered features/day (Rolling stats, Lags, Day-of-week) |
| **Temporal Window** | 14 days (378 input channels per product) |
| **Hidden Units** | 256 |
| **Ensemble** | 5 random seeds (42, 123, 456, 789, 1011) |
| **Best R²** | **0.7364** |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Supply-Chain-GNN.git
cd Supply-Chain-GNN
