# Crowd Zero Trust (CZT) - Credit Card Fraud Detection

A federated learning framework for credit card fraud detection using Nvidia Flare's framework via XGBoost and GNNs with zero-trust principles. This project demonstrates how to convert centralized credit card fraud detection into a federated ETL pipeline while maintaining data privacy and security.

## Project Overview

Crowd Zero Trust implements a federated learning approach for credit card fraud detection that:

- **Preserves Data Privacy**: Keeps sensitive transaction data distributed across multiple sites
- **Enables Collaborative Learning**: Allows multiple financial institutions to train fraud detection models without sharing raw data
- **Implements Zero-Trust Security**: Ensures secure communication and computation across federated sites
- **Provides Feature Engineering**: Combines rule-based feature enrichment with GNN-based feature encoding

## Architecture

The project follows a federated learning architecture with the following components:

1. **Data Preparation**: Synthetic transaction data generation for testing
2. **Feature Engineering**:
   - Rule-based feature enrichment
   - GNN-based feature encoding (optional)
3. **Federated ETL**: Distributed data processing across multiple sites
4. **Model Training**: Federated XGBoost with enhanced features

## Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd czt
```

### 2. Install Dependencies with uv

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 3. Activate the Virtual Environment

```bash
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows
```

### 4. Run the Main Notebook

```bash
jupyter lab main.ipynb
```

## Project Structure

```
czt/
├── main.ipynb                # Main notebook
├── pyproject.toml            # Project configuration and dependencies
├── notebooks/                # Step-by-step tutorial notebooks
│   ├── 1.1.prepare_data.ipynb
│   ├── 2.1.1.feature_enrichment.ipynb
│   ├── 2.1.2.pre_process.ipynb
│   ├── 2.2.1.graph_construct.ipynb
│   └── 2.2.2.gnn_train_encode.ipynb
├── src/                      # Source code for federated ETL
│   ├── enrich.py             # Feature enrichment script
│   ├── pre_process.py        # Data preprocessing script
│   ├── graph_construct.py    # Graph construction script
│   └── gnn_train_encode.py   # GNN training and encoding script
├── utils/                    # Utility functions
├── xgb_data_loader.py        # XGBoost data loader
└── xgb_embed_data_loader.py  # XGBoost data loader with embeddings
```

## Key Features

### Federated ETL Pipeline

The project implements a complete federated ETL (Extract, Transform, Load) pipeline:

1. **Feature Enrichment**: Adds derived features like transaction statistics per currency
2. **Data Preprocessing**: Normalizes and prepares data for model training
3. **Graph Construction**: Builds transaction graphs for GNN-based feature encoding
4. **GNN Training**: Trains graph neural networks in a federated manner

### Zero-Trust Security

- Secure communication between federated sites
- No raw data sharing between participants
- Encrypted model updates and gradients
- Authentication and authorization at every step

### Model Architecture

- **XGBoost**: Primary fraud detection model
- **Graph Neural Networks**: Feature encoding and representation learning
- **Ensemble Methods**: Combines multiple model outputs for better performance

## Usage Guide

Everything is done within `main.ipynb`. The files within `notebooks` are performing each individual step on 1 single node, whereas `main.ipynb` uses NVIDIA Flare to distribute work to each node (in a multi-node architecture).


### Step 1: Data Preparation

Start with the data preparation notebook to generate synthetic transaction data:

```bash
jupyter lab notebooks/1.1.prepare_data.ipynb
```

### Step 2: Feature Engineering

Explore feature enrichment and preprocessing:

```bash
jupyter lab notebooks/2.1.1.feature_enrichment.ipynb
jupyter lab notebooks/2.1.2.pre_process.ipynb
```

### Step 3: Graph Neural Networks

For advanced feature encoding using GNNs:

```bash
jupyter lab notebooks/2.2.1.graph_construct.ipynb
jupyter lab notebooks/2.2.2.gnn_train_encode.ipynb
```

