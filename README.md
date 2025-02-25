# Exploring Sparse Autoencoders for Mechanistic Interpretability

Part of my mechanistic interpretability research journey exploring how sparse autoencoders can help understand neural networks.

## Overview
This project implements and analyzes sparse autoencoders (SAEs) to study feature extraction and representation learning in neural networks. It includes training, visualization, and analysis tools.

## Features
- Custom SAE implementation with configurable architecture
- Interactive visualization dashboard
- Sparsity and activation analysis tools
- Integration with W&B for experiment tracking
- MNIST dataset testing ground

## Structure
```
exploring_saes/
├── models/           # Model implementations
├── training/         # Training logic
├── evaluation/       # Analysis tools
├── visualization/    # Visualization utilities
├── dashboard/        # Interactive Streamlit dashboard
└── config/          # Configuration management
```

## Setup
```bash
git clone https://github.com/ashioyajotham/exploring_saes.git
cd exploring_saes
pip install -r requirements.txt
```

## Usage
Train SAE:
```bash
python main.py --hidden-dim 256 --lr 0.001 --epochs 100
```

Launch visualization dashboard:
```bash
streamlit run dashboard/app.py
```

## Research Context
This project explores SAEs as tools for understanding neural networks by:
- Extracting interpretable features
- Analyzing sparsity patterns
- Visualizing learned representations
- Studying activation distributions

## Dependencies
- PyTorch
- Streamlit
- Weights & Biases
- NumPy