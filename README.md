# Exploring Sparse Autoencoders for Mechanistic Interpretability

## Overview
Research project investigating how Sparse Autoencoders (SAEs) learn and represent features from transformer models. Focuses on understanding concept emergence, activation patterns, and neuron specialization.

## Key Research Questions
- How do different activation functions affect feature learning in SAEs?
- What drives concept emergence in hidden layers when training on transformer activations?
- How reliable are activation frequency patterns as indicators of neuron specialization?
- Can we quantify and visualize neuron behavior during training?

## Project Architecture
```
exploring_saes/
├── models/
│   └── autoencoder.py      # SAE implementation
├── experiments/
│   ├── activation_study.py # Activation function analysis
│   ├── frequency_analysis.py # Neuron firing patterns
│   ├── concept_emergence.py # Feature learning analysis
│   └── transformer_data.py  # Transformer integration
├── visualization/
│   ├── ascii_viz.py        # Terminal visualizations
│   └── wandb_viz.py        # W&B dashboard integration
└── config/
    └── config.py           # Experiment configuration
```

## Features
- **Model Architecture**
  - Configurable hidden dimensions
  - Multiple activation functions (ReLU, JumpReLU, TopK)
  - Sparsity regularization
  - Transformer activation processing

- **Analysis Tools**
  - Activation pattern tracking
  - Feature clustering
  - Concept emergence detection
  - Neuron frequency analysis
  - Sparsity measurements

- **Visualization**
  - Real-time ASCII training metrics
  - W&B experiment tracking
  - Feature map visualization
  - Activation heatmaps
  - Concept embedding plots

## Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage
Basic training:
```bash
python run_experiments.py --hidden-dim 256 --lr 0.001 --epochs 100
```

Transformer analysis:
```bash
python run_experiments.py --model-name gpt2-small --layer 0 --n-samples 1000 --use-wandb
```

Configuration options:
- `--hidden-dim`: Hidden layer size (default: 256)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Training epochs (default: 100)
- `--activation`: Activation function [relu|jump_relu|topk]
- `--model-name`: Transformer model
- `--layer`: Transformer layer to analyze
- `--n-samples`: Number of activation samples

## Results Visualization
1. **Terminal Output**
```
╔════════════════════ EXPERIMENT RESULTS ════════════════════╗
║ Activation Function Comparison:
║   ReLU     - Loss: 0.9870, Sparsity: 27.87%
║   JumpReLU - Loss: 0.9700, Sparsity: 30.15%
║   TopK     - Loss: 1.0427, Sparsity: 98.05%
╚═══════════════════════════════════════════════════════════╝
```

2. **W&B Dashboard**
- Training metrics
- Activation patterns
- Feature maps
- Concept embeddings

## Dependencies
- PyTorch >= 1.9.0
- transformer-lens >= 1.0.0
- Weights & Biases >= 0.19.7
- scikit-learn >= 1.0.2
- PyQt5 >= 5.15.0

## Research Methodology
1. **Data Collection**
   - Extract activations from transformer layers
   - Process and normalize activation patterns
   - Cache for efficient training

2. **Training Process**
   - Compare activation functions
   - Track neuron behavior
   - Measure feature emergence

3. **Analysis Pipeline**
   - Frequency pattern analysis
   - Concept clustering
   - Sparsity evaluation

## References
[1] ["Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning), Anthropic (2024)

[2] ["Sparse Autoencoders Find Highly Interpretable Features in Language Models"](https://arxiv.org/abs/2309.08600), Lee et al. (2023)