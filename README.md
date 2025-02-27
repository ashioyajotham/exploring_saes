# Exploring Sparse Autoencoders for Mechanistic Interpretability

## Overview
Research project investigating how Sparse Autoencoders (SAEs) learn and represent features from transformer models. Focuses on understanding concept emergence, activation patterns, and neuron specialization through multiple activation functions and comprehensive analysis tools.

## Key Research Questions
- How do different activation functions affect feature learning in SAEs?
- What drives concept emergence in hidden layers when training on transformer activations?
- How reliable are activation frequency patterns as indicators of neuron specialization?
- Can we quantify and visualize neuron behavior during training?
- How do different sparsity mechanisms impact feature interpretability?

## Installation

### Requirements
- Python 3.8+
- CUDA capable GPU (recommended)
- 8GB+ RAM

### Setup
```bash
# Create virtual environment
python -m venv venv

# Activate environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install additional visualization tools
pip install umap-learn wandb
```

## Project Structure
```
exploring_saes/
├── experiments/
│   ├── activation_study.py   # Activation function analysis
│   ├── concept_emergence.py  # Feature learning tracking
│   ├── frequency_analysis.py # Neuron firing patterns
│   ├── checkpointing.py     # Experiment state management
│   └── transformer_data.py   # Model integration
├── models/
│   └── autoencoder.py       # SAE implementation
├── visualization/
│   ├── ascii_viz.py         # Terminal visualizations
│   └── wandb_viz.py         # W&B dashboard integration
├── config/
│   └── config.py            # Configuration management
└── run_experiments.py       # Main entry point
```

## Usage

### Basic Training
```bash
python run_experiments.py --hidden-dim 256 --epochs 100
```

### Transformer Analysis
```bash
python run_experiments.py --model-name gpt2-small --layer 0 --n-samples 1000 --use-wandb
```

### Configuration Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| --hidden-dim | Hidden layer size | 256 |
| --lr | Learning rate | 0.001 |
| --epochs | Training epochs | 100 |
| --batch-size | Batch size | 64 |
| --activation | Activation type [relu/jump_relu/topk] | relu |
| --model-name | Transformer model | gpt2-small |
| --layer | Layer to analyze | 0 |
| --n-samples | Number of samples | 1000 |
| --use-wandb | Enable W&B logging | False |

## Features

### Analysis Tools
- Multiple activation function comparison
- Neuron frequency analysis
- Concept emergence tracking
- Feature clustering
- Attribution scoring
- Sparsity measurements

### Visualization
- Real-time ASCII training metrics
- W&B experiment tracking
- Feature map visualization
- Activation heatmaps
- Concept embedding plots

### Checkpointing
Automatic experiment state saving enables:
- Recovery from interruptions
- Continuation of training
- Progress tracking
- Result caching

## Results Visualization

### Terminal Output
```
╔════════════════════ EXPERIMENT RESULTS ════════════════════╗
║ Activation Function Comparison:
║   ReLU     - Loss: 0.9870, Sparsity: 27.87%
║   JumpReLU - Loss: 0.9700, Sparsity: 30.15%
║   TopK     - Loss: 1.0427, Sparsity: 98.05%
╚═══════════════════════════════════════════════════════════╝
```

### W&B Dashboard
Access experiment tracking at: https://wandb.ai/ashioyajotham/sae-interpretability

Features:
- Loss curves
- Activation patterns
- Feature maps
- Concept embeddings
- Neuron statistics

## Error Recovery
Training can be resumed using checkpoints:
```bash
# Training will continue from last successful state
python run_experiments.py [previous-args] --resume
```

## References
[1] ["Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning), Anthropic (2024)

[2] ["Sparse Autoencoders Find Highly Interpretable Features in Language Models"](https://arxiv.org/abs/2309.08600), Lee et al. (2023)

[3] ["Scaling and evaluating sparse autoencoders"](https://cdn.openai.com/papers/sparse-autoencoders.pdf), OpenAI (2022)