# Exploring Sparse Autoencoders for Mechanistic Interpretability

## Overview

Research project investigating how Sparse Autoencoders (SAEs) learn and represent features from transformer models. Focuses on understanding concept emergence, activation patterns, and neuron specialization through multiple activation functions and comprehensive analysis tools.

## Methodology

### Decomposing the "Neural Soup" of GPT-2

Welcome to a from-scratch implementation of Sparse Autoencoders designed to interpret the internal activations of Large Language Models (specifically GPT-2 Small).

This project is an exploration into Mechanistic Interpretability—the science of reverse-engineering neural networks to understand not just what they do, but how they think.

### The Core Concepts

- **Sparse Autoencoders (SAEs)**: Neural networks that learn efficient representations of input data by enforcing sparsity in the hidden layers. Sparsity means that only a small fraction of neurons are active at any given time, which can lead to more interpretable features.
- **Activation Functions**: Different functions (ReLU, JumpReLU, TopK) are employed to enforce sparsity and study their effects on feature learning.
- **Concept Emergence**: Tracking how distinct features or "concepts" arise in the hidden layers during training.
- **Neuron Specialization**: Analyzing how individual neurons develop specific roles based on their activation patterns.

#### 1. The Problem: Polysemanticity ("The Soup")

- Large language models often exhibit polysemanticity, where single neurons respond to multiple unrelated features. This makes it challenging to interpret what each neuron represents. In a standard neural network, individual neurons are polysemantic (Many-Meanings) for instance, a neuron might activate for both "cat" and "satellite," making it hard to decipher its true function.
- This is because the model is compressing trillions of concepts into limited space, a concept known as ["Superposition."](https://transformer-circuits.pub/2022/toy_model/index.html)
- The result is that looking at raw neuron activations is like looking at a bowl of soup—you see a mix of ingredients but can't easily identify each one. The internal representations are entangled and hard to interpret.

#### 2. The Solution: Monosemanticity ("The Ingredients")

- We want to map the network to a state of Monosemanticity (One-Meaning), where each neuron corresponds to a single, distinct concept. This makes it easier to understand what each neuron is doing.
- We want to find specific features in the model's activations, such as "cat," "satellite," or "the concept of 'being in space,'" and have individual neurons represent these features clearly.

#### 3. The Tool: Sparse Autoencoders (SAEs)

- The SAE acts like a `Prism`.

  - It takes the "white light" of raw activations (the soup)
  - It expands it into a massive sparse dimension
  - It forces the data to separate into distinct, interpretable "rays" (the features)

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

### The Stack

- `transformer_data.py` (The Harvester): Hooks into a pre-trained transformer model (GPT-2 Small) using `Transformer_lens` to extract raw "resid_pre" activations from specific layers.
- `models/autoencoder.py` (The Prism): a custom Pytorch implementation of the SAE.

  - Encoders: supports standard `ReLU`, `JumpReLU`, and `TopK` activation functions to enforce sparsity.
  - Decoders: reconstructs the original activations from the sparse representations learned by the encoder (reconstructs the signal to measure fidelity).

- `experiments/` (The Laboratory): Contains scripts for various experiments analyzing activation functions, concept emergence, frequency patterns, and more. Scripts to run training loops, managing the trade-off between reconstruction loss (MSE Loss) and sparsity (L1 Loss). L1 loss is the loss resulting from the L1 norm of the activations, encouraging sparsity while MSE loss measures how well the SAE reconstructs the original input.

- `visualization/` (The Microscope): Tools for visualizing training progress and results, including ASCII terminal outputs and W&B dashboards.

```tree
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

```terminal
╔════════════════════ EXPERIMENT RESULTS ════════════════════╗
║ Activation Function Comparison:
║   ReLU     - Loss: 0.9870, Sparsity: 27.87%
║   JumpReLU - Loss: 0.9700, Sparsity: 30.15%
║   TopK     - Loss: 1.0427, Sparsity: 98.05%
╚═══════════════════════════════════════════════════════════╝
```

### W&B Dashboard

Access experiment tracking at: [https://wandb.ai/ashioyajotham/sae-interpretability](https://wandb.ai/ashioyajotham/sae-interpretability)

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

## Roadmap: From Understanding to Control

Currently, this project focuses on **Feature Discovery**(finding the dictionary)—identifying and understanding the features learned by SAEs. The next phase involves **Steering**—developing methods to manipulate and control these features within the model.
This could include techniques for targeted editing of neuron activations or guiding the model's behavior based on the learned features.

- Phase 1: Feature Discovery (Current)
  - Train SAEs to decompose transformer activations into sparse, interpretable features.
  - Identify and analyze features learned by SAEs.
  - Understand how different activation functions impact feature learning.

- Phase 2: Feature Identification (In Progress)
  - `find_feature.py` Identify specific features corresponding to concepts eg "The Golden Gate Bridge", "Quantum Mechanics", etc.

- Phase 3: Steering (Future Work)
  - `steer_model.py` "Clamp" these features to control model behavior, e.g., during inference to force the model to hallucinate specific concepts like discussing "Quantum Mechanics" or describing "The Golden Gate Bridge" irrespective of the prompt.

## References

[1] ["Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning), Anthropic (2023)

[2] ["Sparse Autoencoders Find Highly Interpretable Features in Language Models"](https://arxiv.org/abs/2309.08600), Lee et al. (2023)

[3] ["Scaling and evaluating sparse autoencoders"](https://cdn.openai.com/papers/sparse-autoencoders.pdf), OpenAI (2022)
