# Exploring Sparse Autoencoders for Mechanistic Interpretability

## Research Focus
This project investigates the fundamental behavior of Sparse Autoencoders (SAEs) for neural network interpretability research. Key research questions:

- How do different activation functions affect feature learning?
- What drives concept emergence in hidden layers?
- How reliable are activation frequency patterns?
- Can we quantify neuron specialization?

## Implementation & Analysis Tools
- Custom SAE with configurable architectures
- Multiple activation functions (ReLU, JumpReLU, TopK)
- Comprehensive neuron behavior analysis:
  - Activation frequency tracking
  - Temporal stability metrics
  - Feature clustering
  - Semantic structure analysis
  
## Analysis Pipeline
```python
# Core analysis components
activation_study     # Compare activation function impacts
frequency_analysis  # Track neuron firing patterns
concept_emergence   # Study feature learning dynamics
```

## Experimental Setup
- Architecture: Configurable hidden dimensions
- Dataset: MNIST (initial testbed)
- Training: Adam optimizer with L1 sparsity regularization
- Metrics tracked:
  - Reconstruction loss
  - Sparsity measurements
  - Neuron death rate
  - Feature correlation

## Visualization
- Real-time training dashboard
- Neuron activation patterns
- Feature space embeddings
- W&B experiment tracking

## Research Applications
This tooling supports mechanistic interpretability research:
- Feature extraction analysis
- Activation pattern studies
- Concept formation tracking
- Sparsity impact evaluation

## Usage
```bash
# Basic training run
python main.py --hidden-dim 256 --lr 0.001 --epochs 100

# With visualization
python main.py --hidden-dim 256 --use-wandb

# Activation function comparison
python main.py --activation topk --k 5
```

## Future Research Directions
- Extend to transformer architectures 
- Compare with other interpretability methods
- Investigate concept hierarchies
- Study transfer learning effects

## Dependencies
- PyTorch
- Weights & Biases
- scikit-learn
- PyQt5

## References
[1] Anthropic's SAE Research
[2] Mechanistic Interpretability Papers