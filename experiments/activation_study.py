"""
Activation Function Analysis Module
=================================

Compares different activation functions for Sparse Autoencoders.
Analyzes performance, sparsity patterns, and feature learning characteristics.

Key Components:
- ReLU vs JumpReLU vs TopK comparison
- Sparsity pattern analysis
- Feature emergence tracking
- Performance metrics calculation

Usage:
------
results = run_activation_comparison(config, train_model_fn)
"""

import torch
import numpy as np
from typing import Dict, Any, Callable
import wandb
from pathlib import Path

class ActivationStudy:
    """
    Analyzes activation function behavior in SAEs.
    
    Attributes:
        config: Experiment configuration
        train_fn: Model training function
        results: Comparison results
    """
    
    def __init__(self, config, train_model_fn: Callable):
        self.config = config
        self.train_fn = train_model_fn
        self.results = {}

    def run_comparison(self) -> Dict[str, Any]:
        """
        Compare different activation functions.
        
        Tests:
        - Training convergence
        - Sparsity patterns
        - Feature interpretability
        
        Returns:
            Dictionary with comparison metrics
        """
        for activation in ['relu', 'jump_relu', 'topk']:
            self.config.activation_type = activation
            model, freq_analyzer, losses = self.train_fn(self.config)
            
            self.results[activation] = {
                'final_loss': losses[-1],
                'loss_trend': losses,
                'sparsity': self._compute_sparsity(model),
                'feature_stats': self._analyze_features(model)
            }
            
        return self.results

    def _compute_sparsity(self, model) -> float:
        """Calculate activation sparsity."""
        with torch.no_grad():
            test_input = torch.randn(100, model.input_dim)
            _, encoded = model(test_input)
            return (encoded.abs() < 1e-5).float().mean().item()

    def _analyze_features(self, model) -> Dict[str, float]:
        """Analyze learned feature characteristics."""
        weights = model.encoder.weight.data
        return {
            'mean_magnitude': weights.abs().mean().item(),
            'std_magnitude': weights.abs().std().item(),
            'orthogonality': self._compute_orthogonality(weights)
        }

    def _compute_orthogonality(self, weights) -> float:
        """Measure feature orthogonality."""
        norm_weights = F.normalize(weights, dim=1)
        gram = torch.mm(norm_weights, norm_weights.t())
        off_diag = gram - torch.eye(gram.shape[0], device=gram.device)
        return off_diag.abs().mean().item()

def run_activation_comparison(config, train_model_fn):
    """
    Entry point for activation function comparison.
    
    Args:
        config: Experiment configuration
        train_model_fn: Function to train model
        
    Returns:
        Dictionary with comparison results
    """
    study = ActivationStudy(config, train_model_fn)
    results = study.run_comparison()
    
    if config.use_wandb:
        wandb.init(
            project="sae-interpretability",
            name=f"activation_comparison_{config.hidden_dim}",
            config=vars(config)
        )
        
        for act_type, metrics in results.items():
            wandb.log({
                f"{act_type}/loss": metrics['final_loss'],
                f"{act_type}/sparsity": metrics['sparsity'],
                f"{act_type}/feature_stats": metrics['feature_stats']
            })
            
    return results
