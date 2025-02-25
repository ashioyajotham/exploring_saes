import torch
import numpy as np
from typing import Dict, Any
import wandb
from pathlib import Path

def run_activation_comparison(config, train_model_fn):
    """Compare different activation functions"""
    if config.use_wandb:
        wandb.init(
            project="sae-interpretability",
            name=f"activation_comparison_{config.hidden_dim}",
            config=vars(config)
        )
    
    results = {}
    for activation in ['relu', 'jump_relu', 'topk']:
        config.activation_type = activation
        model = train_model_fn(config)
        results[activation] = analyze_model(model)
        
        if config.use_wandb:
            wandb.log({f"{activation}_stats": results[activation]})
    
    return results

def analyze_model(model) -> Dict[str, Any]:
    """Analyze model characteristics"""
    with torch.no_grad():
        stats = {
            'sparsity': model.activation_stats['sparsity'],
            'mean_activation': model.activation_stats['mean'],
            'activation_pattern': _analyze_activation_pattern(model),
            'weight_stats': _analyze_weights(model)
        }
    return stats

def _analyze_activation_pattern(model):
    return {
        'dead_neurons': (model.activation_stats['frequency'] == 0).sum().item(),
        'hyperactive_neurons': (model.activation_stats['frequency'] > 0.1).sum().item(),
        'avg_frequency': model.activation_stats['frequency'].mean().item()
    }

def _analyze_weights(model):
    return {
        'encoder_norm': torch.norm(model.encoder.weight).item(),
        'decoder_norm': torch.norm(model.decoder.weight).item(),
        'weight_correlation': _compute_weight_correlation(model)
    }

def _compute_weight_correlation(model):
    weights = model.encoder.weight
    corr_matrix = torch.corrcoef(weights)
    return torch.triu(corr_matrix, diagonal=1).mean().item()
