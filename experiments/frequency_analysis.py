"""
Frequency Analysis Module
========================

Analyzes neuron firing patterns and activation frequencies in Sparse Autoencoders.
Tracks how often neurons activate and identifies high/low frequency patterns.

Key Components:
- Activation frequency tracking
- High-frequency neuron detection
- Temporal pattern analysis
- Distribution statistics

References:
[1] "Understanding Sparsity in Autoencoders", Ng et al.
[2] "Feature Selectivity in Neural Networks", Zhou et al.
"""

import torch
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt

class FrequencyAnalyzer:
    """
    Tracks and analyzes neuron activation frequencies.
    
    Attributes:
        model: SAE model being analyzed
        activation_history: Raw activation values
        frequency_history: Binary activation frequencies
        activation_type: Type of activation function
    """
    
    def __init__(self, model):
        """
        Initialize frequency analyzer.
        
        Args:
            model: Trained SAE model to analyze
        """
        self.model = model
        self.activation_history = []
        self.frequency_history = []
        self.activation_type = model.activation_type
    
    def update(self, activations: torch.Tensor):
        """
        Update neuron activation frequencies.
        
        Args:
            activations: Current batch activations
        """
        active = (activations > 0).float()
        self.frequency_history.append(active.mean(0).cpu())
        self.activation_history.append(activations.detach().cpu())
        
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze activation patterns.
        
        Returns:
            Dictionary containing:
            - high_freq_neurons: Count of frequently firing neurons
            - mean_frequencies: Average activation rates
            - freq_distribution: Statistical distribution
            - temporal_stats: Time-based patterns
        """
        if len(self.frequency_history) < 2:
            return self._empty_analysis()
            
        frequencies = torch.stack(self.frequency_history)
        mean_freq = frequencies.mean(0)
        
        return {
            'high_freq_neurons': (mean_freq > 0.1).sum().item(),
            'mean_frequencies': mean_freq,
            'freq_distribution': self._compute_distribution(mean_freq),
            'temporal_stats': self._analyze_temporal_patterns(frequencies)
        }
        
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis for insufficient data."""
        return {
            'high_freq_neurons': 0,
            'mean_frequencies': torch.zeros(1),
            'freq_distribution': {'percentiles': [0, 0, 0]},
            'temporal_stats': {'stability': 0, 'trends': []}
        }
        
    def _compute_distribution(self, frequencies: torch.Tensor) -> Dict[str, List[float]]:
        """Calculate frequency distribution statistics."""
        return {
            'percentiles': torch.quantile(
                frequencies, 
                torch.tensor([0.25, 0.5, 0.75])
            ).tolist()
        }
        
    def _analyze_temporal_patterns(self, frequencies: torch.Tensor) -> Dict[str, Any]:
        """Analyze how activation patterns change over time."""
        stability = torch.corrcoef(frequencies.T)[0, 1].item()
        trends = torch.diff(frequencies.mean(1)).tolist()
        
        return {
            'stability': stability,
            'trends': trends
        }
