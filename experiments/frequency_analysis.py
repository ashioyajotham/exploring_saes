import torch
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt

class FrequencyAnalyzer:
    def __init__(self, model):
        self.model = model
        self.frequency_history = []
    
    def update(self, activations: torch.Tensor):
        """Update neuron activation frequencies"""
        active = (activations > 0).float()
        self.frequency_history.append(active.mean(0).cpu())
    
    def analyze(self) -> Dict[str, Any]:
        frequencies = torch.stack(self.frequency_history)
        mean_freq = frequencies.mean(0)
        
        return {
            'mean_frequencies': mean_freq.numpy(),
            'high_freq_neurons': (mean_freq > 0.1).sum().item(),
            'low_freq_neurons': (mean_freq < 0.01).sum().item(),
            'freq_distribution': self._analyze_distribution(mean_freq),
            'temporal_stats': self._analyze_temporal_patterns(frequencies)
        }
    
    def _analyze_distribution(self, frequencies):
        return {
            'percentiles': np.percentile(frequencies, [25, 50, 75]).tolist(),
            'skewness': float(torch.mean((frequencies - frequencies.mean())**3)),
            'kurtosis': float(torch.mean((frequencies - frequencies.mean())**4))
        }
    
    def _analyze_temporal_patterns(self, frequencies):
        return {
            'stability': self._compute_stability(frequencies),
            'trend': self._compute_trend(frequencies)
        }
