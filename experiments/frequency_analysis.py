import torch
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt

class FrequencyAnalyzer:
    def __init__(self, model):
        self.model = model
        self.activation_history = []
        self.frequency_history = []
        self.activation_type = model.activation_type
    
    def update(self, activations: torch.Tensor):
        """Update neuron activation frequencies"""
        active = (activations > 0).float()
        self.frequency_history.append(active.mean(0).cpu())
        """Track activations during training"""
        self.activation_history.append(activations.detach().cpu())
    
    def analyze(self) -> Dict[str, Any]:
        if len(self.frequency_history) < 2:
            return {
                'high_freq_neurons': 0,
                'mean_frequencies': torch.zeros(1),
                'freq_distribution': {
                    'percentiles': [0, 0, 0]
                }
            }
    
        frequencies = torch.stack(self.frequency_history)
        mean_frequencies = frequencies.mean(0)  # Calculate mean frequencies
        
        return {
            'high_freq_neurons': (mean_frequencies > 0.1).sum().item(),
            'mean_frequencies': mean_frequencies,  # Return tensor directly
            'freq_distribution': {
                'percentiles': torch.quantile(mean_frequencies, torch.tensor([0.25, 0.5, 0.75])).tolist()
            }
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
    
    def _compute_stability(self, frequencies):
        """Compute stability of neuron activations over time"""
        # Compute variance over time for each neuron
        temporal_variance = torch.var(frequencies, dim=0)
        return {
            'mean_stability': float(1 - temporal_variance.mean()),
            'min_stability': float(1 - temporal_variance.max()),
            'unstable_neurons': int((temporal_variance > 0.1).sum())
        }

    def _compute_trend(self, frequencies):
        """Compute activation trends over time"""
        time_steps = torch.arange(len(frequencies))
        # Compute correlation with time for each neuron
        correlations = torch.tensor([
            torch.corrcoef(torch.stack([time_steps, freq]))[0,1]
            for freq in frequencies.T
        ])
        return {
            'increasing': int((correlations > 0.5).sum()),
            'decreasing': int((correlations < -0.5).sum()),
            'stable': int((correlations.abs() <= 0.5).sum())
        }
