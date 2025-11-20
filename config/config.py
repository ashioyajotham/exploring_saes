from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class SAEConfig:
    # Core model parameters (centralized defaults)
    input_dim: int = 784
    hidden_dim: int = 256
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 64
    activation_type: str = 'relu'

    # Sparsity controls
    sparsity_param: float = 0.1
    k: int = 5  # For TopK activation
    min_activation: float = 1e-5
    sparsity_threshold: float = 0.1

    # Experiment type
    is_comparison: bool = False
    experiment_name: Optional[str] = None

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_freq: int = 10
    resume: bool = False
    
    # Visualization
    use_wandb: bool = False
    wandb_project: str = "sae-interpretability"
    log_freq: int = 1
    
    # Training controls
    early_stopping_patience: int = 10
    lr_schedule_factor: float = 0.5
    lr_patience: int = 5
    grad_clip: float = 1.0
    
    # Analysis parameters
    concept_similarity_threshold: float = 0.7
    high_freq_threshold: float = 0.1
    feature_cluster_n: int = 10
    
    # Transformer parameters
    model_name: str = 'gpt2-small'
    layer: int = 0
    n_samples: int = 10000
    seq_len: int = 20

    # Dead neuron resampling
    dead_neuron_threshold: float = 1e-3  # below this mean firing rate neuron is 'dead'
    dead_resample: bool = True
    dead_resample_window: int = 1000  # steps to consider before resampling
    dead_resample_strength: float = 1.0  # scaling for resampled weights

    # Normalization
    normalize_activations: bool = True
