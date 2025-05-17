from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class SAEConfig:
    # Core model parameters
    input_dim: int
    hidden_dim: int
    learning_rate: float
    epochs: int
    batch_size: int
    activation_type: str

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
    n_samples: int = 1000
