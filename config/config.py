from dataclasses import dataclass
from typing import Optional

@dataclass
class SAEConfig:
    # Existing parameters
    input_dim: int
    hidden_dim: int
    learning_rate: float = 0.001
    
    # Analysis parameters
    activation_type: str = 'relu'
    frequency_threshold: float = 0.1
    concept_analysis_interval: int = 10
    feature_clustering: bool = True

    l1_coefficient: float = 0.001
    sparsity_param: float = 0.1
    k: int = 5  # for TopK activation
    batch_size: int = 64
    epochs: int = 100
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
