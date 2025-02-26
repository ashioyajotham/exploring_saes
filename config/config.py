from dataclasses import dataclass
from typing import Optional

@dataclass
class SAEConfig:
    input_dim: int
    hidden_dim: int
    learning_rate: float
    epochs: int
    batch_size: int
    activation_type: str
    sparsity_param: float = 0.1
    k: int = 5  # For TopK activation
    use_wandb: bool = False
    
    # Transformer specific parameters
    model_name: str = 'gpt2-small'
    layer: int = 0
    n_samples: int = 1000
