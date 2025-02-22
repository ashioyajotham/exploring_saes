from dataclasses import dataclass
from typing import Optional

@dataclass
class SAEConfig:
    input_dim: int
    hidden_dim: int
    learning_rate: float = 0.001
    l1_coefficient: float = 0.001
    sparsity_param: float = 0.1
    batch_size: int = 64
    epochs: int = 100
    tied_weights: bool = False
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
