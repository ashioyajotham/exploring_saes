from dataclasses import dataclass, asdict
import json
import torch
from pathlib import Path

@dataclass
class ExperimentState:
    completed_activations: list
    current_activation: str
    epoch: int
    model_state: dict
    optimizer_state: dict
    frequency_stats: dict
    losses: list

class CheckpointManager:
    def __init__(self, save_dir="checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def _tensor_to_list(self, obj):
        """Convert tensors to native Python types"""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().tolist()
        elif isinstance(obj, dict):
            return {k: self._tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._tensor_to_list(item) for item in obj]
        return obj

    def save_checkpoint(self, state: ExperimentState, config):
        checkpoint = {
            'config': asdict(config),
            'completed_activations': state.completed_activations,
            'current_activation': state.current_activation,
            'epoch': state.epoch,
            'frequency_stats': self._tensor_to_list(state.frequency_stats),
            'losses': self._tensor_to_list(state.losses)
        }
        
        # Save model and optimizer states separately
        torch.save({
            'model_state': state.model_state,
            'optimizer_state': state.optimizer_state
        }, self.save_dir / 'model_states.pt')
        
        # Save experiment state
        with open(self.save_dir / 'experiment_state.json', 'w') as f:
            json.dump(checkpoint, f)
    
    def load_checkpoint(self):
        """Load checkpoint with error handling"""
        try:
            if not (self.save_dir / 'experiment_state.json').exists():
                print("No checkpoint found. Starting fresh.")
                return None
                
            with open(self.save_dir / 'experiment_state.json', 'r') as f:
                try:
                    checkpoint = json.load(f)
                except json.JSONDecodeError:
                    print("Corrupted checkpoint file. Starting fresh.")
                    return None
                    
            if (self.save_dir / 'model_states.pt').exists():
                try:
                    states = torch.load(self.save_dir / 'model_states.pt')
                    model_state = states['model_state']
                    optimizer_state = states['optimizer_state']
                except Exception:
                    print("Error loading model states. Starting fresh.")
                    return None
            else:
                model_state = None
                optimizer_state = None
                
            return ExperimentState(
                completed_activations=checkpoint.get('completed_activations', []),
                current_activation=checkpoint.get('current_activation', ''),
                epoch=checkpoint.get('epoch', 0),
                model_state=model_state,
                optimizer_state=optimizer_state,
                frequency_stats=checkpoint.get('frequency_stats', {}),
                losses=checkpoint.get('losses', [])
            )
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}. Starting fresh.")
            return None
