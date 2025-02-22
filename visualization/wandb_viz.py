import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from typing import Optional, Dict, List

class WandBVisualizer:
    def __init__(self, model_name: str, run_name: Optional[str] = None):
        self.run = wandb.init(
            project="sae-interpretability",
            name=run_name,
            tags=[model_name],
            config={
                "model": model_name,
                "learning_rate": 0.001,
                "hidden_dim": 64,
                "sparsity_param": 0.1
            }
        )
        
    def log_feature_embeddings(self, features: torch.Tensor, metadata: Optional[Dict] = None):
        """Log feature embeddings for projection visualization"""
        wandb.log({
            "feature_embeddings": wandb.Table(
                columns=["features", "metadata"],
                data=[[f.tolist(), metadata] for f in features]
            )
        })

    def log_neuron_analysis(self, 
                           activations: torch.Tensor,
                           neuron_ids: List[int],
                           example_inputs: List[str]):
        """Log individual neuron analysis"""
        for neuron_id in neuron_ids:
            neuron_acts = activations[:, neuron_id]
            
            # Create histogram of activations
            fig, ax = plt.subplots()
            ax.hist(neuron_acts.cpu().numpy(), bins=50)
            ax.set_title(f"Neuron {neuron_id} Activation Distribution")
            
            # Find top activating examples
            top_k = 5
            top_indices = torch.topk(neuron_acts, top_k).indices
            top_examples = [example_inputs[i] for i in top_indices]
            
            wandb.log({
                f"neuron_{neuron_id}/activation_dist": wandb.Image(fig),
                f"neuron_{neuron_id}/top_examples": wandb.Table(
                    columns=["example", "activation"],
                    data=[[ex, neuron_acts[i].item()] for i, ex in zip(top_indices, top_examples)]
                )
            })
            plt.close()

    def log_training_progress(self, 
                            epoch: int,
                            losses: Dict[str, float],
                            model_state: Dict):
        """Log training metrics and model state"""
        wandb.log({
            "epoch": epoch,
            **losses,
            "model_state": {
                "encoder_norm": torch.norm(model_state['encoder.weight']).item(),
                "decoder_norm": torch.norm(model_state['decoder.weight']).item()
            }
        })

    def create_interactive_dashboard(self):
        """Create an interactive W&B dashboard"""
        wandb.log({"custom_chart": wandb.plot.line_series(
            xs=list(range(100)),
            ys=[[x**2] for x in range(100)],
            keys=["squared"],
            title="Training Progress",
            xname="epoch"
        )})

    def log_training_step(self, step: int, loss: float, activations: torch.Tensor, 
                         weights: torch.Tensor):
        """Log training metrics and visualizations"""
        # Calculate sparsity
        sparsity = (activations > 0).float().mean().item()
        
        # Create weight distribution plot
        fig, ax = plt.subplots()
        ax.hist(weights.flatten().cpu().numpy(), bins=50)
        ax.set_title("Weight Distribution")
        
        # Log metrics
        wandb.log({
            "step": step,
            "loss": loss,
            "sparsity": sparsity,
            "weight_dist": wandb.Image(fig),
            "activation_heatmap": wandb.Image(
                activations.T.cpu().numpy(),
                caption="Neuron Activations"
            )
        })
        plt.close()
    
    def log_feature_directions(self, features: torch.Tensor, step: int):
        """Log learned feature directions"""
        wandb.log({
            "step": step,
            "feature_directions": wandb.Image(
                features.T.cpu().numpy(),
                caption=f"Feature Directions at Step {step}"
            )
        })

    def finish(self):
        """Close the W&B run"""
        wandb.finish()
