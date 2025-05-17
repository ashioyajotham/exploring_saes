import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from PIL import Image
import io
from typing import Optional, Dict, List
from datetime import datetime

class WandBVisualizer:
    def __init__(self, model_name: str, config: dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = model_name.replace('-', '')
        dim = config.get('hidden_dim', 256)
        layer = config.get('layer', 0)
        
        # Check for comparison mode
        if config.get('is_comparison', False):
            run_type = 'comparison'
            run_name = f"sae_{run_type}_{model_short}_l{layer}_d{dim}_{timestamp}"
        else:
            run_type = 'single'
            run_name = f"sae_{run_type}_{model_short}_l{layer}_d{dim}_{config.get('activation_type', 'relu')}_{timestamp}"
            
        wandb.init(
            project="sae-interpretability",
            name=run_name,
            config=config
        )
        
    def log_config(self, config):
        """Log experiment configuration"""
        wandb.config.update(vars(config))
        
    def log_activation_study(self, activation_type: str, results: dict):
        """Log activation function analysis"""
        # Log metrics
        wandb.log({
            f"metrics/{activation_type}/loss": results["final_loss"],
            f"metrics/{activation_type}/sparsity": results["sparsity"]
        })

        # Log visualizations
        wandb.log({
            f"viz/{activation_type}/feature_maps": wandb.Image(
                self._plot_feature_maps(results["feature_weights"])
            ),
            f"viz/{activation_type}/activations": wandb.Image(
                self._plot_activation_heatmap(results["frequency_stats"]["activations"])
            )
        })
        
        # Update run config instead of logging as metric
        wandb.config.update({
            f"activation_types/{activation_type}": {
                "final_loss": results["final_loss"],
                "sparsity": results["sparsity"]
            }
        }, allow_val_change=True)
        
    def log_results(self, results: dict):
        """Log comprehensive analysis results"""
        # Frequency analysis
        freq = results["frequency_analysis"]
        mean_freq = torch.tensor(freq["mean_frequencies"]) if isinstance(freq["mean_frequencies"], list) else freq["mean_frequencies"]
        
        wandb.log({
            "freq/high_freq_neurons": freq["high_freq_neurons"],
            "freq/mean_activation": mean_freq.mean().item(),
            "freq/activation_pattern": wandb.Image(
                self._plot_frequency_pattern(mean_freq)
            )
        })
        
        # Concept analysis
        concept = results["concept_analysis"]
        wandb.log({
            "concept/n_clusters": len(concept["feature_clusters"]),
            "concept/similarity": concept["concept_similarity"]["mean_similarity"],
            "concept/embedding": wandb.Image(
                self._plot_concept_embedding(concept.get("embedding", torch.zeros(1, 2)))
            ) if "embedding" in concept else None
        })

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

    def log_frequency_analysis(self, freq_stats, epoch):
        wandb.log({
            "epoch": epoch,
            "frequency/high_freq_neurons": freq_stats['high_freq_neurons'],
            "frequency/distribution": wandb.Histogram(freq_stats['mean_frequencies']),
            "frequency/temporal": freq_stats['temporal_stats']
        })
    
    def log_concept_analysis(self, concept_stats, epoch):
        wandb.log({
            "epoch": epoch,
            "concepts/similarity": concept_stats['concept_similarity'],
            "concepts/clusters": concept_stats['feature_clusters']
        })

    def log_training(self, data: dict):
        """Log training metrics and visualizations"""
        wandb.log({
            'epoch': data['epoch'],
            'loss': data['loss'],
            'activation_heatmap': wandb.Image(
                self._plot_activation_heatmap(data['encoded'])
            ),
            'feature_maps': wandb.Image(
                self._plot_feature_maps(data['weights'])
            ),
            'activation_distribution': wandb.Histogram(
                data['encoded'].abs().flatten().cpu().numpy()
            )
        })

    def finish(self):
        """Close the W&B run"""
        wandb.finish()

    def _plot_activation_heatmap(self, activations):
        """Generate heatmap of neuron activations"""
        plt.figure(figsize=(10, 4))
        plt.imshow(activations.cpu().detach().numpy().T, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.xlabel('Sample')
        plt.ylabel('Neuron')
        plt.title('Activation Patterns')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf)
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    def _plot_feature_maps(self, weights):
        """Visualize learned features"""
        plt.figure(figsize=(12, 4))
        plt.imshow(weights.cpu().detach().numpy(), aspect='auto', cmap='RdBu')
        plt.colorbar()
        plt.xlabel('Input Dimension')
        plt.ylabel('Feature')
        plt.title('Feature Maps')
        
        buf = io.BytesIO()
        plt.savefig(buf)
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    def _plot_frequency_pattern(self, frequencies):
        """Plot neuron activation frequencies"""
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(frequencies)), frequencies.cpu().numpy())
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Frequency')
        plt.title('Neuron Firing Patterns')
        
        buf = io.BytesIO()
        plt.savefig(buf)
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    def _plot_concept_embedding(self, embeddings):
        """Visualize concept embeddings using UMAP"""
        plt.figure(figsize=(10, 10))
        
        # Reduce dimensionality for visualization if needed
        if embeddings.shape[1] > 2:
            reducer = umap.UMAP(random_state=42)
            embedding_2d = reducer.fit_transform(embeddings.cpu().numpy())
        else:
            embedding_2d = embeddings.cpu().numpy()
            
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.5)
        plt.title('Concept Embedding Space')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        
        buf = io.BytesIO()
        plt.savefig(buf)
        plt.close()
        buf.seek(0)
        return Image.open(buf)
