"""
Sparse Autoencoder Experiment Runner
===================================

Main entry point for running SAE experiments on transformer activations.
Handles training, analysis, visualization and checkpointing.

Key Components:
- Model training with multiple activation functions
- Neuron frequency analysis
- Concept emergence tracking
- Experiment checkpointing
- W&B and ASCII visualization

Usage:
------
Basic training:
    python run_experiments.py --hidden-dim 256 --epochs 100

Transformer analysis:
    python run_experiments.py --model-name gpt2-small --layer 0 --n-samples 1000 --use-wandb

Functions:
----------
parse_args(): Configure experiment parameters
get_dataset(): Load and cache transformer activations
train_model(): Train SAE with specified config
compute_sparsity(): Calculate activation sparsity
run_activation_study(): Compare activation functions
run_full_analysis(): Execute complete analysis suite
"""

import argparse
import torch
from torch.utils.data import DataLoader
from config.config import SAEConfig
import wandb
from tqdm import tqdm
import transformer_lens
from visualization.ascii_viz import ASCIIVisualizer
from visualization.wandb_viz import WandBVisualizer

from models.autoencoder import SparseAutoencoder
from experiments.frequency_analysis import FrequencyAnalyzer
from experiments.concept_emergence import ConceptAnalyzer
from experiments.transformer_data import TransformerActivationDataset
from experiments.checkpointing import CheckpointManager, ExperimentState

# Add model caching
_cached_model = None
_cached_dataset = None

def parse_args():
    """Parse command line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(description='Run SAE Experiments')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--activation', type=str, choices=['relu', 'jump_relu', 'topk'],
                       default='relu', help='Activation function type')
    parser.add_argument('--model-name', type=str, default='gpt2-small')
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    return parser.parse_args()

def get_dataset(config):
    global _cached_dataset
    if (_cached_dataset is None):
        _cached_dataset = TransformerActivationDataset(
            model_name=config.model_name,
            layer=config.layer,
            n_samples=config.n_samples
        )
    return _cached_dataset

def train_model(config, track_frequency=True, visualizer=None):
    """
    Train Sparse Autoencoder model.
    
    Args:
        config: SAEConfig object with model/training parameters
        track_frequency: Enable neuron firing rate tracking
        visualizer: Optional W&B visualization handler
        
    Returns:
        model: Trained SAE model
        freq_analyzer: Frequency analysis results
        losses: Training loss history
    """
    # Get dataset first to determine input dimension
    dataset = get_dataset(config)
    sample = dataset[0]["pixel_values"]
    input_dim = sample.shape[0]  # Get actual dimension from transformer

    model = SparseAutoencoder(
        input_dim=input_dim,  # Use transformer's hidden dimension
        hidden_dim=config.hidden_dim,
        activation_type=config.activation_type
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Setup tracking
    frequency_analyzer = FrequencyAnalyzer(model) if track_frequency else None
    losses = []
    
    # Get dataset
    dataset = get_dataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training loop
    progress_bar = tqdm(range(config.epochs), desc=f"Training {config.activation_type}")
    for epoch in progress_bar:
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch["pixel_values"]
            reconstructed, encoded = model(inputs)
            
            if frequency_analyzer:
                frequency_analyzer.update(encoded)
                
            loss = torch.nn.functional.mse_loss(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if visualizer:
                visualizer.log_training({
                    'epoch': epoch,
                    'loss': loss.item(),
                    'encoded': encoded.detach(),
                    'weights': model.encoder.weight.data
                })
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if config.use_wandb:
            wandb.log({
                'epoch': epoch,
                'loss': avg_loss,
                'activation': config.activation_type
            })
            
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return model, frequency_analyzer, losses

def compute_sparsity(model, config):
    """Compute activation sparsity using cached dataset"""
    with torch.no_grad():
        dataset = get_dataset(config)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        batch = next(iter(dataloader))
        
        _, encoded = model(batch["pixel_values"])
        zeros = (encoded.abs() < 1e-5).float().mean().item()
        return zeros

def run_activation_study(config, visualizer=None):
    """
    Compare different activation functions.
    
    Trains models with ReLU, JumpReLU and TopK activations,
    tracking performance metrics and neuron behavior.
    
    Args:
        config: Experiment configuration
        visualizer: Optional visualization handler
        
    Returns:
        Dictionary containing results for each activation
    """
    checkpoint_manager = CheckpointManager()
    state = checkpoint_manager.load_checkpoint()
    
    results = {}
    activations = ['relu', 'jump_relu', 'topk']
    
    # Resume from checkpoint if exists
    if state:
        results = {act: {} for act in state.completed_activations}
        start_idx = activations.index(state.current_activation)
    else:
        start_idx = 0
        
    for activation in activations[start_idx:]:
        try:
            config.activation_type = activation
            print(f"\nStudying {activation} activation:")
            model, freq_analyzer, losses = train_model(config, visualizer=visualizer)
            
            results[activation] = {
                'final_loss': losses[-1],
                'loss_trend': losses,
                'frequency_stats': freq_analyzer.analyze() if freq_analyzer else None,
                'sparsity': compute_sparsity(model, config),
                'feature_weights': model.encoder.weight.data,
                'model': model
            }
            
            # Save checkpoint after each activation
            state = ExperimentState(
                completed_activations=list(results.keys()),
                current_activation=activation,
                epoch=config.epochs,
                model_state=model.state_dict(),
                optimizer_state=None,  # Add if needed
                frequency_stats=results[activation]['frequency_stats'],
                losses=losses
            )
            checkpoint_manager.save_checkpoint(state, config)
            
        except Exception as e:
            print(f"\nError during {activation} training: {str(e)}")
            print("You can resume from this point using the checkpoint")
            raise e
            
    return results

def run_full_analysis(config):
    """Execute comprehensive analysis suite."""
    # Set comparison mode flag
    config.is_comparison = True
    
    visualizer = WandBVisualizer(
        model_name=config.model_name,
        config=vars(config)
    )
    
    # Get activation study results and final model
    print("\n=== Running Activation Function Study ===")
    activation_results = run_activation_study(config, visualizer)
    
    # Use last trained model instead of training new one
    print("\n=== Analyzing Final Model ===")
    model = activation_results[config.activation_type]['model']  # Get cached model
    freq_analyzer = FrequencyAnalyzer(model)
    freq_stats = freq_analyzer.analyze()
    
    print("\n=== Analyzing Concept Emergence ===")
    concept_analyzer = ConceptAnalyzer(model, get_dataset(config))
    concept_stats = concept_analyzer.analyze_concepts()
    
    results = {
        'activation_comparison': activation_results,
        'frequency_analysis': freq_stats,
        'concept_analysis': concept_stats
    }
    
    visualizer.log_results(results)
    ASCIIVisualizer.print_results(results)
    return results

if __name__ == "__main__":
    args = parse_args()
    config = SAEConfig(
        input_dim=784,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        activation_type=args.activation,
        use_wandb=args.use_wandb,
        model_name=args.model_name,
        layer=args.layer,
        n_samples=args.n_samples
    )
    
    results = run_full_analysis(config)