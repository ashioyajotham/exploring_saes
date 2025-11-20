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
_cached_datasets = {}

# Instantiate a default config to centralize CLI defaults
_DEFAULT_CFG = SAEConfig()

def parse_args():
    """Parse command line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(description='Run SAE Experiments')
    parser.add_argument('--hidden-dim', type=int, default=_DEFAULT_CFG.hidden_dim,
                        help=f"hidden dimension (default: {_DEFAULT_CFG.hidden_dim})")
    parser.add_argument('--lr', type=float, default=_DEFAULT_CFG.learning_rate,
                        help=f"learning rate (default: {_DEFAULT_CFG.learning_rate})")
    parser.add_argument('--epochs', type=int, default=_DEFAULT_CFG.epochs,
                        help=f"epochs (default: {_DEFAULT_CFG.epochs})")
    parser.add_argument('--batch-size', type=int, default=_DEFAULT_CFG.batch_size,
                        help=f"batch size (default: {_DEFAULT_CFG.batch_size})")
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--activation', type=str, choices=['relu', 'jump_relu', 'topk'],
                       default=_DEFAULT_CFG.activation_type, help='Activation function type')
    parser.add_argument('--model-name', type=str, default=_DEFAULT_CFG.model_name,
                        help=f"transformer model (default: {_DEFAULT_CFG.model_name})")
    parser.add_argument('--layer', type=int, default=_DEFAULT_CFG.layer,
                        help=f"layer index (default: {_DEFAULT_CFG.layer})")
    parser.add_argument('--n-samples', type=int, default=_DEFAULT_CFG.n_samples,
                        help=f"number of activation samples (default: {_DEFAULT_CFG.n_samples})")
    parser.add_argument('--seq-len', type=int, default=_DEFAULT_CFG.seq_len,
                        help=f"token sequence length used when sampling (default: {_DEFAULT_CFG.seq_len})")
    # Normalization and dead-neuron resampling flags
    parser.add_argument('--normalize-activations', dest='normalize_activations', action='store_true')
    parser.add_argument('--no-normalize', dest='normalize_activations', action='store_false')
    parser.set_defaults(normalize_activations=_DEFAULT_CFG.normalize_activations)
    parser.add_argument('--dead-resample', dest='dead_resample', action='store_true')
    parser.add_argument('--no-dead-resample', dest='dead_resample', action='store_false')
    parser.set_defaults(dead_resample=_DEFAULT_CFG.dead_resample)
    parser.add_argument('--dead-threshold', type=float, default=_DEFAULT_CFG.dead_neuron_threshold,
                        help=f"dead neuron threshold (default: {_DEFAULT_CFG.dead_neuron_threshold})")
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    return parser.parse_args()

def get_dataset(config):
    global _cached_datasets
    key = (config.model_name, config.layer, config.n_samples, config.seq_len, config.normalize_activations)
    if key not in _cached_datasets:
        _cached_datasets[key] = TransformerActivationDataset(
            model_name=config.model_name,
            layer=config.layer,
            n_samples=config.n_samples,
            seq_len=config.seq_len,
            normalize=config.normalize_activations
        )
    return _cached_datasets[key]

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

        avg_loss = epoch_loss / max(1, len(dataloader))
        losses.append(avg_loss)

        # Frequency analysis & dead neuron resampling
        if frequency_analyzer:
            freq_stats = frequency_analyzer.analyze()
            mean_freq = freq_stats.get('mean_frequencies', None)
            if isinstance(mean_freq, torch.Tensor):
                mean_activation_rate = float(mean_freq.mean().item())
                high_freq_neurons = int((mean_freq > config.high_freq_threshold).sum().item())
            else:
                mean_activation_rate = 0.0
                high_freq_neurons = 0

            # Log to W&B
            if config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'loss': avg_loss,
                    'mean_activation_rate': mean_activation_rate,
                    'high_freq_neurons': high_freq_neurons,
                    'activation': config.activation_type
                })

            # Dead neuron resampling: detect and resample low-firing neurons
            if config.dead_resample and isinstance(mean_freq, torch.Tensor):
                dead_mask = (mean_freq < config.dead_neuron_threshold)
                dead_indices = torch.nonzero(dead_mask).squeeze(1).tolist()
                if len(dead_indices) > 0:
                    # Resample using a hard-example from a small scan of the dataset
                    _resample_dead_neurons(model, dataloader, dead_indices, config)

        else:
            # Basic logging if frequency analyzer is not present
            if config.use_wandb:
                wandb.log({'epoch': epoch, 'loss': avg_loss, 'activation': config.activation_type})

        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return model, frequency_analyzer, losses


def _resample_dead_neurons(model, dataloader, dead_indices, config):
    """
    Resample (reinitialize) encoder/decoder weights for dead neurons.

    Strategy:
    - Scan a small number of batches to find the sample with highest
      reconstruction error (hard example).
    - For each dead neuron, set the encoder row to a scaled copy of the
      hard sample and initialize the corresponding decoder column so the
      neuron can reconstruct that sample.
    """
    model_device = next(model.parameters()).device
    best_sample = None
    best_err = -1.0
    mse = torch.nn.functional.mse_loss

    # Scan up to 5 batches for a hard example
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            x = batch['pixel_values'].to(model_device)
            decoded, _ = model(x)
            # per-sample errors
            errs = ((decoded - x)**2).mean(dim=1)
            max_idx = torch.argmax(errs).item()
            if errs[max_idx].item() > best_err:
                best_err = errs[max_idx].item()
                best_sample = x[max_idx].detach().cpu()

    if best_sample is None:
        return

    # Apply resampling to dead neurons
    with torch.no_grad():
        enc_w = model.encoder.weight.data
        dec_w = model.decoder.weight.data
        enc_b = model.encoder.bias.data
        dec_b = model.decoder.bias.data

        # Compute a normalized direction from the hard sample
        sample = best_sample.float()
        sample_norm = sample.norm() + 1e-9
        direction = (sample / sample_norm).to(enc_w.device)

        for j in dead_indices:
            if j < 0 or j >= model.hidden_dim:
                continue
            # set encoder row to be aligned with the hard sample
            scale = enc_w.std().item() if enc_w.std().item() > 0 else 0.1
            enc_w[j, :] = direction * (scale * config.dead_resample_strength)
            enc_b[j] = 0.0
            # set decoder column so that passing the neuron's activation reconstructs the sample
            dec_w[:, j] = sample.to(dec_w.device) * (1.0 / max(1.0, model.hidden_dim))
            dec_b[:] = 0.0

    print(f"Resampled {len(dead_indices)} dead neurons: indices={dead_indices}")

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