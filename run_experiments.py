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

# Add model caching
_cached_model = None
_cached_dataset = None

def parse_args():
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
    return parser.parse_args()

def get_dataset(config):
    global _cached_dataset
    if _cached_dataset is None:
        _cached_dataset = TransformerActivationDataset(
            model_name=config.model_name,
            layer=config.layer,
            n_samples=config.n_samples
        )
    return _cached_dataset

def train_model(config, track_frequency=True, visualizer=None):
    """Train SAE with detailed tracking"""
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
    """Compare different activation functions"""
    results = {}
    for activation in ['relu', 'jump_relu', 'topk']:
        config.activation_type = activation
        print(f"\nStudying {activation} activation:")
        model, freq_analyzer, losses = train_model(config, visualizer=visualizer)
        
        results[activation] = {
            'final_loss': losses[-1],
            'loss_trend': losses,
            'frequency_stats': freq_analyzer.analyze() if freq_analyzer else None,
            'sparsity': compute_sparsity(model, config)  # Pass config
        }
        
        if visualizer:
            visualizer.log_activation_study(activation, results[activation])
    
    return results

def run_full_analysis(config):
    visualizer = WandBVisualizer(
        model_name=config.model_name,
        run_name=f"sae_{config.hidden_dim}_{config.activation_type}"
    )
    
    # Log experiment configuration
    visualizer.log_config(config)
    
    # Track activation studies
    activation_results = run_activation_study(config, visualizer)
    
    # Track final model analysis
    model, freq_analyzer, _ = train_model(config, track_frequency=True, visualizer=visualizer)
    freq_stats = freq_analyzer.analyze()
    
    # Track concept emergence
    concept_analyzer = ConceptAnalyzer(model, get_dataset(config))
    concept_stats = concept_analyzer.analyze_concepts()
    
    # Log comprehensive results
    visualizer.log_results({
        'activation_comparison': activation_results,
        'frequency_analysis': freq_stats,
        'concept_analysis': concept_stats
    })
    
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