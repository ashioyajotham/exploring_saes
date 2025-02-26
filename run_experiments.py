import argparse
import torch
from torch.utils.data import DataLoader
from config.config import SAEConfig  # Use this only
import wandb
from tqdm import tqdm
import transformer_lens

from models.autoencoder import SparseAutoencoder
from experiments.frequency_analysis import FrequencyAnalyzer
from experiments.concept_emergence import ConceptAnalyzer
from experiments.transformer_data import TransformerActivationDataset

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
    return TransformerActivationDataset(
        model_name=config.model_name,
        layer=config.layer,
        n_samples=config.n_samples
    )

def train_model(config, track_frequency=True):
    """Train SAE with detailed tracking"""
    model = SparseAutoencoder(
        input_dim=784,
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

def compute_sparsity(model):
    """Compute activation sparsity of the model"""
    with torch.no_grad():
        total_zeros = 0
        total_elements = 0
        
        # Get dataset sample
        dataset = get_dataset()
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        batch = next(iter(dataloader))
        
        # Forward pass
        _, encoded = model(batch["pixel_values"])
        
        # Calculate sparsity
        zeros = (encoded.abs() < 1e-5).float().mean().item()
        return zeros

def run_activation_study(config):
    """Compare different activation functions"""
    results = {}
    for activation in ['relu', 'jump_relu', 'topk']:
        config.activation_type = activation
        print(f"\nStudying {activation} activation:")
        model, freq_analyzer, losses = train_model(config)
        
        results[activation] = {
            'final_loss': losses[-1],
            'loss_trend': losses,
            'frequency_stats': freq_analyzer.analyze() if freq_analyzer else None,
            'sparsity': compute_sparsity(model)
        }
    
    return results

def run_full_analysis(config):
    """Run comprehensive analysis suite"""
    if config.use_wandb:
        wandb.init(
            project="sae-interpretability",
            name=f"sae_{config.hidden_dim}_{config.activation_type}",
            config=vars(config)
        )
    
    print("\n=== Running Activation Function Study ===")
    activation_results = run_activation_study(config)
    
    print("\n=== Training Final Model ===")
    model, freq_analyzer, _ = train_model(config, track_frequency=True)
    
    print("\n=== Analyzing Concept Emergence ===")
    concept_analyzer = ConceptAnalyzer(model, get_dataset())
    concept_stats = concept_analyzer.analyze_concepts()
    
    results = {
        'activation_comparison': activation_results,
        'frequency_analysis': freq_analyzer.analyze() if freq_analyzer else None,
        'concept_analysis': concept_stats
    }
    
    print_detailed_results(results)
    return results

def print_detailed_results(results):
    """Print comprehensive results summary"""
    print("\n╔════════════════════ EXPERIMENT RESULTS ════════════════════╗")
    
    print("\n=== Activation Function Comparison ===")
    for act_type, stats in results['activation_comparison'].items():
        print(f"\n{act_type.upper()}:")
        print(f"  Final Loss: {stats['final_loss']:.4f}")
        print(f"  Sparsity: {stats['sparsity']:.4f}")
    
    print("\n=== Frequency Analysis ===")
    freq = results['frequency_analysis']
    print(f"High Frequency Neurons: {freq['high_freq_neurons']}")
    print(f"Mean Activation Rate: {freq['mean_frequencies'].mean():.4f}")
    
    print("\n=== Concept Analysis ===")
    concept = results['concept_analysis']
    print(f"Number of Clusters: {len(concept['feature_clusters'])}")
    print(f"Average Similarity: {concept['concept_similarity']['mean_similarity']:.4f}")
    
    print("\n╚═══════════════════════════════════════════════════════════╝")

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

class SAEConfig:
    def __init__(self, **kwargs):
        self.input_dim = kwargs.get('input_dim', 784)
        self.hidden_dim = kwargs.get('hidden_dim', 256)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 64)
        self.activation_type = kwargs.get('activation_type', 'relu')
        self.use_wandb = kwargs.get('use_wandb', False)
        self.model_name = kwargs.get('model_name', 'gpt2-small')
        self.layer = kwargs.get('layer', 0)
        self.n_samples = kwargs.get('n_samples', 1000)
