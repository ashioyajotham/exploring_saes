import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import wandb
from torchvision import transforms

from models.autoencoder import SparseAutoencoder
from experiments.activation_study import run_activation_comparison
from experiments.frequency_analysis import FrequencyAnalyzer
from experiments.concept_emergence import ConceptAnalyzer
from config.config import SAEConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Run SAE Experiments')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--activation', type=str, choices=['relu', 'jump_relu', 'topk'],
                       default='relu', help='Activation function type')
    return parser.parse_args()

def get_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    mnist_dataset = load_dataset("mnist")
    return mnist_dataset["train"].with_transform(
        lambda examples: {
            "pixel_values": torch.stack([transform(image) for image in examples["image"]])
        }
    )

def train_model(config):
    # Initialize model and optimizer
    model = SparseAutoencoder(
        input_dim=784,  # MNIST dimension
        hidden_dim=config.hidden_dim,
        activation_type=config.activation_type
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Get dataset
    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(config.epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch["pixel_values"]
            reconstructed, encoded = model(inputs)
            
            # Update frequency analyzer
            if hasattr(model, 'frequency_analyzer'):
                model.frequency_analyzer.update(encoded)
                
            loss = torch.nn.functional.mse_loss(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if config.use_wandb:
            # Log metrics separately from metadata
            wandb.log({
                'epoch': epoch,
                'loss': total_loss/len(dataloader)
            }, commit=False)  # Don't commit yet
            
            # Log metadata as separate entry
            wandb.run.summary.update({
                'activation_type': config.activation_type
            })
            
            wandb.log({})  # Commit the step
            
    return model

def run_full_analysis(config):
    print("Running activation comparison...")
    activation_results = run_activation_comparison(config, train_model)
    
    print("Analyzing frequency patterns...")
    model = train_model(config)
    frequency_analyzer = FrequencyAnalyzer(model)
    freq_stats = frequency_analyzer.analyze()
    
    print("Analyzing concept emergence...")
    concept_analyzer = ConceptAnalyzer(model, get_dataset())
    concept_stats = concept_analyzer.analyze_concepts()
    
    return {
        'activation_comparison': activation_results,
        'frequency_analysis': freq_stats,
        'concept_analysis': concept_stats
    }

def print_ascii_results(results):
    """Print experiment results in ASCII art format"""
    print("""
╔══════════════════════════════════════════════╗
║             Experiment Results               ║
╠══════════════════════════════════════════════╣
║                                             ║""")
    
    print(f"║  Activation Functions Analyzed: {len(results['activation_comparison']):>8}        ║")
    print(f"║  High Frequency Neurons: {results['frequency_analysis']['high_freq_neurons']:>13}        ║")
    print(f"║  Concept Clusters Found: {len(results['concept_analysis']['feature_clusters']):>12}        ║")
    
    print("""║                                             ║
║  Activation Stats:                          ║""")
    for act_type, stats in results['activation_comparison'].items():
        print(f"║    • {act_type:<10}: {stats.get('sparsity', 0):.3f} sparsity    ║")
    
    print("""╚══════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    args = parse_args()
    config = SAEConfig(
        input_dim=784,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_wandb=args.use_wandb,
        activation_type=args.activation  # Add activation type
    )
    
    results = run_full_analysis(config)
    print_ascii_results(results)
    print("\nExperiment Results:")
    print("==================")
    print(f"Activation Comparisons: {len(results['activation_comparison'])} functions analyzed")
    print(f"High Frequency Neurons: {results['frequency_analysis']['high_freq_neurons']}")
    print(f"Concept Clusters: {len(results['concept_analysis']['feature_clusters'])}")
