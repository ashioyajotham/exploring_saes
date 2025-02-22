import torch
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms
import wandb

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder import SparseAutoencoder
from training.trainer import SAETrainer
from evaluation.analyzer import SAEAnalyzer
from config.config import SAEConfig
from visualization.wandb_viz import WandBVisualizer

import datasets

def get_activations_for_samples(model, samples):
    model.eval()
    with torch.no_grad():
        activations = []
        for sample in samples:
            input_tensor = torch.tensor(sample).float().unsqueeze(0)
            _, encoded = model(input_tensor)
            activations.append(encoded)
    return activations

def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Autoencoder')
    parser.add_argument('--input-dim', type=int, default=784)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--use-wandb', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize W&B visualizer
    visualizer = WandBVisualizer(
        model_name="sae-experiment",
        run_name=f"sae_{args.hidden_dim}_{args.lr}"
    )
    
    # Load configuration
    config = SAEConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Initialize model, optimizer, and trainer
    model = SparseAutoencoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        sparsity_param=config.sparsity_param
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    trainer = SAETrainer(model, optimizer, config)

    # Define data transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
    ])
    
    # Load and transform MNIST dataset
    mnist_dataset = datasets.load_dataset("mnist")
    train_dataset = mnist_dataset["train"].with_transform(
        lambda examples: {
            "pixel_values": torch.stack([transform(image) for image in examples["image"]])
        }
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True
    )

    # Training loop with visualization
    print("Starting training...")
    for epoch in range(config.epochs):
        epoch_loss, encoded = trainer.train_epoch(dataloader, epoch)
        if config.use_wandb:
            wandb.log({
                "epoch": epoch,
                "loss": epoch_loss
            })
        
        # Log to W&B
        losses = trainer.get_losses()
        visualizer.log_training_progress(
            epoch=epoch,
            losses={k: sum(d[k] for d in losses)/len(losses) for k in losses[0]},
            model_state=model.state_dict()
        )
        
        if epoch % 10 == 0:  # Every 10 epochs
            # Get feature embeddings
            with torch.no_grad():
                _, encoded = model(next(iter(dataloader)))
                visualizer.log_feature_embeddings(
                    features=encoded,
                    metadata={"epoch": epoch}
                )

    # Final analysis
    analyzer = SAEAnalyzer(model)
    stats = analyzer.analyze_sparsity(dataloader)
    
    # Log neuron analysis
    sample_texts = ["The quick brown fox jumps over the lazy dog",
                    "The five boxing wizards jump quickly"]
    activations = get_activations_for_samples(model, sample_texts)
    visualizer.log_neuron_analysis(
        activations=activations,
        neuron_ids=range(10),  # Analyze first 10 neurons
        example_inputs=sample_texts
    )
    
    visualizer.create_interactive_dashboard()
    visualizer.finish()

if __name__ == "__main__":
    main()