"""
Sparse Autoencoder Training and Analysis Pipeline
==============================================

This module orchestrates the training and analysis of sparse autoencoders (SAEs)
for mechanistic interpretability research. It coordinates:

1. Model initialization and training
2. Real-time visualization
3. Activation pattern analysis 
4. Concept emergence tracking
5. W&B experiment logging

Components:
----------
- SAE Model: Custom autoencoder with configurable sparsity
- Training: Batch processing with activation tracking
- Analysis: Frequency patterns and concept formation
- Visualization: Live training metrics and neuron behavior

Usage:
------
```bash
python main.py --hidden-dim 256 --lr 0.001 --epochs 100 --use-wandb
```

Returns:
    argparse.Namespace: Parsed arguments including:
        - hidden_dim: Dimension of latent space
        - lr: Learning rate
        - epochs: Number of training epochs
        - batch_size: Training batch size
        - use_wandb: Whether to log to W&B

"""
import torch
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms
from datasets import load_dataset
import wandb
import sys
from PyQt5.QtWidgets import QApplication

from models.autoencoder import SparseAutoencoder
from training.trainer import SAETrainer
from config.config import SAEConfig
from visualization.wandb_viz import WandBVisualizer
from visualization.training_viz import TrainingVisualizer
from experiments.activation_study import run_activation_comparison
from experiments.frequency_analysis import FrequencyAnalyzer
from experiments.concept_emergence import ConceptAnalyzer

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
    """
    Workflow:
    1. Initialize model, optimizer, and datasets
    2. Set up visualization and analysis tools
    3. Train model while tracking:
        - Loss metrics
        - Activation patterns
        - Feature emergence
    4. Log results and visualizations
    """
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
        batch_size=args.batch_size,
        use_wandb=args.use_wandb
    )

    # Initialize model and optimizer
    model = SparseAutoencoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        sparsity_param=config.sparsity_param
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    trainer = SAETrainer(model, optimizer, config)

    # Data pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    mnist_dataset = load_dataset("mnist")
    train_dataset = mnist_dataset["train"].with_transform(
        lambda examples: {
            "pixel_values": torch.stack([transform(image) for image in examples["image"]])
        }
    )
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    # Initialize analyzers
    frequency_analyzer = FrequencyAnalyzer(model)
    concept_analyzer = ConceptAnalyzer(model, train_dataset)

    # Initialize Qt application
    app = QApplication(sys.argv)
    viz = TrainingVisualizer()
    viz.show()

    print("Starting training...")
    for epoch in range(config.epochs):
        # Train epoch
        epoch_losses, encoded = trainer.train_epoch(dataloader, epoch)
        frequency_analyzer.update(encoded)
        
        # Update visualization
        viz.update_plots(
            loss=epoch_losses['total_loss'],
            activations=encoded,
            hidden_weights=model.encoder.weight.data
        )
        app.processEvents()  # Update GUI
        
        # Log metrics
        if config.use_wandb:
            wandb.log({
                "epoch": epoch,
                **epoch_losses
            })
        
        # Feature visualization and analysis every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                batch = next(iter(dataloader))
                inputs = batch["pixel_values"]
                _, encoded = model(inputs)
                visualizer.log_feature_embeddings(
                    features=encoded,
                    metadata={"epoch": epoch}
                )
                freq_stats = frequency_analyzer.analyze()
                concept_stats = concept_analyzer.analyze_concepts()
                
                if config.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        **epoch_losses,
                        **freq_stats,
                        **concept_stats
                    })
    
    visualizer.finish()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()