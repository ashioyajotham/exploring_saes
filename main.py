import torch
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms
from datasets import load_dataset
import wandb

from models.autoencoder import SparseAutoencoder
from training.trainer import SAETrainer
from config.config import SAEConfig
from visualization.wandb_viz import WandBVisualizer

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

    print("Starting training...")
    for epoch in range(config.epochs):
        # Train epoch
        epoch_losses, encoded = trainer.train_epoch(dataloader, epoch)
        
        # Log metrics
        if config.use_wandb:
            wandb.log({
                "epoch": epoch,
                **epoch_losses
            })
        
        # Feature visualization every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                batch = next(iter(dataloader))
                inputs = batch["pixel_values"]
                _, encoded = model(inputs)
                visualizer.log_feature_embeddings(
                    features=encoded,
                    metadata={"epoch": epoch}
                )
    
    visualizer.finish()

if __name__ == "__main__":
    main()