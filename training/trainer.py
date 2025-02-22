import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import wandb
import argparse
import io
from PIL import Image

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.autoencoder import SparseAutoencoder
from models.model_loader import ModelLoader

class SAETrainer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer  # Use the passed optimizer directly
        self.config = config
        self.current_losses = {}

    def compute_loss(self, reconstructed, inputs, encoded):
        recon_loss = torch.nn.functional.mse_loss(reconstructed, inputs)
        sparsity_loss = self.config.l1_coefficient * torch.norm(encoded, 1)  # Use dot notation
        total_loss = recon_loss + sparsity_loss
        
        self.current_losses = {
            'reconstruction_loss': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'total_loss': total_loss.item()
        }
        return total_loss

    def train_step(self, batch):
        self.optimizer.zero_grad()
        inputs = batch["pixel_values"]  # Extract tensor from dict
        reconstructed, encoded = self.model(inputs)
        loss = self.compute_loss(reconstructed, inputs, encoded)
        loss.backward()
        self.optimizer.step()
        return self.current_losses, encoded

    def train_epoch(self, dataloader, epoch: int):
        self.model.train()
        epoch_losses = {
            'reconstruction_loss': 0.0,
            'sparsity_loss': 0.0,
            'total_loss': 0.0
        }
        
        for batch_idx, batch in enumerate(dataloader):
            batch_losses, encoded = self.train_step(batch)
            for k in epoch_losses:
                epoch_losses[k] += batch_losses[k]
        
        # Average losses over batches
        num_batches = len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            
        return epoch_losses, encoded

    def _compute_activation_sparsity(self):
        """Compute fraction of zero activations"""
        with torch.no_grad():
            sample_batch = next(iter(self.dataloader))
            _, encoded = self.model(sample_batch)
            return (encoded == 0).float().mean().item()

    def _plot_activation_heatmap(self):
        """Generate activation heatmap for visualization"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        with torch.no_grad():
            sample_batch = next(iter(self.dataloader))
            _, encoded = self.model(sample_batch)
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(encoded[0].cpu().numpy(), cmap='viridis')
            plt.title("Neuron Activations")
            plt.xlabel("Hidden Dimension")
            plt.ylabel("Sample")
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf)
            plt.close()
            buf.seek(0)
            return Image.open(buf)

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                loss, encoded = self.train_step(batch)
                total_loss += loss
                
                if self.config['use_wandb']:
                    wandb.log({
                        "loss": loss,
                        "sparsity": (encoded > 0).float().mean().item()
                    })
                    
            print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}")

    def get_losses(self):
        return self.current_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--model", default="bert", choices=["bert", "gpt2"])
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    # Initialize model and get data
    model_loader = ModelLoader(args.model)
    sample_text = "The quick brown fox jumps over the lazy dog"
    activations = model_loader.get_activations(sample_text)
    
    # Reshape activations: (batch_size, seq_length, hidden_size) -> (batch_size * seq_length, hidden_size)
    activations = activations.view(-1, activations.size(-1))
    print(f"Input shape: {activations.shape}")
    
    # Setup SAE with correct dimensions
    sae = SparseAutoencoder(
        input_dim=activations.shape[1],  # hidden_size
        hidden_dim=64
    )
    print(f"Encoder weight shape: {sae.encoder.weight.shape}")

    # Initialize trainer
    optimizer = optim.Adam(sae.parameters(), lr=0.001)
    config = {
        "l1_coef": 0.001,
        "use_wandb": args.wandb
    }
    trainer = SAETrainer(model=sae, optimizer=optimizer, config=config)

    # Create dataloader
    dataset = torch.utils.data.TensorDataset(activations)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    print("Starting training...")
    trainer.train(dataloader, args.epochs)

if __name__ == "__main__":
    main()