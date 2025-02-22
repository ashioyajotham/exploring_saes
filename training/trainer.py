import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import wandb
import argparse

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.autoencoder import SparseAutoencoder
from src.models.model_loader import ModelLoader

class SAETrainer:
    def __init__(
        self, 
        model: nn.Module,
        lr: float = 0.001,
        l1_coef: float = 0.001,
        use_wandb: bool = False,
        wandb_config: dict = None
    ):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.l1_coef = l1_coef
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(
                project="sae-interpretability",
                config=wandb_config or {
                    "learning_rate": lr,
                    "l1_coefficient": l1_coef,
                    "model_type": model.__class__.__name__,
                    "input_dim": model.input_dim,
                    "hidden_dim": model.hidden_dim
                }
            )
            wandb.watch(model, log_freq=100)

    def compute_loss(self, reconstructed, original, encoded):
        """Compute reconstruction loss with L1 sparsity penalty"""
        # MSE reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(reconstructed, original)
        
        # L1 sparsity loss
        sparsity_loss = self.l1_coef * torch.norm(encoded, 1)
        
        return recon_loss + sparsity_loss

    def train_step(self, batch):
        self.optimizer.zero_grad()
        # Unpack batch tuple and ensure tensor
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        reconstructed, encoded = self.model(inputs)
        loss = self.compute_loss(reconstructed, inputs, encoded)
        loss.backward()
        self.optimizer.step()
        return loss.item(), encoded

    def train_epoch(self, dataloader, epoch: int):
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            losses = self.train_step(batch)
            epoch_losses.append(losses)
            
            if self.use_wandb:
                # Log detailed metrics
                wandb.log({
                    "epoch": epoch,
                    "batch": batch_idx,
                    **losses,
                    "activation_sparsity": self._compute_activation_sparsity(),
                    "weight_histogram": wandb.Histogram(self.model.encoder.weight.data.cpu()),
                    "activation_heatmap": wandb.Image(self._plot_activation_heatmap())
                })
        
        return epoch_losses

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
                
                if self.use_wandb:
                    wandb.log({
                        "loss": loss,
                        "sparsity": (encoded > 0).float().mean().item()
                    })
                    
            print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}")

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
    trainer = SAETrainer(model=sae, use_wandb=args.wandb)

    # Create dataloader
    dataset = torch.utils.data.TensorDataset(activations)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    print("Starting training...")
    trainer.train(dataloader, args.epochs)

if __name__ == "__main__":
    main()