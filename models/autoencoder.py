import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_param=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_param = sparsity_param
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # Ensure input is 2D: (batch_size, input_dim)
        if len(x.shape) > 2:
            x = x.view(-1, x.shape[-1])
        encoded = torch.relu(self.encoder(x))
        sparse_encoded = self.apply_sparsity(encoded)
        decoded = self.decoder(sparse_encoded)
        return decoded, sparse_encoded
    
    def apply_sparsity(self, activations):
        return activations * (activations > self.sparsity_param).float()

    def get_dead_neurons(self, threshold=0.01):
        """Identify neurons that are consistently inactive"""
        with torch.no_grad():
            weights = self.encoder.weight.data
            return torch.where(weights.abs().mean(dim=1) < threshold)[0]
    
    def get_feature_directions(self):
        """Extract feature directions from encoder weights"""
        return self.encoder.weight.data.T
