import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_param=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_param = sparsity_param
        
        # Initialize weights with Xavier/Glorot initialization
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self._init_weights()
        
        # Track activations for analysis
        self.activation_stats = {
            'mean': 0,
            'sparsity': 0,
            'dead_neurons': 0
        }
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
    
    def forward(self, x):
        # Ensure input is 2D: (batch_size, input_dim)
        if len(x.shape) > 2:
            x = x.view(-1, self.input_dim)
            
        # Encode
        encoded = self.encoder(x)
        encoded_activated = F.relu(encoded)  # ReLU for sparsity
        
        # Update activation statistics
        with torch.no_grad():
            self.activation_stats['mean'] = encoded_activated.mean().item()
            self.activation_stats['sparsity'] = (encoded_activated == 0).float().mean().item()
            self.activation_stats['dead_neurons'] = (encoded_activated.sum(0) == 0).float().mean().item()
        
        # Decode
        decoded = self.decoder(encoded_activated)
        
        return decoded, encoded_activated
    
    def get_activation_stats(self):
        """Return activation statistics for analysis"""
        return self.activation_stats
    
    def get_feature_visualizations(self):
        """Return encoder weights as feature visualizations"""
        return self.encoder.weight.data.view(self.hidden_dim, -1)
    
    def check_reconstruction(self, x):
        """Check reconstruction quality for input"""
        with torch.no_grad():
            decoded, _ = self.forward(x)
            recon_error = F.mse_loss(decoded, x)
        return recon_error.item()
