from transformer_lens import HookedTransformer
import torch
from torch.utils.data import Dataset, DataLoader

class TransformerActivationDataset(Dataset):
    def __init__(self, model_name="gpt2-small", layer=0, n_samples=1000):
        self.model = HookedTransformer.from_pretrained(model_name)
        self.layer = layer
        self.n_samples = n_samples
        self.activations = self._collect_activations()
        
    def _collect_activations(self):
        activations = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                # Generate random sequence
                input_ids = torch.randint(0, self.model.cfg.vocab_size, (1, 20))
                # Get activations from specified layer
                _, cache = self.model.run_with_cache(input_ids)
                layer_act = cache["mlp_out", self.layer][0]  # Shape: [seq_len, hidden_dim]
                activations.append(layer_act.flatten())
        return torch.stack(activations)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {"pixel_values": self.activations[idx]}
