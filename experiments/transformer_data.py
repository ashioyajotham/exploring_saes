from transformer_lens import HookedTransformer
"""
experiments.transformer_data
=================================

Utilities for extracting and packaging activations from a pretrained
transformer (via TransformerLens / HookedTransformer) into a
PyTorch `Dataset` suitable for training analysis models (e.g. SAEs).

This module provides `TransformerActivationDataset`, a lightweight
dataset wrapper that:

- loads a HookedTransformer model from TransformerLens
- samples/generates input token sequences
- runs the model with the internal cache to collect MLP (or other)
  layer activations
- flattens and caches the activations to speed downstream training

Design goals:
- Simple, reproducible interface for harvesting activations
- Caching to avoid repeated model runs during experiments
- Minimal dependencies so it can be used as a drop-in dataset

Typical usage:

```py
from experiments.transformer_data import TransformerActivationDataset

ds = TransformerActivationDataset(model_name='gpt2-small', layer=0, n_samples=1000)
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
for batch in loader:
    x = batch['pixel_values']  # flattened activations tensor
    # train SAE...
```

Notes:
- The dataset caches activations in memory on creation; for large
  n_samples or very high-dimension layers you may want to modify the
  class to stream activations to disk instead.
- The default token generation here is random; replace with real
  data generation if you want meaningful semantics.
"""

from transformer_lens import HookedTransformer
import torch
from torch.utils.data import Dataset


class TransformerActivationDataset(Dataset):
    """
    Dataset of flattened activations harvested from a HookedTransformer.

    Args:
        model_name (str): Pretrained model id accepted by HookedTransformer.
        layer (int): Index of the transformer layer to harvest activations
            from (e.g. the MLP output). The implementation uses the cache
            key `'mlp_out'` by default.
        n_samples (int): Number of random input samples to generate and
            harvest activations for. The activations are collected and
            cached in memory as a tensor of shape `(n_samples, features)`.

    Attributes:
        model (HookedTransformer): Loaded transformer model.
        layer (int): Layer index used for harvesting.
        n_samples (int): Number of samples harvested.
        activations (torch.Tensor): Cached activation tensor.
    """

    def __init__(self, model_name="gpt2-small", layer=0, n_samples=1000):
        self.model = HookedTransformer.from_pretrained(model_name)
        self.layer = layer
        self.n_samples = n_samples
        self.activations = self._collect_activations()
        
    def _collect_activations(self):
        """
        Run the transformer on randomly generated token sequences and
        collect activations from the configured layer.

        Returns:
            torch.Tensor: stacked activations of shape (n_samples, features)
        """
        activations = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                # Generate a short random token sequence. TransformerLens
                # configs expose their vocabulary dimension as `d_vocab`.
                input_ids = torch.randint(0, self.model.cfg.d_vocab, (1, 20))
                _, cache = self.model.run_with_cache(input_ids)
                # 'mlp_out' is a common key for MLP outputs in HookedTransformer
                layer_act = cache["mlp_out", self.layer][0]  # [seq_len, hidden_dim]
                activations.append(layer_act.flatten())
        return torch.stack(activations)
    
    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        """
        Return a single sample as a dict with key `'pixel_values'` so it
        can be used interchangeably with code that expects image-like
        flattened inputs.
        """
        return {"pixel_values": self.activations[idx]}
