from transformers import AutoModel, AutoTokenizer
import torch

class ModelLoader:
    SUPPORTED_MODELS = {
        "bert": "bert-base-uncased",
        "gpt2": "gpt2",
        "roberta": "roberta-base"
    }

    def __init__(self, model_name: str):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_name = self.SUPPORTED_MODELS[model_name]
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.cached_activations = None
        
    def get_layer_activations(self, text: str, layer_idx: int = -1) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get activations from specified layer
        hidden_states = outputs.hidden_states[layer_idx]
        return hidden_states
        
    def get_activations(self, text: str = None, layer_idx: int = -1):
        """Get model activations for input text"""
        if text is None and self.cached_activations is None:
            raise ValueError("Please provide text for analysis")
            
        if text is not None:
            activations = self.get_layer_activations(text, layer_idx)
            self.cached_activations = activations
        
        return self.cached_activations
