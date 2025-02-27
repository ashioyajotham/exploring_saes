"""
Concept Emergence Analysis Module
================================

This module analyzes how concepts emerge and evolve in Sparse Autoencoders (SAEs).
It tracks feature learning, identifies concept clusters, and measures relationships
between learned representations.

Key Components:
- Feature clustering using K-means
- Similarity analysis between features
- Dimensionality reduction for visualization
- Activation pattern analysis
- Concept attribution scoring

Usage:
------
```python
model = SparseAutoencoder(input_dim=784, hidden_dim=256)
dataset = load_dataset('mnist')
analyzer = ConceptAnalyzer(model, dataset)
concept_stats = analyzer.analyze_concepts()
```
"""

import torch
import numpy as np
from typing import Dict, Any
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from torch.nn import functional as F


class ConceptAnalyzer:
    """
    Analyzes concept emergence in Sparse Autoencoders.
    
    Attributes:
        model: Trained SAE model
        dataset: Dataset of transformer activations
        concepts: Dictionary storing identified concepts
        activation_type: Type of activation function used
        
    Methods:
        analyze_concepts(): Run full concept analysis suite
        cluster_features(): Group similar features into concepts
        compute_similarities(): Calculate feature similarity metrics
        get_feature_embeddings(): Extract feature space representations
    """

    def __init__(self, model, dataset):
        """
        Initialize concept analyzer.

        Args:
            model: Trained SAE model to analyze
            dataset: Dataset containing transformer activations
        """
        self.model = model
        self.dataset = dataset
        self.concepts = {}
        self.activation_type = model.activation_type

    def analyze_concepts(self) -> Dict[str, Any]:
        """
        Run comprehensive concept analysis.
        
        Returns:
            Dictionary containing:
            - feature_clusters: Grouped similar features
            - concept_similarity: Inter-feature similarity metrics
            - embedding: Feature space embeddings
            - attribution_scores: Concept attribution measures
        """
        return {
            'feature_clusters': self.cluster_features(),
            'concept_similarity': self.compute_similarities(),
            'embedding': self.get_feature_embeddings(),
            'attribution': self.compute_attribution_scores()
        }

    def cluster_features(self) -> Dict[str, Any]:
        """
        Group similar features into concept clusters using K-means.
        
        Returns:
            Dictionary containing cluster assignments and centroids
        """
        pca = PCA(n_components=3)
        projected = pca.fit_transform(self._extract_features())
        
        return {
            'principal_components': pca.components_,
            'explained_variance': pca.explained_variance_ratio_,
            'feature_clusters': self._get_clusters(projected)
        }

    def compute_similarities(self) -> Dict[str, float]:
        """
        Calculate similarity metrics between learned features.
        
        Returns:
            Dictionary containing mean, max, and min similarities
        """
        features = self._extract_features()
        similarity_matrix = cosine_similarity(features)
        return {
            'mean_similarity': float(np.mean(similarity_matrix)),
            'max_similarity': float(np.max(similarity_matrix - np.eye(len(features)))),
            'similarity_distribution': np.percentile(similarity_matrix.flatten(), [25, 50, 75]).tolist()
        }

    def get_feature_embeddings(self) -> torch.Tensor:
        """
        Extract and process feature embeddings from model.
        
        Returns:
            Tensor of feature embeddings
        """
        with torch.no_grad():
            features = self.model.encoder.weight.data
            return features

    def _extract_features(self):
        """Extract learned features from encoder weights"""
        return self.model.encoder.weight.detach().cpu()
    
    def _get_clusters(self, features, n_clusters=5):
        """Cluster features using K-means"""
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(features)
        return {
            'labels': clusters.tolist(),
            'centers': kmeans.cluster_centers_.tolist(),
            'inertia': float(kmeans.inertia_),
            'sizes': np.bincount(clusters).tolist()
        }
