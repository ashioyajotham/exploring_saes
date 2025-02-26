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
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.concepts = {}
        self.activation_type = model.activation_type
    
    def analyze_concepts(self) -> Dict[str, Any]:
        features = self._extract_features()
        
        return {
            'feature_clusters': self._cluster_features(features),
            'concept_similarity': self._analyze_concept_similarity(features),
            'semantic_structure': self._analyze_semantic_structure(features),
            'emergence_patterns': self._analyze_emergence_patterns()
        }
    
    def _extract_features(self):
        """Extract learned features from encoder weights"""
        return self.model.encoder.weight.detach().cpu()
    
    def _cluster_features(self, features):
        pca = PCA(n_components=3)
        projected = pca.fit_transform(features)
        
        return {
            'principal_components': pca.components_,
            'explained_variance': pca.explained_variance_ratio_,
            'feature_clusters': self._get_clusters(projected)
        }
    
    def _analyze_concept_similarity(self, features):
        similarity_matrix = cosine_similarity(features)
        return {
            'mean_similarity': float(np.mean(similarity_matrix)),
            'max_similarity': float(np.max(similarity_matrix - np.eye(len(features)))),
            'similarity_distribution': np.percentile(similarity_matrix.flatten(), [25, 50, 75]).tolist()
        }
    
    def _analyze_semantic_structure(self, features):
        """Analyze semantic relationships between learned features"""
        # Compute pairwise distances
        distances = pdist(features.numpy(), metric='cosine')
        distance_matrix = squareform(distances)
        
        # Dimensionality reduction for visualization
        tsne = TSNE(n_components=2, random_state=42)
        embedded = tsne.fit_transform(features)
        
        return {
            'semantic_distances': {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances))
            },
            'embedding': embedded.tolist(),
            'clusters': self._identify_semantic_groups(distance_matrix)
        }
        
    def _identify_semantic_groups(self, distance_matrix, threshold=0.5):
        """Identify groups of semantically related features"""
        similar_pairs = np.where(distance_matrix < threshold)
        groups = []
        for i, j in zip(*similar_pairs):
            if i < j:  # Avoid duplicates
                groups.append((int(i), int(j)))
        return groups
    
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
    
    def _analyze_emergence_patterns(self):
        """Analyze how concepts emerge and develop during training"""
        features = self._extract_features()
        
        return {
            'feature_norms': torch.norm(features, dim=1).tolist(),
            'specialization': self._compute_specialization(features),
            'layer_statistics': {
                'mean_magnitude': float(features.abs().mean()),
                'sparsity': float((features.abs() < 0.1).float().mean()),
                'peak_responses': torch.max(features, dim=1)[0].tolist()
            }
        }
    
    def _compute_specialization(self, features):
        """Compute how specialized each feature is"""
        cosine_sim = F.cosine_similarity(features.unsqueeze(1), 
                                       features.unsqueeze(0), dim=2)
        specialization = 1 - (cosine_sim.sum(1) - 1) / (len(features) - 1)
        return specialization.tolist()
