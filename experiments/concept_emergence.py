import torch
import numpy as np
from typing import Dict, Any
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

class ConceptAnalyzer:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.concepts = {}
    
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
