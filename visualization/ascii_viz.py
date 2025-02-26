from typing import Dict, Any
import numpy as np

class ASCIIVisualizer:
    @staticmethod
    def print_results(results: Dict[str, Any]):
        """Print experiment results in ASCII art format"""
        # Header
        print("\n" + "═" * 60)
        print("║" + "SPARSE AUTOENCODER ANALYSIS RESULTS".center(58) + "║")
        print("═" * 60)

        # Activation Function Results
        print("\n╔" + "═" * 56 + "╗")
        print("║" + " ACTIVATION FUNCTION COMPARISON ".center(56) + "║")
        print("╠" + "═" * 56 + "╣")
        
        for act_type, stats in results['activation_comparison'].items():
            print("║  " + f"{act_type.upper()}:".ljust(15) + "║")
            print("║    " + f"Final Loss: {stats['final_loss']:.4f}".ljust(52) + "║")
            print("║    " + f"Sparsity: {stats['sparsity']:.4f}".ljust(52) + "║")
            print("║" + "─" * 56 + "║")

        # Frequency Analysis
        print("╠" + "═" * 56 + "╣")
        print("║" + " FREQUENCY ANALYSIS ".center(56) + "║")
        print("╠" + "═" * 56 + "╣")
        freq = results['frequency_analysis']
        print("║  " + f"High Frequency Neurons: {freq['high_freq_neurons']}".ljust(54) + "║")
        print("║  " + f"Mean Activation Rate: {freq['mean_frequencies'].mean():.4f}".ljust(54) + "║")
        
        # Concept Analysis
        print("╠" + "═" * 56 + "╣")
        print("║" + " CONCEPT ANALYSIS ".center(56) + "║")
        print("╠" + "═" * 56 + "╣")
        concept = results['concept_analysis']
        print("║  " + f"Feature Clusters: {len(concept['feature_clusters'])}".ljust(54) + "║")
        print("║  " + f"Semantic Similarity: {concept['concept_similarity']['mean_similarity']:.4f}".ljust(54) + "║")
        
        # Footer
        print("╚" + "═" * 56 + "╝\n")
