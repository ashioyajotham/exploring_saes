from experiments.activation_study import run_activation_comparison
from experiments.frequency_analysis import FrequencyAnalyzer
from experiments.concept_emergence import ConceptAnalyzer

def run_full_analysis(config):
    # Run activation comparison
    activation_results = run_activation_comparison(config)
    
    # Initialize analyzers
    model = train_model(config)  # Train base model
    frequency_analyzer = FrequencyAnalyzer(model)
    concept_analyzer = ConceptAnalyzer(model, get_dataset())
    
    # Analyze
    freq_stats = frequency_analyzer.analyze()
    concept_stats = concept_analyzer.analyze_concepts()
    
    return {
        'activation_comparison': activation_results,
        'frequency_analysis': freq_stats,
        'concept_analysis': concept_stats
    }
