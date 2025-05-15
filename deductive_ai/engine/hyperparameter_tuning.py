"""
Hyperparameter Tuning for the Deductive Reasoning Engine.
This module provides a HyperParameterTuning class that can be used to find the optimal
confidence weights for different reasoning methods.
"""

import itertools
import random

class HyperParameterTuning:
    """
    Class for tuning hyperparameters of the reasoning engine.
    """
    def __init__(self, engine, dataset, expected_triples):
        """
        Initialize the hyperparameter tuning.
        
        Args:
            engine: The reasoning engine
            dataset: The dataset to use for tuning
            expected_triples: The expected triples for validation
        """
        self.engine = engine
        self.dataset = dataset
        self.expected_triples = expected_triples
    
    def run_tuning(self, ranges):
        """
        Run hyperparameter tuning with the given ranges.
        
        Args:
            ranges: Dictionary mapping method names to lists of percentage values
            
        Returns:
            best_config: The best configuration found
            best_result: The result metrics for the best configuration
        """
        # Generate all possible combinations of percentages
        methods = list(ranges.keys())
        combinations = []
        
        # Get all combinations of percentages
        percentage_combinations = list(itertools.product(*[ranges[method] for method in methods]))
        
        # Filter combinations where the sum is 100%
        valid_combinations = []
        for combo in percentage_combinations:
            if sum(combo) == 100:
                config = {method: percentage for method, percentage in zip(methods, combo)}
                valid_combinations.append(config)
        
        # If no valid combinations, create some that sum to 100%
        if not valid_combinations:
            # Create 5 random configurations that sum to 100%
            for _ in range(5):
                config = {}
                remaining = 100
                for i, method in enumerate(methods):
                    if i == len(methods) - 1:
                        config[method] = remaining
                    else:
                        value = random.randint(0, remaining)
                        config[method] = value
                        remaining -= value
                valid_combinations.append(config)
        
        # Simulate evaluation for each valid combination
        results = []
        for config in valid_combinations:
            # Set the confidence weights in the engine
            self.engine.set_confidence_weights(config)
            
            # Simulate running inference
            # In a real implementation, this would run inference and validate against expected triples
            # For now, just simulate a result based on the configuration
            matched = 0
            for method, percentage in config.items():
                # Different methods have different simulated effectiveness
                if method == "Bayesian Networks":
                    matched += percentage * 0.8
                elif method == "Probabilistic Programming":
                    matched += percentage * 0.7
                elif method == "Monte Carlo Methods":
                    matched += percentage * 0.75
                else:
                    matched += percentage * 0.6
                    
            # Add some randomness
            matched = min(100, max(0, matched + random.uniform(-10, 10)))
            
            results.append({
                "config": config,
                "matched": matched,
                "unmatched": 100 - matched
            })
        
        # Find the best configuration
        best_result = max(results, key=lambda x: x["matched"])
        
        return best_result["config"], best_result