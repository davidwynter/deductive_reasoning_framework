import ray
from ray import tune
import numpy as np

# Hyperparameter Tuning Class
class HyperParameterTuning:
    def __init__(self, engine, dataset, expected_triples):
        self.engine = engine
        self.dataset = dataset
        self.expected_triples = expected_triples

    def objective(self, config):
        """
        Objective function for Ray Tune.
        Config contains the percentage contribution of each ML model.
        """
        weights = {
            "Bayesian Networks": config["bayesian"],
            "Probabilistic Programming": config["pymc"],
            "Monte Carlo Methods": config["pyro"],
            "Markov Logic Networks": config["pyreason"]
        }
        self.engine.set_confidence_weights(weights)
        self.engine.run_inference(self.dataset)
        matched, _ = self.engine.get_validation_results()
        return matched  # The goal is to maximize the percentage of matched triples

    def run_tuning(self, ranges):
        """
        Run the hyperparameter tuning with Ray Tune.
        """
        search_space = {
            "bayesian": tune.choice(ranges["Bayesian Networks"]),
            "pymc": tune.choice(ranges["Probabilistic Programming"]),
            "pyro": tune.choice(ranges["Monte Carlo Methods"]),
            "pyreason": tune.choice(ranges["Markov Logic Networks"])
        }

        def tune_wrapper(config):
            # Ensure the weights add up to 100%
            if sum(config.values()) == 100:
                return self.objective(config)
            else:
                return -np.inf  # Invalid configuration

        analysis = tune.run(
            tune_wrapper,
            config=search_space,
            num_samples=100,
            metric="matched",
            mode="max",
            verbose=1
        )

        return analysis.best_config, analysis.best_result