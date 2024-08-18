import random
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import pymc as pm
import pyro 
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
from owlready2 import *
from rdflib import Graph, Namespace


class DeductiveReasoningEngine:
    def __init__(self, ml_model=None, integration_method="combine_confidence", confidence_weights=None):
        self.graph = Graph()
        self.rules = ""
        self.validation_results = []
        self.confidence_scores = {}
        self.ml_model = ml_model
        self.integration_method = integration_method
        self.confidence_weights = confidence_weights if confidence_weights else {"bayesian": 1.0}
        self.onto = None
        self.inferred_graph = set()
        self.pyro_model = self.define_pyro_model()  # Assuming you have a Pyro model defined
        # Initialize probabilistic models based on the confidence methods provided
        if "bayesian" in self.confidence_weights:
            self.bayesian_network = self.create_bayesian_network()
        if "pymc3" in self.confidence_weights:
            self.pymc3_model = self.create_pymc3_model()
        if "pyro" in self.confidence_weights:
            self.pyro_model = self.create_pyro_model()

    def create_pyro_model(self, variables, relationships):
        """
        Define a generalized Pyro model using user-defined variables and relationships.
        """
        def model():
            params = {}
            for var in variables:
                params[var] = pyro.sample(f"{var}_rate", dist.Beta(2, 5))
            for parent, child in relationships:
                with pyro.plate(f"data_{child}", size=100):
                    pyro.sample(f"obs_{child}", dist.Bernoulli(params[parent]))
            return params

        self.pyro_model = model

    def create_bayesian_network(self, variables, relationships, cpds):
        """
        Create a generalized Bayesian Network using user-defined variables and relationships.
        """
        model = BayesianNetwork(relationships)
        for cpd in cpds:
            model.add_cpds(cpd)
        self.bayesian_network = model

    def create_pymc3_model(self, variables, relationships):
        """
        Create a generalized PyMC3 model using user-defined variables and relationships.
        """
        with pm.Model() as model:
            params = {}
            for var in variables:
                params[var] = pm.Bernoulli(var, p=0.5)
            for parent, child in relationships:
                params[child] = pm.Bernoulli(child, p=params[parent] * 0.8 + (1 - params[parent]) * 0.2)
            self.pymc3_model = model
            
    def load_ontology(self, ontology_path):
        """
        Load the ontology from a given path.
        """
        self.onto = get_ontology(ontology_path).load()

    def load_rules(self, rules_text, rule_format="n3"):
        """
        Load reasoning rules from text.
        Currently, SWRL is the main supported rule format.
        """
        self.rules = rules_text

        # Parse and apply rules using OWLReady2
        with self.onto:
            rule = Imp()
            rule.set_as_rule(self.rules)

    def apply_reasoning(self):
        """
        Apply the loaded reasoning rules to the graph and generate inferred triples with confidence scores.
        """
        if not self.onto:
            raise ValueError("Ontology not loaded. Please load an ontology first.")
        
        # Sync reasoner to apply the rules and infer new facts
        sync_reasoner()

        # Collect inferred triples
        for subj in self.onto.individuals():
            for obj in self.onto.individuals():
                for prop in self.onto.properties():
                    if prop[subj] and obj in prop[subj]:
                        fact = (subj.iri, prop.iri, obj.iri)
                        confidence = self.estimate_confidence(fact)
                        if hasattr(self, 'ml_model') and self.ml_model:
                            ml_confidence = self.apply_ml_model(fact)
                            confidence = self.combine_confidences(confidence, ml_confidence)
                        self.inferred_graph.add(fact)
                        self.confidence_scores[fact] = confidence

    def estimate_confidence(self, fact):
        """
        Estimate confidence based on the weighted combination of methods.
        """
        total_weight = sum(self.confidence_weights.values())
        combined_confidence = 0.0

        if "bayesian" in self.confidence_weights:
            bayesian_confidence = self.estimate_confidence_bayesian(fact)
            combined_confidence += self.confidence_weights["bayesian"] * bayesian_confidence
        
        if "pymc3" in self.confidence_weights:
            pymc3_confidence = self.estimate_confidence_pymc3(fact)
            combined_confidence += self.confidence_weights["pymc3"] * pymc3_confidence
        
        if "pyro" in self.confidence_weights:
            pyro_confidence = self.estimate_confidence_pyro(fact)
            combined_confidence += self.confidence_weights["pyro"] * pyro_confidence

        return combined_confidence / total_weight if total_weight > 0 else random.uniform(0.5, 1.0)

    def estimate_confidence_bayesian(self, query_var, evidence_var, evidence_value):
        """
        Estimate confidence using Bayesian Networks with generalized variables.
        """
        inference = VariableElimination(self.bayesian_network)
        query = inference.query(variables=[query_var], evidence={evidence_var: evidence_value})
        return query.values.max()

    def estimate_confidence_pymc3(self, query_var):
        """
        Estimate confidence using PyMC3 with generalized variables.
        """
        with self.pymc3_model:
            trace = pm.sample(1000, return_inferencedata=False)
        return trace[query_var].mean()

    def estimate_confidence_pyro(self, query_var):
        """
        Estimate confidence using Pyro with generalized variables.
        """
        # Clear the parameter store to reset any previous parameter values
        pyro.clear_param_store()

        # Define the model and guide (variational distribution)
        svi = SVI(
            model=self.pyro_model,
            guide=AutoDiagonalNormal(self.pyro_model),
            optim=pyro.optim.Adam({"lr": 0.01}),
            loss=Trace_ELBO()
        )

        # Perform stochastic variational inference
        for _ in range(1000):
            svi.step()

        # Extract the posterior distribution of the variable of interest
        param_name = f"auto_{query_var}_rate_loc"
        rate = pyro.param(param_name).item()

        return rate

    def apply_ml_model(self, fact):
        """
        Placeholder for applying the machine learning model.
        """
        return random.uniform(0.5, 1.0)

    def combine_confidences(self, deductive_confidence, ml_confidence):
        """
        Combine confidence scores from deductive reasoning and the ML model.
        """
        if self.integration_method == "combine_confidence":
            return (deductive_confidence + ml_confidence) / 2
        elif self.integration_method == "weighted_combine":
            return 0.7 * ml_confidence + 0.3 * deductive_confidence
        elif self.integration_method == "ml_override":
            return ml_confidence if ml_confidence else deductive_confidence
        else:
            return deductive_confidence

    # Other methods (validate_inferences, iterate_rules, etc.) remain unchanged

# Example usage of the generalized framework with weighted confidence estimation
def example_usage(base_url = "http://example.org/"):
    # Define variables and relationships for a new domain, e.g., "Weather"
    variables = ["Rain", "WetGrass"]
    relationships = [("Rain", "WetGrass")]
    cpds = [
        TabularCPD(variable='Rain', variable_card=2, values=[[0.7], [0.3]]),
        TabularCPD(variable='WetGrass', variable_card=2, values=[[0.1, 0.9], [0.9, 0.1]], evidence=['Rain'], evidence_card=[2])
    ]

    engine = DeductiveReasoningEngine()

    # Create the models
    engine.create_pyro_model(variables, relationships)
    engine.create_bayesian_network(variables, relationships, cpds)
    engine.create_pymc3_model(variables, relationships)

    # Estimate confidence using different models
    confidence_bayesian = engine.estimate_confidence_bayesian("WetGrass", "Rain", 1)
    confidence_pymc3 = engine.estimate_confidence_pymc3("WetGrass")
    confidence_pyro = engine.estimate_confidence_pyro("WetGrass")

    print(f"Bayesian confidence: {confidence_bayesian}")
    print(f"PyMC3 confidence: {confidence_pymc3}")
    print(f"Pyro confidence: {confidence_pyro}")
    EX = Namespace(base_url)
    
    # Initialize the reasoning engine with weighted confidence methods
    confidence_weights = {"bayesian": 0.4, "pymc3": 0.3, "pyro": 0.2, "pyreason": 0.1}
    engine = DeductiveReasoningEngine(confidence_weights=confidence_weights)
    
    engine.load_data('data.ttl', format='ttl')
    engine.load_rules(f'''
    @prefix ex: <{base_url}/>.
    {{ ?patient ex:hasSymptom ?symptom .
      ?disease ex:hasSymptom ?symptom .
    }} => {{ ?patient ex:mayHave ?disease . }}.
    ''')
    engine.apply_reasoning()
    
    # Validate and evaluate
    known_outcomes = {(EX.Patient1, EX.mayHave, EX.DiseaseX)}
    engine.validate_inferences(known_outcomes)
    accuracy, avg_confidence = engine.evaluate_performance()
    print(f"Weighted Method - Accuracy: {accuracy * 100:.2f}%, Average Confidence: {avg_confidence:.2f}")
    
    # Retrieve the inferred graph
    inferred_graph = engine.get_inferred_graph()
    for triple in inferred_graph:
        confidence = engine.confidence_scores.get(triple, 1.0)
        print(f"{triple} with confidence {confidence:.2f}")



if __name__ == "__main__":
    example_usage()
