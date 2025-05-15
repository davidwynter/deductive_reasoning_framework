import sys
import os

# Add the parent directory of 'deductive_reasoning_framework' to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

import random
from typing import Dict, Tuple, List, Optional, Any, Set
from rdflib import Namespace, Graph

from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from sklearn import tree
import pymc as pm
import pyro 
import pyro.distributions as dist
import pyro.optim
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
from owlready2 import Ontology, Imp, get_ontology, sync_reasoner, ObjectProperty, Thing, Restriction, entity
from deductive_ai.engine.confidence_model import ConfidenceAdjuster
from deductive_ai.core.text_to_rdf import TextToRDFConverter


class DeductiveReasoningEngine:
    def __init__(self, ml_model: Optional[Any] = None, integration_method: str = "combine_confidence", 
                 confidence_weights: Optional[Dict[str, float]] = None):
        self.graph = None  # Initialize as None
        self.rules = ""
        self.validation_results = []
        self.confidence_scores: Dict[Tuple, float] = {}
        self.ml_model = ml_model
        self.integration_method = integration_method
        self.confidence_weights = confidence_weights if confidence_weights else {"bayesian": 1.0}
        self.onto: Optional[Ontology] = None
        self.inferred_graph: Set[Tuple] = set()
        self.nl_convertor = None
        
        # Initialize ConfidenceAdjuster only if ml_model is provided
        self.confidence_adjuster = ConfidenceAdjuster(ml_model) if ml_model else None
        
        # Initialize probabilistic models based on the confidence methods provided
        self.bayesian_network = None
        self.pymc3_model = None
        self.pyro_model = None

        self.graph = Graph()
        
    def populate_from_texts(self, text_sources: List[str]):
        """Populates KG from unstructured texts"""
        converter = TextToRDFConverter(self.onto)
        new_triples = converter.populate_kg(text_sources)
        self.graph += new_triples
        
        # Validate against ontology
        validation_errors = self._validate_new_triples(new_triples)
        if validation_errors:
            raise ValueError(f"Ontology violations: {validation_errors}")

    def create_pyro_model(self, variables: List[str], relationships: List[Tuple[str, str]]):
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
        
    def get_variables_from_ontology(self, class_name_patterns=None):
        """
        Extract variables (classes) from the ontology based on provided class name patterns.

        Parameters:
        - class_name_patterns: List of strings to match in class names. Defaults to ["Disease", "Symptom"].

        Returns:
        - variables: List of class IRIs matching the patterns.
        """
        variables = []

        # Extract classes of interest based on patterns
        for cls in self.onto.classes():
            if any(pattern in cls.name for pattern in class_name_patterns):
                variables.append(cls.iri)

        return variables

    def get_relationships_from_ontology(self, source_class_pattern, target_class_pattern, object_property_name):
        """
        Extract relationships from the ontology based on the specified ObjectProperty and class patterns.

        Parameters:
        - source_class_pattern: String pattern to match in source class names.
        - target_class_pattern: String pattern to match in target class names.
        - object_property_name: Name of the ObjectProperty to use for relationships.

        Returns:
        - relationships: List of tuples (source_instance_uri, target_instance_uri) representing relationships.
        """
        relationships = []

        # Get the ObjectProperty from the ontology
        object_property = self.onto.search_one(label=object_property_name)
        if not object_property:
            raise ValueError(f"ObjectProperty '{object_property_name}' not found in ontology.")

        src_name = self.onto.search_one(label=object_property.domain[0].name)
        dest_name = self.onto.search_one(label=object_property.domain[1].name)
        relationships.append((src_name, dest_name))
        
        return relationships
    
    def create_bayesian_network(self, variables, relationships, cpds, latent_variables=None):
        """
        Create a Bayesian Network using variables and relationships extracted from the ontology.
        """

        # Create the Bayesian Network
        model = BayesianNetwork(ebunch=relationships, latents=latent_variables)
        # Add any isolated variables (nodes not connected by edges)
        for var in variables:
            if var not in model.nodes():
                model.add_node(var)

        # Add CPDs to the model
        model.add_cpds(*cpds)

        # Verify the model
        model.check_model()

        self.bayesian_network = model


    def create_pymc3_model(self, variables: List[str], relationships: List[Tuple[str, str]]):
        """
        Create a generalized PyMC3 model using user-defined variables and relationships.
        """
        with pm.Model() as model:
            params = {}
            # Create Bernoulli variables for all independent variables (those that are not children in relationships)
            for var in variables:
                if var not in [child for _, child in relationships]:
                    params[var] = pm.Bernoulli(var, p=0.5)

            # Create child variables based on parent-child relationships
            for parent, child in relationships:
                if child not in params:  # Ensure child is not already created
                    # Define the child variable with dependency on the parent variable
                    params[child] = pm.Bernoulli(child, p=params[parent] * 0.8 + (1 - params[parent]) * 0.2)
            self.pymc3_model = model
            
    def load_ontology(self, ontology_path: str):
        """
        Load the ontology from a given path.
        """
        try:
            return get_ontology(ontology_path).load()
        except Exception as e:
            raise ValueError(f"Failed to load ontology from {ontology_path}: {str(e)}")

    def load_rules(self, rules_text: str):
        """
        Load reasoning rules from text.
        Currently, SWRL is the main supported rule format.
        """
        self.rules = rules_text

        if not self.onto:
            raise ValueError("Ontology not loaded. Please load an ontology first.")

        try:
            # Parse and apply rules using OWLReady2
            with self.onto:
                rule = Imp()
                rule.set_as_rule(self.rules)
        except Exception as e:
            raise ValueError(f"Failed to load rules: {str(e)}")

    def load_data(self, data_path: str):
        """
        Load data into the graph.
        """
        try:
            if self.graph is None:
                filename_with_extension = os.path.basename(data_path)
                filename, _ = os.path.splitext(filename_with_extension)
                self.graph = Graph() 
            self.graph.parse(data_path)
        except Exception as e:
            raise ValueError(f"Failed to load data from {data_path}: {str(e)}")

    def apply_reasoning(self):
        """
        Apply the loaded reasoning rules to the graph and generate inferred triples with confidence scores.
        """
        if not self.onto:
            raise ValueError("Ontology not loaded. Please load an ontology first.")
        
        try:
            # Sync reasoner to apply the rules and infer new facts
            sync_reasoner()

            # Collect inferred triples
            for subj in self.onto.individuals():
                for obj in self.onto.individuals():
                    for prop in self.onto.properties():
                        if prop[subj] and obj in prop[subj]:
                            fact = (subj.iri, prop.iri, obj.iri)
                            confidence = self.estimate_confidence(fact)
                            if self.confidence_adjuster:
                                ml_confidence = self.confidence_adjuster.apply_ml_model(fact)
                                confidence = self.confidence_adjuster.combine_confidences(confidence, ml_confidence)
                            self.inferred_graph.add(fact)
                            self.confidence_scores[fact] = confidence
        except Exception as e:
            raise RuntimeError(f"Error during reasoning: {str(e)}")

    def estimate_confidence(self, fact: Tuple) -> float:
        """
        Estimate confidence based on the weighted combination of methods.
        """
        total_weight = sum(self.confidence_weights.values())
        if total_weight == 0:
            return random.uniform(0.5, 1.0)

        combined_confidence = 0.0

        for method, weight in self.confidence_weights.items():
            if method == "bayesian" and self.bayesian_network:
                confidence = self.estimate_confidence_bayesian(fact)
            elif method == "pymc3" and self.pymc3_model:
                confidence = self.estimate_confidence_pymc3(fact)
            elif method == "pyro" and self.pyro_model:
                confidence = self.estimate_confidence_pyro(fact)
            else:
                continue
            combined_confidence += weight * confidence

        return combined_confidence / total_weight

    def estimate_confidence_bayesian(self, fact: Tuple) -> float:
        """
        Estimate confidence using Bayesian Networks with generalized variables.
        """
        # This is a simplified example. In practice, you'd need to map the fact to your Bayesian network structure.
        query_var, evidence_var, evidence_value = fact
        inference = VariableElimination(self.bayesian_network)
        query = inference.query(variables=[query_var], evidence={evidence_var: evidence_value})
        return query.values.max()

    def estimate_confidence_pymc3(self, fact: Tuple) -> float:
        """
        Estimate confidence using PyMC3 with generalized variables.
        """
        # This is a simplified example. In practice, you'd need to map the fact to your PyMC3 model structure.
        query_var = fact[0]
        with self.pymc3_model:
            trace = pm.sample(1000, return_inferencedata=False)
        return trace[query_var].mean()

    def estimate_confidence_pyro(self, fact: Tuple) -> float:
        """
        Estimate confidence using Pyro with generalized variables.
        """
        # This is a simplified example. In practice, you'd need to map the fact to your Pyro model structure.
        query_var = fact[0]
        
        pyro.clear_param_store()

        svi = SVI(
            model=self.pyro_model,
            guide=AutoDiagonalNormal(self.pyro_model),
            optim=pyro.optim.Adam({"lr": 0.01}),
            loss=Trace_ELBO()
        )

        for _ in range(1000):
            svi.step()

        param_name = f"auto_{query_var}_rate_loc"
        rate = pyro.param(param_name).item()

        return rate

    def validate_inferences(self, known_outcomes: Set[Tuple]):
        """
        Validate inferred triples against known outcomes.
        """
        self.validation_results = []
        for outcome in known_outcomes:
            if outcome in self.inferred_graph:
                self.validation_results.append((outcome, True, self.confidence_scores.get(outcome, 1.0)))
            else:
                self.validation_results.append((outcome, False, 0.0))

    def evaluate_performance(self) -> Tuple[float, float]:
        """
        Evaluate the performance of the reasoning engine.
        """
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate_inferences first.")

        correct_inferences = sum(1 for _, is_correct, _ in self.validation_results if is_correct)
        total_inferences = len(self.validation_results)
        accuracy = correct_inferences / total_inferences if total_inferences > 0 else 0

        total_confidence = sum(confidence for _, _, confidence in self.validation_results)
        avg_confidence = total_confidence / total_inferences if total_inferences > 0 else 0

        return accuracy, avg_confidence

    def get_inferred_graph(self) -> Set[Tuple]:
        """
        Retrieve the inferred graph.
        """
        return self.inferred_graph
        
    def get_datasets(self) -> List[str]:
        """
        Get a list of available datasets.
        This is a placeholder method that returns a list of example datasets.
        In a real implementation, this would scan a directory or database for available datasets.
        """
        # Create data_management directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_management")
        os.makedirs(data_dir, exist_ok=True)
        
        # Return a list of example datasets
        return ["example_dataset", "disease_dataset", "symptom_dataset"]
        
    def get_expected_triples(self, dataset: str) -> List[Tuple]:
        """
        Get the expected triples for a dataset.
        This is a placeholder method that returns a list of example triples.
        In a real implementation, this would load the expected triples from a file or database.
        """
        # Return a list of example triples
        return [
            ("http://example.org/Patient1", "http://example.org/hasSymptom", "http://example.org/Fever"),
            ("http://example.org/Patient1", "http://example.org/mayHave", "http://example.org/Flu")
        ]

    def train_ml_model(self, training_data: List[Tuple[Tuple, float]], epochs: int = 100):
        """
        Train the ML model for confidence adjustment.
        """
        if not self.confidence_adjuster:
            raise ValueError("ConfidenceAdjuster not initialized. Provide an ML model in the constructor.")
        self.confidence_adjuster.train(training_data, epochs)

    def save_ml_model(self, path: str):
        """
        Save the trained ML model.
        """
        if not self.confidence_adjuster:
            raise ValueError("ConfidenceAdjuster not initialized. Provide an ML model in the constructor.")
        self.confidence_adjuster.save_model(path)

    def load_ml_model(self, path: str):
        """
        Load a pre-trained ML model.
        """
        if not self.confidence_adjuster:
            raise ValueError("ConfidenceAdjuster not initialized. Provide an ML model in the constructor.")
        self.confidence_adjuster.load_model(path)
        
    def set_confidence_weights(self, weights):
        """
        Set the confidence weights for different reasoning methods.
        
        Args:
            weights: Dictionary mapping method names to their weights (percentages)
        """
        # Convert the weights to a normalized form (sum to 1.0)
        total = sum(weights.values())
        normalized_weights = {k: v / total for k, v in weights.items()}
        
        # Update the confidence weights
        self.confidence_weights = normalized_weights
        
        print(f"Confidence weights updated: {self.confidence_weights}")
        
    def initialize_nl_converter(self, ontology_path):
        from nl_to_swrl import NLToSWRLConverter
        self.nl_converter = NLToSWRLConverter(ontology_path)
        
    def convert_nl_to_swrl(self, nl_rule: str) -> str:
        if not self.nl_converter:
            raise ValueError("NL converter not initialized with ontology")
        return self.nl_converter.convert(nl_rule)
    
    def validate_swrl_rule(self, rule_text: str) -> dict:
        validation_result = {
            "valid": False,
            "message": "",
            "suggestions": []
        }
        
        try:
            # Parse rule components
            parsed = self._parse_swrl_components(rule_text)
            
            # Validate ontology elements
            for cls in parsed["classes"]:
                if not self.ontology.validate_class(cls):
                    validation_result["suggestions"].append(
                        f"Did you mean: {self.ontology.suggest_similar_class(cls)}?"
                    )
            
            # Add more validation checks
            validation_result["valid"] = True
            return validation_result
            
        except Exception as e:
            validation_result["message"] = str(e)
            return validation_result

# Example usage of the generalized framework with weighted confidence estimation
def example_usage(base_url: str = "http://example.org/"):
    # Define variables and relationships for the disease domain
    variables = ["http://purl.obolibrary.org/obo/ogms.owl#Symptom", "http://purl.obolibrary.org/obo/ogms.owl#Disease"]
    relationships = [("http://purl.obolibrary.org/obo/ogms.owl#Symptom", "http://purl.obolibrary.org/obo/ogms.owl#Disease")]
    cpds = [
        TabularCPD(variable='http://purl.obolibrary.org/obo/ogms.owl#Symptom', variable_card=2, values=[[0.7], [0.3]]),
        TabularCPD(variable='http://purl.obolibrary.org/obo/ogms.owl#Disease', variable_card=2, values=[[0.1, 0.9], [0.9, 0.1]], evidence=['Symptom'], evidence_card=[2])
    ]

    # Initialize the reasoning engine with weighted confidence methods
    confidence_weights = {"bayesian": 0.4, "pymc3": 0.3, "pyro": 0.2}
    engine = DeductiveReasoningEngine(confidence_weights=confidence_weights)
    
    # Load ontology and rules
    # Get the directory two levels above the current file
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

    # Define the paths to the files
    data_path = os.path.join(parent_dir, 'deductive_reasoning_framework', 'data', 'disease_data.n3')

    # Load the ontologies
    engine.onto = engine.load_ontology("http://purl.obolibrary.org/obo/ogms.owl").load()
    
    engine.load_data(data_path)
    # Access classes
    Patient = engine.onto.search_one(label="Patient")
    Disease = engine.onto.search_one(label="Disease")
    Symptom = engine.onto.search_one(label="Symptom")

    # Access properties
    hasSymptom = engine.onto.search_one(label="hasSymptom")
    mayHave = engine.onto.search_one(label="mayHave")

    with engine.onto:
        # Define classes if they don't exist
        if not Patient:
            class Patient(Thing):
                name = 'Patient'
                pass

        if not Disease:
            class Disease(Thing):
                name = 'Disease'
                pass

        if not Symptom:
            class Symptom(Thing):
                name = 'Symptom'
                pass

        # Define properties if they don't exist
        if not hasSymptom:
            class hasSymptom(ObjectProperty):
                domain = [Patient, Disease]
                name = 'hasSympton'
                range = [Symptom]
                label = "has symptom"

        if not mayHave:
            class mayHave(ObjectProperty):
                domain = [Patient]
                name = 'mayHave'
                range = [Disease]
                label = "may have"

    object_properties = {
        "hasSymptom": {"domain": [Disease, Patient], "range": [Symptom], "label": "hasSymptom"},
        "mayHave": {"domain": [Patient], "range": [Disease], "label": "mayHave"}
    }

    with engine.onto:
        for prop_name, prop_attrs in object_properties.items():
            prop = engine.onto.search_one(label=prop_attrs["label"])
            if not prop:
                # Define the property if it doesn't exist
                class_def = type(prop_name, (ObjectProperty,), {})
                class_def.domain = prop_attrs["domain"]
                class_def.range = prop_attrs["range"]
                class_def.label = prop_attrs["label"]

    # Now load rules as SWRL    
    engine.load_rules('hasSymptom(?patient, ?symptom) ^ hasSymptom(?disease, ?symptom) -> mayHave(?patient, ?disease)')
    
    # Define the class name patterns to extract variables
    class_name_patterns = ["Patient", "Disease", "Symptom"]
    variables = engine.get_variables_from_ontology(class_name_patterns=class_name_patterns)

    # Define parameters for relationships
    source_class_pattern = "Disease"
    target_class_pattern = "Symptom"
    object_property_name = "hasSymptom"
    relationships = engine.get_relationships_from_ontology(
        source_class_pattern=source_class_pattern,
        target_class_pattern=target_class_pattern,
        object_property_name=object_property_name
    )

    # Create CPDs
    cpds = [
        TabularCPD(
            variable='http://purl.obolibrary.org/obo/ogms.owl#Symptom',
            variable_card=2,
            values=[[0.7], [0.3]]
        ),
        TabularCPD(
            variable='http://purl.obolibrary.org/obo/ogms.owl#Disease',
            variable_card=2,
            values=[[0.1, 0.9], [0.9, 0.1]],
            evidence=['http://purl.obolibrary.org/obo/ogms.owl#Symptom'],
            evidence_card=[2]
        )
    ]

    # Create the models
    engine.create_pyro_model(variables, relationships)
    engine.create_bayesian_network(variables=variables, relationships=relationships, cpds=cpds)
    engine.create_pymc3_model(variables, relationships)

    # Apply reasoning
    engine.apply_reasoning()
    
    # Validate and evaluate
    EX = Namespace(base_url)
    known_outcomes = {(EX.Patient1, EX.mayHave, EX.Flu)}
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
