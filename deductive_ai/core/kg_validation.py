# Validates the KG
from typing import List
from rdflib import Graph, URIRef

from deductive_reasoning_framework.deductive_ai.core.ontology_manager import OntologyManager


class KnowledgeGraphValidator:
    @staticmethod
    def validate_triples(graph: Graph, ontology: OntologyManager) -> List[str]:
        errors = []
        for s, p, o in graph:
            # Check if predicate exists in ontology
            if not ontology.validate_property(str(p).split("/")[-1]):
                errors.append(f"Invalid property: {p}")
                
            # Check object type constraints
            if isinstance(o, URIRef) and not ontology.validate_class(str(o).split("/")[-1]):
                errors.append(f"Invalid class: {o}")
                
        return errors