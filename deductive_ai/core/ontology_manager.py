from rdflib import Graph, URIRef
from rdflib.namespace import RDF, OWL
import Levenshtein

class OntologyManager:
    def __init__(self, ontology_path: str):
        self.graph = Graph()
        self.graph.parse(ontology_path)
        self._build_index()

    def _build_index(self):
        """Precompute ontology terms for faster validation"""
        self.classes = {str(cls).split("#")[-1].split("/")[-1] 
                       for cls in self.graph.subjects(RDF.type, OWL.Class)}
        self.properties = {str(p).split("#")[-1].split("/")[-1] 
                          for p in self.graph.predicates()}

    def validate_class(self, term: str) -> bool:
        return term in self.classes

    def validate_property(self, term: str) -> bool:
        return term in self.properties

    def suggest_similar_class(self, term: str, threshold=0.7) -> list:
        """Fuzzy matching for class suggestions"""
        return [
            cls for cls in self.classes
            if Levenshtein.ratio(term.lower(), cls.lower()) > threshold
        ][:3]

    def suggest_similar_property(self, term: str, threshold=0.7) -> list:
        """Fuzzy matching for property suggestions"""
        return [
            prop for prop in self.properties
            if Levenshtein.ratio(term.lower(), prop.lower()) > threshold
        ][:3]