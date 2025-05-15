# core/text_to_rdf.py
from rdflib import Graph, URIRef, Literal
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Tuple
import torch
from llm.memory_manager import XPUMemoryManager

class TextToRDFConverter:
    def __init__(self, ontology_manager):
        self.om = ontology_manager
        self.mem_manager = XPUMemoryManager()
        self.device = "xpu" if torch.xpu.is_available() else "cpu"
        # Load DeBERTa ofr OpenIE
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base-openie")
        self.model = AutoModelForTokenClassification.from_pretrained("microsoft/deberta-v3-base-openie")
        
    def populate_kg(self, text_sources: List[str]) -> Graph:
        """Main method for knowledge graph population"""
        kg = Graph()
        self.mem_manager.clear_cache()
        
        try:
            with torch.xpu.stream(torch.xpu.Stream()):
                for text in text_sources:
                    triples = self._process_text(text)
                    for s, p, o in triples:
                        kg.add((URIRef(s), URIRef(p), Literal(o) if isinstance(o, str) else URIRef(o)))
                        
            return kg
        finally:
            self.mem_manager.clear_cache()

    def _process_text(self, text: str) -> List[Tuple[str, str, str]]:
        """XPU-optimized text processing"""
        # Placeholder - implement your actual NLP logic here
        if "fever" in text.lower():
            return [
                ("http://example.org/PatientX", "http://example.org/hasSymptom", "http://example.org/Fever"),
                ("http://example.org/PatientX", "http://example.org/hasAge", "45")
            ]
        return []
    
    def extract_subject_predicate_object(self, text):
        triples = []

        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            outputs = self.model(**inputs)
            triples.append(outputs.triples)
        except Exception as e:
            print("Error during annotation:", e)
            return []
        return triples

    def convert_to_rdf(self, text, url, rdf_format="ttl"):
        """
        Convert extracted triples to RDF format.
        :param text: Unstructured text string.
        :param rdf_format: The desired RDF format ("ttl" or "n3").
        :return: RDF data in the specified format.
        """
        g = Graph()

        # Extract triples from the text
        triples = self.extract_subject_predicate_object(text)

        # Convert triples to RDF
        for subj, pred, obj in triples:
            subj_uri = URIRef(f"{url}/{subj.replace(' ', '_')}")
            pred_uri = URIRef(f"{url}/{pred.replace(' ', '_')}")
            obj_uri = URIRef(f"{url}/{obj.replace(' ', '_')}")
            g.add((subj_uri, pred_uri, obj_uri))

        # Serialize the graph to the specified RDF format
        rdf_data = g.serialize(format=rdf_format).decode("utf-8")
        return rdf_data


if __name__ == "__main__":
    converter = TextToRDFConverter()
    text = "Barack Obama was born in Hawaii. He was elected president in 2008."
    rdf_data = converter.convert_to_rdf(text, rdf_format="ttl")
    print(rdf_data)
