from rdflib import Graph, URIRef
import os
from openie import StanfordOpenIE
from dotenv import load_dotenv

load_dotenv()


class TextToRDFConverter:
    def __init__(self):
        openie_path = os.getenv('CORENLP_HOME')
        os.environ['CORENLP_HOME'] = openie_path
        # Load the OpenIE model from StanfordOpenIE
        properties = {
            'openie.affinity_probability_cap': 2/3,
            'openie.model.path': openie_path
        }
        self.client = StanfordOpenIE(properties=properties)

    def extract_subject_predicate_object(self, text):
        triples = []

        try:
            triples = self.client.annotate(text)
            triples.append(triples)
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
        triples = self.extract_triples(text)

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
