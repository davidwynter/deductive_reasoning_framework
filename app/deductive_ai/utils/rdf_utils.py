from rdflib import Graph, URIRef

def load_rdf_file(file_path, format="ttl"):
    """
    Load an RDF file into a rdflib Graph.
    
    :param file_path: Path to the RDF file.
    :param format: Format of the RDF file ('ttl', 'n3', 'xml', etc.).
    :return: A rdflib Graph object populated with the data from the file.
    """
    graph = Graph()
    graph.parse(file_path, format=format)
    return graph

def convert_triples_to_rdf(triples, base_url="http://example.org", format="ttl"):
    """
    Convert a list of triples to an RDF graph in the specified format.
    
    :param triples: List of triples (subject, predicate, object).
    :param base_url: The base URL to be used for creating URIs.
    :param format: The desired RDF format ("ttl" or "n3").
    :return: RDF data as a string in the specified format.
    """
    graph = Graph()
    for subj, pred, obj in triples:
        subj_uri = URIRef(f"{base_url}/{subj.replace(' ', '_')}")
        pred_uri = URIRef(f"{base_url}/{pred.replace(' ', '_')}")
        obj_uri = URIRef(f"{base_url}/{obj.replace(' ', '_')}")
        graph.add((subj_uri, pred_uri, obj_uri))
    
    return graph.serialize(format=format).decode("utf-8")

def save_rdf_to_file(graph, file_path, format="ttl"):
    """
    Save an RDF graph to a file in the specified format.
    
    :param graph: A rdflib Graph object.
    :param file_path: Path to the file where the graph will be saved.
    :param format: Format of the RDF file ('ttl', 'n3', 'xml', etc.).
    """
    graph.serialize(destination=file_path, format=format)

def parse_rdf_data(rdf_data, base_url="http://example.org", format="ttl"):
    """
    Parse RDF data from a string into a rdflib Graph.
    
    :param rdf_data: RDF data as a string.
    :param base_url: The base URL to be used for creating URIs if needed.
    :param format: Format of the RDF data ('ttl', 'n3', 'xml', etc.).
    :return: A rdflib Graph object populated with the parsed data.
    """
    graph = Graph()
    graph.parse(data=rdf_data, format=format)
    return graph
