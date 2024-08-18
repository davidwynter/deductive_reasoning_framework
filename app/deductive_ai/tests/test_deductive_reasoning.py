import pytest
from rdflib import Namespace, Literal
from owlready2 import get_ontology

'''
You'll need:

export PYTHONPATH=$(pwd)
pytest tests/

'''
from engine.deductive_reasoning import DeductiveReasoningEngine

# Define the base namespace
EX = Namespace("http://example.org/animals#")

# Test data
ontology_data = """
@prefix : <http://example.org/animals#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

:Animal rdf:type owl:Class .
:Habitat rdf:type owl:Class .
:Food rdf:type owl:Class .

:Lion rdf:type owl:Class ;
      rdfs:subClassOf :Animal .
:Tiger rdf:type owl:Class ;
       rdfs:subClassOf :Animal .
:Elephant rdf:type owl:Class ;
          rdfs:subClassOf :Animal .
:Deer rdf:type owl:Class ;
      rdfs:subClassOf :Animal .
:Crocodile rdf:type owl:Class ;
           rdfs:subClassOf :Animal .

:Grassland rdf:type owl:Class ;
           rdfs:subClassOf :Habitat .
:Forest rdf:type owl:Class ;
        rdfs:subClassOf :Habitat .
:River rdf:type owl:Class ;
       rdfs:subClassOf :Habitat .
:Wetland rdf:type owl:Class ;
         rdfs:subClassOf :Habitat .

:Carnivore rdf:type owl:Class ;
           rdfs:subClassOf :Food .
:Herbivore rdf:type owl:Class ;
           rdfs:subClassOf :Food .

:livesIn rdf:type owl:ObjectProperty ;
         rdfs:domain :Animal ;
         rdfs:range :Habitat .

:eats rdf:type owl:ObjectProperty ;
      rdfs:domain :Animal ;
      rdfs:range :Food .
"""

swrl_rules = """
Animal(?a) ^ livesIn(?a, Grassland) ^ eats(?a, Carnivore) -> huntsInGroups(?a, true)
Animal(?a) ^ livesIn(?a, Forest) ^ eats(?a, Carnivore) -> solitaryHunter(?a, true)
Animal(?a) ^ eats(?a, Carnivore) ^ livesIn(?a, River) -> ambushPredator(?a, true)
"""

expected_triples = [
    (EX.Lion, EX.livesIn, EX.Grassland),
    (EX.Lion, EX.eats, EX.Carnivore),
    (EX.Lion, EX.huntsInGroups, Literal(True)),

    (EX.Tiger, EX.livesIn, EX.Forest),
    (EX.Tiger, EX.eats, EX.Carnivore),
    (EX.Tiger, EX.solitaryHunter, Literal(True)),

    (EX.Elephant, EX.livesIn, EX.Grassland),
    (EX.Elephant, EX.livesIn, EX.Forest),
    (EX.Elephant, EX.eats, EX.Herbivore),

    (EX.Deer, EX.livesIn, EX.Forest),
    (EX.Deer, EX.livesIn, EX.Grassland),
    (EX.Deer, EX.eats, EX.Herbivore),

    (EX.Crocodile, EX.livesIn, EX.River),
    (EX.Crocodile, EX.livesIn, EX.Wetland),
    (EX.Crocodile, EX.eats, EX.Carnivore),
    (EX.Crocodile, EX.ambushPredator, Literal(True)),
]


@pytest.fixture
def reasoning_engine():
    # Initialize the deductive reasoning engine
    engine = DeductiveReasoningEngine()
    
    # Create a temporary ontology from the test data
    ontology = get_ontology("http://example.org/animals").load()
    ontology.load_from_string(ontology_data)
    
    # Load the ontology into the engine
    engine.onto = ontology
    
    # Load SWRL rules into the engine
    engine.load_rules(swrl_rules, rule_format="n3")
    
    return engine


def test_inferred_triples(reasoning_engine):
    # Apply reasoning to the ontology
    reasoning_engine.apply_reasoning()

    # Extract inferred triples from the engine's inferred graph
    inferred_triples = set(reasoning_engine.inferred_graph)
    
    # Compare each expected triple with the inferred triples
    for triple in expected_triples:
        assert triple in inferred_triples, f"Expected triple {triple} not found in inferred triples"


def test_no_extra_triples(reasoning_engine):
    # Apply reasoning to the ontology
    reasoning_engine.apply_reasoning()

    # Extract inferred triples from the engine's inferred graph
    inferred_triples = set(reasoning_engine.inferred_graph)
    
    # Ensure no extra triples were inferred
    for triple in inferred_triples:
        assert triple in expected_triples, f"Extra triple {triple} found in inferred triples"
