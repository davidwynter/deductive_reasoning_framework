import streamlit as st
from deductive_ai.engine.deductive_engine import DeductiveReasoningEngine
from deductive_ai.utils.file_utils import save_json, load_json


# Inference & Validation Page
def inference_validation_page():
    st.title("Inference & Validation")
    
    # Ensure the engine is initialized
    if 'engine' not in st.session_state:
        st.session_state.engine = DeductiveReasoningEngine()

    # Dataset selection
    st.subheader("Select Dataset for Inference")
    selected_dataset = st.selectbox("Choose Dataset", options=st.session_state.engine.get_datasets())

    # Load existing variables and relationships if available
    variables_file = f"{selected_dataset}_variables.json"
    if variables_file.exists():
        dataset_info = load_json(variables_file.stem)
        variables = dataset_info.get("variables", [])
        relationships = dataset_info.get("relationships", [])
    else:
        variables = []
        relationships = []

    # Input for variables
    st.subheader("Define Variables")
    variable_input = st.text_input("Add Variable")
    if st.button("Add Variable"):
        if variable_input:
            variables.append(variable_input)
            st.success(f"Variable '{variable_input}' added.")
        else:
            st.warning("Please enter a valid variable name.")

    st.write("Current Variables:", variables)

    # Input for relationships
    st.subheader("Define Relationships")
    parent = st.selectbox("Select Parent Variable", options=variables)
    child = st.selectbox("Select Child Variable", options=variables)
    if st.button("Add Relationship"):
        if parent and child and parent != child:
            relationships.append((parent, child))
            st.success(f"Relationship '{parent} -> {child}' added.")
        else:
            st.warning("Please select valid and distinct parent and child variables.")

    st.write("Current Relationships:", relationships)

    # Save variables and relationships
    if st.button("Save Variables and Relationships"):
        save_json({"variables": variables, "relationships": relationships}, variables_file.stem)
        st.success(f"Variables and relationships for dataset '{selected_dataset}' saved successfully.")

    # Ontology loading
    st.subheader(f"Load Ontology for Dataset: {selected_dataset}")
    ontology_file = st.file_uploader("Choose an ontology file", type=["owl", "rdf", "ttl"])
    
    if ontology_file is not None:
        # Construct the filename using the dataset name
        ontology_filename = f"{selected_dataset}_ontology.owl"
        
        # Save uploaded ontology file locally with the constructed filename
        with open(ontology_filename, "wb") as f:
            f.write(ontology_file.getbuffer())
        
        # Load the ontology into the engine
        st.session_state.engine.load_ontology(ontology_filename)
        st.success(f"Ontology for dataset '{selected_dataset}' loaded successfully as '{ontology_filename}'.")

    # Reasoning methods selection
    st.subheader("Select Reasoning Methods")
    methods = ["Bayesian Networks", "Markov Logic Networks", "Probabilistic Programming", "Monte Carlo Methods"]
    weights = {}
    for method in methods:
        weights[method] = st.slider(f"{method} Contribution (%)", 0, 100, 0)
    
    # Validate that the weights sum to 100%
    if sum(weights.values()) != 100:
        st.error("The weights must sum to 100%.")
    else:
        if st.button("Run Inference"):
            engine = st.session_state.engine
            engine.set_confidence_weights(weights)
            engine.run_inference(selected_dataset)
            st.success("Inference run successfully.")
            st.write("Results:")
            
            # Show results
            matched, unmatched = engine.get_validation_results()
            st.metric("Matched Triples", f"{matched}%")
            st.metric("Unmatched Triples", f"{unmatched}%")
            
            st.subheader("Matched Triples")
            st.table(pd.DataFrame(engine.get_matched_triples()))  # Show matched triples
            
            st.subheader("Unmatched Triples")
            st.table(pd.DataFrame(engine.get_unmatched_triples()))  # Show unmatched triples
