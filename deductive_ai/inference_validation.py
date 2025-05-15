import streamlit as st
import pandas as pd
import os
from engine.validation_engine import ValidationEngine
from engine.deductive_engine import DeductiveReasoningEngine
from utils.file_utils import save_json, load_json
from engine.nl_to_swrl import NLToSWRLConverter
from core.text_to_rdf import TextToRDFConvertor  # Your enhanced class
from core.ontology_manager import OntologyManager

def load_model_dataset_associations():
    """Load the existing model-dataset associations"""
    return load_json("model_dataset_associations")

def save_model_dataset_association(dataset, model_path):
    """Save the association between a dataset and a model"""
    associations = load_model_dataset_associations()
    associations[dataset] = model_path
    save_json(associations, "model_dataset_associations")

# Inference & Validation Page
def inference_validation_page(ontology_path=None):
    st.title("Inference & Validation")
    
    # Initialize session state components
    if 'text_converter' not in st.session_state:
        st.session_state.om = OntologyManager("path/to/ontology.owl")
        st.session_state.text_converter = TextToRDFConvertor(st.session_state.om)
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Rule Validation", "Text to RDF"])
    
    with tab1:
        _show_rule_validation_ui()  # Your existing rule validation UI
    
    with tab2:
        _show_text_to_rdf_ui()  # New text conversion UI

def _show_text_to_rdf_ui():
    """Dedicated tab for text-to-RDF conversion"""
    st.subheader("Unstructured Text to RDF")
    
    text_input = st.text_area(
        "Paste text (clinical notes, product descriptions, etc.)",
        height=200,
        placeholder="Patient X presented with fever and headache..."
    )
    
    if st.button("Convert to RDF"):
        with st.spinner("Extracting triples..."):
            try:
                # Convert and display results
                kg = st.session_state.text_converter.populate_kg([text_input])
                triples = list(kg)
                
                st.success(f"Extracted {len(triples)} triples")
                
                # Visualize as a table
                st.dataframe(
                    pd.DataFrame(triples, columns=["Subject", "Predicate", "Object"]),
                    height=min(300, len(triples) * 35)
                )
                
                # Option to save to knowledge graph
                if st.button("Save to Knowledge Graph"):
                    st.session_state.engine.graph += kg
                    st.toast("Triples added to KG!")
                    
            except Exception as e:
                st.error(f"Conversion failed: {str(e)}")

def _show_rule_validation_ui():
    if 'converter' not in st.session_state:
        st.session_state.converter = NLToSWRLConverter(ontology_path=ontology_path)
        st.session_state.validator = ValidationEngine(st.session_state.converter)

    # Ensure the engine is initialized
    if 'engine' not in st.session_state:
        st.session_state.engine = DeductiveReasoningEngine()
        
    # Initialize ConfidenceAdjuster if not already done
    if not hasattr(st.session_state.engine, 'confidence_adjuster') or st.session_state.engine.confidence_adjuster is None:
        from confidence_model import ConfidenceAdjuster
        st.session_state.engine.confidence_adjuster = ConfidenceAdjuster()

    # Dataset selection
    st.subheader("Select Dataset for Inference")
    selected_dataset = st.selectbox("Choose Dataset", options=st.session_state.engine.get_datasets())

    # Load existing variables and relationships if available
    variables_file = f"{selected_dataset}_variables.json"
    if os.path.exists(variables_file):
        dataset_info = load_json(variables_file)
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
        save_json({"variables": variables, "relationships": relationships}, variables_file)
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

    # ConfidenceAdjuster ML model section
    st.subheader("ConfidenceAdjuster ML Model")
    st.info("This section allows you to train and use a machine learning model to adjust confidence scores.")

    # 1. Import training data
    st.write("Import Training Data")
    training_data_file = st.file_uploader("Choose a training data file", type=["csv"])
    if training_data_file is not None:
        training_data = pd.read_csv(training_data_file)
        st.write("Training data preview:")
        st.write(training_data.head())

        # Derive input_size from the training data
        input_size = len(training_data.columns) - 1  # Assuming the last column is the target
        st.write(f"Derived input size: {input_size}")

    # 2. Define location to save the trained model
    st.write("Define Model Save Location")
    model_save_path = st.text_input("Enter path to save the trained model", f"models/{selected_dataset}_model.pth")

    # 3. Select path to a trained model
    st.write("Select Trained Model")
    model_associations = load_model_dataset_associations()
    default_model_path = model_associations.get(selected_dataset, f"models/{selected_dataset}_model.pth")
    model_load_path = st.text_input("Enter path to load a trained model", default_model_path)

    # 4. Associate trained model with selected dataset
    if st.button("Associate Model with Dataset"):
        save_model_dataset_association(selected_dataset, model_load_path)
        st.success(f"Model at {model_load_path} associated with dataset {selected_dataset}")

    # 5. Train or load the model
    if st.button("Train New Model"):
        if training_data_file is not None:
            try:
                # Convert training_data to the format expected by train_ml_model
                # This depends on your specific data format and might need adjustment
                formatted_training_data = []
                
                # Check if the required columns exist
                required_columns = ['subject', 'predicate', 'object', 'confidence']
                missing_columns = [col for col in required_columns if col not in training_data.columns]
                
                if missing_columns:
                    st.error(f"Training data is missing required columns: {', '.join(missing_columns)}")
                else:
                    # Format the data
                    formatted_training_data = [
                        ((row['subject'], row['predicate'], row['object']), row['confidence'])
                        for _, row in training_data.iterrows()
                    ]
                    
                    # Train the model
                    with st.spinner("Training model..."):
                        st.session_state.engine.train_ml_model(formatted_training_data, epochs=100, learning_rate=0.001)
                        st.session_state.engine.save_ml_model(model_save_path)
                        save_model_dataset_association(selected_dataset, model_save_path)
                        st.success(f"Model trained and saved to {model_save_path}")
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
        else:
            st.error("Please upload training data before training the model.")

    if st.button("Load Existing Model"):
        st.session_state.engine.load_ml_model(model_load_path)
        st.success(f"Model loaded from {model_load_path}")

    # Reasoning methods selection
    st.subheader("Select Reasoning Methods")
    methods = ["Bayesian Networks", "Markov Logic Networks", "Probabilistic Programming", "Monte Carlo Methods", "ML Model"]
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
