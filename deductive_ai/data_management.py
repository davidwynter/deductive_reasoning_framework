import streamlit as st
import pandas as pd
from deductive_reasoning_framework.deductive_ai.core.text_to_rdf import TextToRDFConverter
from utils.file_utils import save_uploaded_file, list_files, delete_file, save_json, load_json

def data_rule_management_page():
    st.title("Data & Rule Management")
    
    # Ensure the engine is initialized
    if 'engine' not in st.session_state:
        from engine.deductive_engine import DeductiveReasoningEngine
        st.session_state.engine = DeductiveReasoningEngine()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Data Upload", "Rule Management", "Ontology Management"])
    
    with tab1:
        data_upload_section()
    
    with tab2:
        rule_management_section()
    
    with tab3:
        ontology_management_section()

# data_management.py
def knowledge_graph_population_tab():
    st.subheader("Knowledge Graph Population")
    
    uploaded_files = st.file_uploader("Upload text files", 
                                    type=["txt", "csv"],
                                    accept_multiple_files=True)
    
    if uploaded_files and st.button("Populate KG"):
        texts = []
        for file in uploaded_files:
            texts.append(file.read().decode())
            
        try:
            with st.spinner("Processing texts..."):
                st.session_state.engine.populate_from_texts(texts)
                st.success(f"Added {len(texts)} documents to knowledge graph")
                
            # Show stats
            graph = st.session_state.engine.graph
            st.metric("Total Triples", len(graph))
            
        except ValueError as e:
            st.error(str(e))
            
def data_upload_section():
    st.subheader("Data Upload")
    
    # File upload section
    st.write("Upload data files in various formats:")
    
    # Create columns for different file types
    col1, col2 = st.columns(2)
    
    with col1:
        # RDF/Turtle files
        ttl_file = st.file_uploader("Upload Turtle (.ttl) file", type=["ttl"], key="ttl_uploader")
        if ttl_file is not None:
            file_path = save_uploaded_file(ttl_file, "ttl")
            st.success(f"File saved to {file_path}")
            
            # Option to load the file into the engine
            if st.button("Load TTL file into engine", key="load_ttl"):
                try:
                    st.session_state.engine.load_data(str(file_path))
                    st.success("Data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        # N3 files
        n3_file = st.file_uploader("Upload N3 (.n3) file", type=["n3"], key="n3_uploader")
        if n3_file is not None:
            file_path = save_uploaded_file(n3_file, "n3")
            st.success(f"File saved to {file_path}")
            
            # Option to load the file into the engine
            if st.button("Load N3 file into engine", key="load_n3"):
                try:
                    st.session_state.engine.load_data(str(file_path))
                    st.success("Data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
    
    with col2:
        # XML/RDF files
        xml_file = st.file_uploader("Upload XML/RDF (.xml, .rdf) file", type=["xml", "rdf"], key="xml_uploader")
        if xml_file is not None:
            file_path = save_uploaded_file(xml_file, "xml")
            st.success(f"File saved to {file_path}")
            
            # Option to load the file into the engine
            if st.button("Load XML file into engine", key="load_xml"):
                try:
                    st.session_state.engine.load_data(str(file_path))
                    st.success("Data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        # Plain text files
        txt_file = st.file_uploader("Upload Text (.txt) file", type=["txt"], key="txt_uploader")
        if txt_file is not None:
            file_path = save_uploaded_file(txt_file, "txt")
            st.success(f"File saved to {file_path}")
    
    # List uploaded files
    st.subheader("Uploaded Files")
    
    # Create tabs for different file types
    file_tab1, file_tab2, file_tab3, file_tab4 = st.tabs(["TTL Files", "N3 Files", "XML Files", "TXT Files"])
    
    with file_tab1:
        display_files("ttl")
    
    with file_tab2:
        display_files("n3")
    
    with file_tab3:
        display_files("xml")
    
    with file_tab4:
        display_files("txt")

def display_files(file_type):
    files = list_files(file_type)
    if files:
        for file in files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(file)
            with col2:
                if st.button("Delete", key=f"delete_{file}"):
                    delete_file(file)
                    st.rerun()
    else:
        st.write(f"No {file_type.upper()} files uploaded yet.")

def rule_management_section():
    st.subheader("Rule Management")

    # Add natural language/SWRL toggle
    rule_input_mode = st.radio("Input Mode:", ["Natural Language", "SWRL"], horizontal=True)
    
    # Load existing rules if available
    rules_file = "rules_data"
    existing_rules = load_json(rules_file)
    
    if rule_input_mode == "Natural Language":
        # Natural Language Input Section
        nl_rule = st.text_area(
            "Describe your rule in natural language",
            value=existing_rules.get("nl_rule", ""),
            height=100,
            help="Example: 'If a customer purchases more than 5 items, apply discount'"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Convert to SWRL"):
                try:
                    # Call the NL-to-SWRL conversion engine
                    swrl_rule = st.session_state.engine.convert_nl_to_swrl(nl_rule)
                    existing_rules["current_rule"] = swrl_rule
                    existing_rules["nl_rule"] = nl_rule
                    save_json(existing_rules, rules_file)
                    st.success("Rule converted successfully!")
                except Exception as e:
                    st.error(f"Conversion error: {str(e)}")
        
        with col2:
            if st.button("Validate Rule"):
                validation_result = st.session_state.engine.validate_swrl_rule(nl_rule)
                if validation_result["valid"]:
                    st.success("Rule is valid!")
                else:
                    st.error(f"Validation failed: {validation_result['message']}")
                    
        # Display converted SWRL if available
        if existing_rules.get("current_rule"):
            st.subheader("Generated SWRL Rule")
            st.code(existing_rules["current_rule"])
            
    else:     
        # Rule input section
        st.write("Enter reasoning rules in SWRL format:")
        
        # Load existing rules if available
        rules_file = "rules_data"
        existing_rules = load_json(rules_file)
        
        # Text area for rule input
        rule_text = st.text_area("Rule (SWRL format)", 
                                value=existing_rules.get("current_rule", ""),
                                height=150,
                                help="Enter a rule in SWRL format, e.g., hasSymptom(?patient, ?symptom) ^ hasSymptom(?disease, ?symptom) -> mayHave(?patient, ?disease)")
        
        # Save rule button
        if st.button("Save Rule"):
            existing_rules["current_rule"] = rule_text
            save_json(existing_rules, rules_file)
            st.success("Rule saved successfully!")
        
        # Apply rule button
        if st.button("Apply Rule to Engine"):
            try:
                st.session_state.engine.load_rules(rule_text)
                st.success("Rule applied successfully!")
            except Exception as e:
                st.error(f"Error applying rule: {str(e)}")
        
        # Rule templates
        st.subheader("Rule Templates")
        
        templates = {
            "Disease Inference": "hasSymptom(?patient, ?symptom) ^ hasSymptom(?disease, ?symptom) -> mayHave(?patient, ?disease)",
            "Property Inheritance": "subClassOf(?c1, ?c2) ^ hasProperty(?c2, ?p) -> hasProperty(?c1, ?p)",
            "Transitive Relation": "ancestor(?x, ?y) ^ ancestor(?y, ?z) -> ancestor(?x, ?z)"
        }
        
        selected_template = st.selectbox("Select a template", list(templates.keys()))
        
        if st.button("Use Template"):
            st.session_state.rule_text = templates[selected_template]
            st.rerun()

def ontology_management_section():
    st.subheader("Ontology Management")
    
    # Ontology upload
    st.write("Upload ontology files:")
    
    ontology_file = st.file_uploader("Upload Ontology (.owl, .rdf, .ttl)", type=["owl", "rdf", "ttl"], key="ontology_uploader")
    
    if ontology_file is not None:
        # Determine file type based on extension
        file_extension = ontology_file.name.split(".")[-1].lower()
        file_path = save_uploaded_file(ontology_file, file_extension)
        st.success(f"Ontology file saved to {file_path}")
        
        # Option to load the ontology into the engine
        if st.button("Load Ontology into Engine"):
            try:
                st.session_state.engine.onto = st.session_state.engine.load_ontology(str(file_path))
                st.success("Ontology loaded successfully!")
            except Exception as e:
                st.error(f"Error loading ontology: {str(e)}")
    
    # Display ontology information if loaded
    if hasattr(st.session_state.engine, 'onto') and st.session_state.engine.onto is not None:
        st.subheader("Loaded Ontology Information")
        
        # Basic ontology info
        st.write(f"Ontology IRI: {st.session_state.engine.onto.base_iri}")
        
        # Classes
        st.write("Classes:")
        classes = list(st.session_state.engine.onto.classes())
        if classes:
            class_df = pd.DataFrame({
                "Class Name": [cls.name for cls in classes[:10]],  # Limit to first 10 for display
                "IRI": [cls.iri for cls in classes[:10]]
            })
            st.dataframe(class_df)
            if len(classes) > 10:
                st.write(f"... and {len(classes) - 10} more classes")
        else:
            st.write("No classes found in the ontology.")
        
        # Properties
        st.write("Properties:")
        properties = list(st.session_state.engine.onto.properties())
        if properties:
            prop_df = pd.DataFrame({
                "Property Name": [prop.name for prop in properties[:10]],  # Limit to first 10 for display
                "IRI": [prop.iri for prop in properties[:10]]
            })
            st.dataframe(prop_df)
            if len(properties) > 10:
                st.write(f"... and {len(properties) - 10} more properties")
        else:
            st.write("No properties found in the ontology.")

def process_text_data(text: str):
    converter = TextToRDFConvertor()
    triples = converter.tag_entity_only(text)  # Or labeled_dataset()
    save_as_rdf(triples)