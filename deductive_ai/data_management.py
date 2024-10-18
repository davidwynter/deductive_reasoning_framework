import streamlit as st
from engine.deductive_engine import DeductiveReasoningEngine
from engine.text_to_rdf import TextToRDFConverter
from utils.file_utils import save_json, load_json, save_uploaded_file, list_files, delete_file
from pathlib import Path

# Data & Rule Management Page
def data_rule_management_page():
    st.title("Data & Rule Management")
    
    # Initialize the DeductiveReasoningEngine if not already done
    if 'engine' not in st.session_state:
        st.session_state.engine = DeductiveReasoningEngine()

    # Base URL input
    base_url = st.text_input("Base URL for RDF URIs", value="http://example.org")

    # File upload options
    st.subheader("Upload Files")
    conversion_option = st.radio("Conversion Option", ("Unstructured Text to TTL/N3", "Direct Upload of TTL/N3/XML"))
    
    if conversion_option == "Unstructured Text to TTL/N3":
        uploaded_file = st.file_uploader("Choose an unstructured text file", type=["txt"])
        format_option = st.selectbox("Convert to", ("TTL", "N3"))
        if st.button("Convert and Load"):
            if uploaded_file is not None:
                text_content = uploaded_file.read().decode("utf-8")
                # Call the conversion function with the specified base URL
                converter = TextToRDFConverter()
                rdf_data = converter.convert_to_rdf(text_content, base_url=base_url, rdf_format=format_option.lower())
                # Load the converted RDF data into the engine
                st.session_state.engine.load_data_from_string(rdf_data, format=format_option.lower())
                st.success(f"File converted to {format_option} and loaded into graph with base URL: {base_url}.")
    else:
        uploaded_file = st.file_uploader("Choose a TTL, N3, or XML file", type=["ttl", "n3", "xml"])
        if st.button("Upload"):
            if uploaded_file is not None:
                file_type = uploaded_file.name.split(".")[-1]
                save_uploaded_file(uploaded_file, file_type)
                st.session_state.engine.load_data(uploaded_file, format=file_type)
                st.success(f"File {uploaded_file.name} loaded into graph with base URL: {base_url}.")

    # Rule entry
    st.subheader("Enter Rules")
    rule_format = st.radio("Save Rules as", ("N3", "SWRL"))
    rules_text = st.text_area("Enter Rules Here")
    if st.button("Save Rules"):
        save_json({"rules": rules_text, "format": rule_format}, "rules")
        st.session_state.engine.load_rules(rules_text)
        st.success(f"Rules saved in {rule_format} format.")

    # Load existing rules
    if st.button("Load Rules"):
        rules_data = load_json("rules")
        if rules_data:
            st.session_state.engine.load_rules(rules_data["rules"])
            st.success(f"Rules loaded in {rules_data['format']} format.")
        else:
            st.warning("No saved rules found.")

    # Expected triples upload
    st.subheader("Upload Expected Triples for Validation")
    expected_triples_file = st.file_uploader("Choose a file with expected triples", type=["ttl", "n3", "xml"])
    if st.button("Upload Expected Triples"):
        if expected_triples_file is not None:
            file_type = expected_triples_file.name.split(".")[-1]
            save_uploaded_file(expected_triples_file, file_type)
            st.session_state.engine.load_expected_triples(expected_triples_file, format=file_type)
            save_json({"file": expected_triples_file.name, "format": file_type}, "expected_triples")
            st.success("Expected triples loaded.")

    # Load existing expected triples
    if st.button("Load Expected Triples"):
        triples_data = load_json("expected_triples")
        if triples_data:
            triples_file = Path(f"uploaded_files/{triples_data['format']}/{triples_data['file']}")
            if triples_file.exists():
                st.session_state.engine.load_expected_triples(triples_file.open("rb"), format=triples_data['format'])
                st.success("Expected triples loaded.")
            else:
                st.warning("Saved expected triples file not found.")
        else:
            st.warning("No saved expected triples found.")

    # Dataset selection and naming
    st.subheader("Manage Named Datasets")
    selected_files = st.multiselect("Select Files to Include in Dataset", options=list_files())  # Replace with actual file list
    dataset_name = st.text_input("Dataset Name")
    if st.button("Save Dataset"):
        if selected_files and dataset_name:
            save_json({"dataset_name": dataset_name, "files": selected_files}, f"dataset_{dataset_name}")
            st.session_state.engine.save_named_dataset(dataset_name, selected_files)
            st.success(f"Dataset '{dataset_name}' saved with selected files.")

    # Load existing datasets
    if st.button("Load Dataset"):
        dataset_name_to_load = st.text_input("Enter Dataset Name to Load")
        dataset_data = load_json(f"dataset_{dataset_name_to_load}")
        if dataset_data:
            st.session_state.engine.save_named_dataset(dataset_data['dataset_name'], dataset_data['files'])
            st.success(f"Dataset '{dataset_name_to_load}' loaded with associated files.")
        else:
            st.warning("No saved dataset found with that name.")

    # Allow file deletion (optional)
    st.subheader("Delete Uploaded Files")
    file_type_to_delete = st.selectbox("Select file type to delete", ["ttl", "n3", "xml", "txt", "all"])
    if file_type_to_delete != "all":
        files_to_delete = list_files(file_type_to_delete)
    else:
        files_to_delete = list_files()
    
    if files_to_delete:
        file_to_delete = st.selectbox("Select a file to delete", files_to_delete)
        if st.button("Delete File"):
            delete_file(file_to_delete)
            st.success(f"File {file_to_delete} deleted successfully.")
