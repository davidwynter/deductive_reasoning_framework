import streamlit as st
from streamlit.logger import get_logger


LOGGER = get_logger(__name__)
st.set_page_config(page_title="Deductive Reasoning Framework", layout="wide")


def run():

    st.write("# Welcome to Streamlit! 👋")
    print("set_page_config executed")

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        from login import login_page
        login_page()
    else:
        menu = st.sidebar.selectbox("Menu", ["Data & Rule Management", "Inference & Validation", "Hyper-Parameter Tuning", "Logout"])
        if menu == "Data & Rule Management":
            from data_management import data_rule_management_page
            data_rule_management_page()
            
        elif menu == "Inference & Validation":
            from inference_validation import inference_validation_page
            inference_validation_page()
            
        elif menu == "Hyper-Parameter Tuning":
            from hyperparameter_tuning import hyperparameter_tuning_page
            hyperparameter_tuning_page() 
            
        elif menu == "Logout":
            st.session_state.authenticated = False
            st.experimental_rerun()

if __name__ == "__main__":
    run()