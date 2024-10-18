import streamlit as st
from streamlit.logger import get_logger


LOGGER = get_logger(__name__)
st.set_page_config(page_title="Deductive Reasoning Framework", layout="wide")

# Function to display bold text
def make_bold(text):
    return f"**{text}**"

def run():

    st.write("# Deductive Reasoning Framework")
    print("set_page_config executed")

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        from login import login_page
        login_page()
    else:
        if st.sidebar.button("Data & Rule Management"):
            st.session_state.menu = "Data & Rule Management"
        elif st.sidebar.button("Inference & Validation"):
            st.session_state.menu = "Inference & Validation"
        elif st.sidebar.button("Hyper-Parameter Tuning"):
            st.session_state.menu = "Hyper-Parameter Tuning"
        elif st.sidebar.button("Logout"):
            st.session_state.menu = "Logout"
            st.session_state.authenticated = False
            st.rerun()

        # Render the selected page with bold formatting in the sidebar
        st.sidebar.markdown(make_bold("Data & Rule Management") if st.session_state.menu == "Data & Rule Management" else "Data & Rule Management")
        st.sidebar.markdown(make_bold("Inference & Validation") if st.session_state.menu == "Inference & Validation" else "Inference & Validation")
        st.sidebar.markdown(make_bold("Hyper-Parameter Tuning") if st.session_state.menu == "Hyper-Parameter Tuning" else "Hyper-Parameter Tuning")
        st.sidebar.markdown(make_bold("Logout") if st.session_state.menu == "Logout" else "Logout")

        # Display content based on selected menu
        if st.session_state.menu == "Data & Rule Management":
            from data_management import data_rule_management_page
            data_rule_management_page()
            
        elif st.session_state.menu == "Inference & Validation":
            from inference_validation import inference_validation_page
            inference_validation_page()

        elif st.session_state.menu == "Hyper-Parameter Tuning":
            from hyperparameter_tuning import hyperparameter_tuning_page
            hyperparameter_tuning_page()

        elif st.session_state.menu == "Logout":
            st.session_state.authenticated = False
            st.rerun()

if __name__ == "__main__":
    run()