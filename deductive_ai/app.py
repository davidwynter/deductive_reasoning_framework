import streamlit as st
from streamlit.logger import get_logger
from core.memory_monitor import MemoryMonitor

LOGGER = get_logger(__name__)
st.set_page_config(page_title="InferIQ - Intelligent Inference Engine", layout="wide", page_icon="🧠")

# Function to display bold text
def make_bold(text):
    return f"**{text}**"

def run():
    # Start memory monitor
    monitor = MemoryMonitor(interval=30)
    monitor.start()
    
    st.write("# InferIQ")
    st.write("### Intelligent Inference & Reasoning Engine")
    print("set_page_config executed")

    # Initialize session state variables
    # Initialize Streamlit
    if not st.session_state.get('initialized'):
        st.session_state.initialized = True
        from inference_validation import inference_validation_page
        inference_validation_page()
        
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    if 'menu' not in st.session_state:
        st.session_state.menu = "Data & Rule Management"
        
    if 'engine' not in st.session_state:
        from engine.deductive_engine import DeductiveReasoningEngine
        st.session_state.engine = DeductiveReasoningEngine()

    if not st.session_state.authenticated:
        from login import login_page
        login_page()
        
    else:
        # Display user info in sidebar
        st.sidebar.markdown(f"**Logged in as:** {st.session_state.username}")
        st.sidebar.markdown(f"**Role:** {st.session_state.role}")
        st.sidebar.divider()
        
        # Main navigation menu - vertical layout
        st.sidebar.subheader("Navigation")
        
        # Create a container for the menu buttons
        menu_container = st.sidebar.container()
        
        # Add menu buttons vertically
        with menu_container:
            # Data & Rules button (top)
            if st.button("📊 Data & Rules",
                        key="data_rules_btn",
                        use_container_width=True,
                        type="primary" if st.session_state.menu == "Data & Rule Management" else "secondary"):
                st.session_state.menu = "Data & Rule Management"
                st.rerun()
            
            # Parameters button (second)
            if st.button("⚙️ Parameters",
                        key="params_btn",
                        use_container_width=True,
                        type="primary" if st.session_state.menu == "Hyper-Parameter Tuning" else "secondary"):
                st.session_state.menu = "Hyper-Parameter Tuning"
                st.rerun()
            
            # Inference button (third)
            if st.button("🔍 Inference",
                        key="inference_btn",
                        use_container_width=True,
                        type="primary" if st.session_state.menu == "Inference & Validation" else "secondary"):
                st.session_state.menu = "Inference & Validation"
                st.rerun()
            
            # Admin-only menu items (last with divider)
            if 'role' in st.session_state and st.session_state.role == "Admin":
                st.sidebar.divider()
                if st.button("👥 Users",
                            key="users_btn",
                            use_container_width=True,
                            type="primary" if st.session_state.menu == "User Management" else "secondary"):
                    st.session_state.menu = "User Management"
                    st.rerun()
        
        # Logout button at the bottom
        st.sidebar.divider()
        if st.sidebar.button("🚪 Logout", key="logout_btn", use_container_width=True):
            st.session_state.menu = "Logout"
            st.session_state.authenticated = False
            st.rerun()

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

        elif st.session_state.menu == "User Management":
            from user_management import user_management_page
            user_management_page()
            
        elif st.session_state.menu == "Logout":
            st.session_state.authenticated = False
            st.rerun()

if __name__ == "__main__":
    run()