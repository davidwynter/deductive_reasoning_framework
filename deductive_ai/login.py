import streamlit as st
import os
from utils.authentication import authenticate, change_password, create_user, register_user

def login_page():
    # Initialize page state if not already set
    if 'login_page' not in st.session_state:
        st.session_state.login_page = "login"  # Options: login, register, forgot_password
    
    # Tabs for different authentication options
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Login", use_container_width=True):
                authenticated, role, first_login = authenticate(username, password)
                if authenticated:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = role
                    st.session_state.menu = "Data & Rule Management"  # Set default menu
                    
                    if first_login:
                        st.session_state.first_login = True
                    else:
                        st.session_state.first_login = False
                    
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Create New Account")
        new_username = st.text_input("Choose Username", key="reg_username")
        new_password = st.text_input("Choose Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        # Optional registration code (can be enabled/disabled)
        registration_enabled = os.environ.get("ENABLE_REGISTRATION", "true").lower() == "true"
        require_code = os.environ.get("REQUIRE_REGISTRATION_CODE", "false").lower() == "true"
        
        registration_code = None
        if require_code:
            registration_code = st.text_input("Registration Code", type="password")
        
        if st.button("Register", disabled=not registration_enabled):
            if not registration_enabled:
                st.error("Self-registration is currently disabled")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long")
            else:
                success, message = register_user(new_username, new_password, registration_code)
                if success:
                    st.success(message)
                    st.info("You can now log in with your new account")
                else:
                    st.error(message)
    
    # Change password section for first login or regular users
    if 'first_login' in st.session_state and st.session_state.first_login:
        st.warning("Please change your password on first login.")
        
        with st.form("change_password_form"):
            st.subheader("Change Password (Required)")
            new_password = st.text_input("New Password", type="password", key="first_login_new_pw")
            confirm_password = st.text_input("Confirm New Password", type="password", key="first_login_confirm_pw")
            
            submitted = st.form_submit_button("Change Password")
            if submitted:
                if new_password == confirm_password:
                    if change_password(st.session_state.username, new_password):
                        st.success("Password changed successfully.")
                        st.session_state.first_login = False
                        st.rerun()
                    else:
                        st.error("Failed to change password.")
                else:
                    st.error("Passwords do not match.")
