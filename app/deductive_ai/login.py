import streamlit as st
from deductive_ai.utils.authentication import authenticate, change_password, create_user

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        authenticated, role, first_login = authenticate(username, password)
        if authenticated:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.role = role
            
            if first_login:
                st.session_state.first_login = True
            else:
                st.session_state.first_login = False
            
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

    # Change password section for first login or regular users
    if 'first_login' in st.session_state and st.session_state.first_login:
        st.warning("Please change your password on first login.")
    elif 'authenticated' in st.session_state and st.session_state.authenticated:
        st.subheader("Change Password")
    else:
        st.subheader("Change Password (Optional)")

    if 'authenticated' in st.session_state and st.session_state.authenticated:
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        if st.button("Change Password"):
            if new_password == confirm_password:
                if change_password(st.session_state.username, new_password):
                    st.success("Password changed successfully.")
                    st.session_state.first_login = False
                else:
                    st.error("Failed to change password.")
            else:
                st.error("Passwords do not match.")

    # Admin UI for creating new users
    if 'role' in st.session_state and st.session_state.role == "Admin":
        st.subheader("Create New User")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New User Password", type="password")
        confirm_password = st.text_input("Confirm New User Password", type="password")
        role = st.selectbox("Role", ["User", "Admin"])
        
        if st.button("Create User"):
            if new_password == confirm_password:
                if create_user(st.session_state.username, new_username, new_password, role):
                    st.success(f"User {new_username} created successfully.")
                else:
                    st.error(f"Failed to create user {new_username}.")
            else:
                st.error("Passwords do not match.")
