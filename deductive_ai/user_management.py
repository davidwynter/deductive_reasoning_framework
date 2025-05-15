import streamlit as st
import pandas as pd
import os
from utils.authentication import load_users, save_users, create_user, delete_user, reset_user_password
import bcrypt

def user_management_page():
    st.title("User Management")
    
    # Load current users
    users = load_users()
    
    # Convert users to DataFrame for display
    user_data = []
    for username, data in users.items():
        user_data.append({
            "Username": username,
            "Role": data["role"],
            "First Login": "Yes" if data["first_login"] else "No"
        })
    
    df = pd.DataFrame(user_data)
    
    # Display users in a table
    st.subheader("Current Users")
    st.dataframe(df)
    
    # Create new user section
    st.subheader("Create New User")
    with st.form("create_user_form"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New User Password", type="password")
        confirm_password = st.text_input("Confirm New User Password", type="password")
        role = st.selectbox("Role", ["User", "Admin"])
        
        submitted = st.form_submit_button("Create User")
        if submitted:
            if new_password == confirm_password:
                if create_user(st.session_state.username, new_username, new_password, role):
                    st.success(f"User {new_username} created successfully.")
                    st.rerun()  # Refresh the page to show the new user
                else:
                    st.error(f"Failed to create user {new_username}. User may already exist or you don't have admin privileges.")
            else:
                st.error("Passwords do not match.")
    
    # Reset password section
    st.subheader("Reset User Password")
    with st.form("reset_password_form"):
        username_to_reset = st.selectbox("Select User", [u for u in users.keys() if u != st.session_state.username])
        new_password = st.text_input("New Password", type="password", key="reset_pw")
        confirm_password = st.text_input("Confirm New Password", type="password", key="reset_confirm")
        
        submitted = st.form_submit_button("Reset Password")
        if submitted:
            if new_password == confirm_password:
                if reset_user_password(st.session_state.username, username_to_reset, new_password):
                    st.success(f"Password for {username_to_reset} reset successfully.")
                else:
                    st.error(f"Failed to reset password for {username_to_reset}.")
            else:
                st.error("Passwords do not match.")
    
    # Delete user section
    st.subheader("Delete User")
    with st.form("delete_user_form"):
        username_to_delete = st.selectbox("Select User to Delete", [u for u in users.keys() if u != st.session_state.username])
        confirm_delete = st.checkbox("I confirm I want to delete this user")
        
        submitted = st.form_submit_button("Delete User")
        if submitted:
            if confirm_delete:
                if delete_user(st.session_state.username, username_to_delete):
                    st.success(f"User {username_to_delete} deleted successfully.")
                    st.rerun()  # Refresh the page to show the updated user list
                else:
                    st.error(f"Failed to delete user {username_to_delete}.")
            else:
                st.error("Please confirm deletion by checking the box.")
    
    # Import/Export Users
    st.subheader("Import/Export Users")
    
    # Export users
    if st.button("Export Users to CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="users.csv",
            mime="text/csv"
        )
    
    # Import users
    st.subheader("Import Users from CSV")
    st.write("CSV should have columns: Username, Password, Role")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            import_df = pd.read_csv(uploaded_file)
            if st.button("Import Users"):
                success_count = 0
                for _, row in import_df.iterrows():
                    username = row["Username"]
                    password = row["Password"]
                    role = row["Role"]
                    
                    if create_user(st.session_state.username, username, password, role):
                        success_count += 1
                
                st.success(f"Successfully imported {success_count} users out of {len(import_df)}.")
                st.rerun()
        except Exception as e:
            st.error(f"Error importing users: {str(e)}")