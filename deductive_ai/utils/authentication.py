import streamlit as st
import json
import bcrypt
import os
import pathlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))

# Get storage path from environment variable, or use default paths
STORAGE_PATH = os.environ.get("INFERIQ_STORAGE", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Path to the JSON file that stores user credentials
USER_DATA_FILE = os.path.join(STORAGE_PATH, "users.json")

# Function to reset the users file with a default admin user
def reset_users_file():
    # Create a fresh admin user with a known password
    admin_password = "admin"
    hashed_password = bcrypt.hashpw(admin_password.encode(), bcrypt.gensalt()).decode()
    
    default_user = {
        "admin": {
            "password": hashed_password,
            "role": "Admin",
            "first_login": True
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(USER_DATA_FILE), exist_ok=True)
    
    # Write to the file
    with open(USER_DATA_FILE, "w") as f:
        json.dump(default_user, f, indent=4)
    
    return default_user

# Function to load users from the JSON file
def load_users():
    # If file doesn't exist, create it with default admin user
    if not os.path.exists(USER_DATA_FILE):
        return reset_users_file()
    
    # Try to load existing users
    try:
        with open(USER_DATA_FILE, "r") as f:
            content = f.read().strip()
            # Check if file is empty or contains just {}
            if not content or content == "{}":
                # Initialize with default admin user
                return reset_users_file()
            
            # Parse JSON content
            return json.loads(content)
    except json.JSONDecodeError:
        # If JSON is invalid, reset the file with default admin user
        st.warning("Invalid users.json file detected. Resetting with default admin user.")
        return reset_users_file()

# Function to save users to the JSON file
def save_users(users):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(USER_DATA_FILE), exist_ok=True)
    
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users, f, indent=4)

# Function to authenticate a user
def authenticate(username, password):
    users = load_users()
    if username in users:
        hashed_pw = users[username]["password"].encode()
        if bcrypt.checkpw(password.encode(), hashed_pw):
            return True, users[username]["role"], users[username]["first_login"]
    return False, None, None

# Function to change a user's password
def change_password(username, new_password):
    users = load_users()
    if username in users:
        users[username]["password"] = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        users[username]["first_login"] = False
        save_users(users)
        return True
    return False

# Function to create a new user (Admin only)
def create_user(admin_username, new_username, new_password, role):
    users = load_users()
    if admin_username in users and users[admin_username]["role"] == "Admin":
        if new_username not in users:
            users[new_username] = {
                "password": bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode(),
                "role": role,
                "first_login": True
            }
            save_users(users)
            return True
    return False

# Function to delete a user (Admin only)
def delete_user(admin_username, username_to_delete):
    users = load_users()
    if admin_username in users and users[admin_username]["role"] == "Admin":
        if username_to_delete in users and username_to_delete != admin_username:
            del users[username_to_delete]
            save_users(users)
            return True
    return False

# Function to reset a user's password (Admin only)
def reset_user_password(admin_username, username_to_reset, new_password):
    users = load_users()
    if admin_username in users and users[admin_username]["role"] == "Admin":
        if username_to_reset in users:
            users[username_to_reset]["password"] = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
            users[username_to_reset]["first_login"] = True
            save_users(users)
            return True
    return False

# Function to register a new user (if self-registration is enabled)
def register_user(new_username, new_password, registration_code=None):
    # Check if registration code is required and valid
    # This is a simple implementation - you can enhance it as needed
    required_code = os.environ.get("REGISTRATION_CODE", "")
    if required_code and required_code != registration_code:
        return False, "Invalid registration code"
    
    users = load_users()
    if new_username in users:
        return False, "Username already exists"
    
    # By default, new self-registered users get the "User" role
    users[new_username] = {
        "password": bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode(),
        "role": "User",
        "first_login": True
    }
    save_users(users)
    return True, "User registered successfully"
