import streamlit as st
import json
import bcrypt
import os

# Path to the JSON file that stores user credentials
USER_DATA_FILE = "users.json"

# Function to load users from the JSON file
def load_users():
    if not os.path.exists(USER_DATA_FILE):
        # If the file doesn't exist, create it with a default admin user
        default_user = {
            "admin": {
                "password": bcrypt.hashpw("admin".encode(), bcrypt.gensalt()).decode(),
                "role": "Admin",
                "first_login": True
            }
        }
        with open(USER_DATA_FILE, "w") as f:
            json.dump(default_user, f, indent=4)
    with open(USER_DATA_FILE, "r") as f:
        return json.load(f)

# Function to save users to the JSON file
def save_users(users):
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

