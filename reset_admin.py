#!/usr/bin/env python3
"""
Reset the users.json file with a default admin user.
Run this script if you need to reset the admin password or if you're having issues with the login system.
"""

import json
import bcrypt
import os
import pathlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(pathlib.Path(__file__).parent, '.env'))

# Get storage path from environment variable, or use default paths
STORAGE_PATH = os.environ.get("INFERIQ_STORAGE", pathlib.Path(__file__).parent)

# Path to the JSON file that stores user credentials
USER_DATA_FILE = os.path.join(STORAGE_PATH, "users.json")

def reset_admin_user():
    """Reset the users.json file with a default admin user."""
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
    
    print(f"Admin user reset successfully. You can now login with:")
    print("Username: admin")
    print("Password: admin")
    print("The file was saved to:", USER_DATA_FILE)

if __name__ == "__main__":
    reset_admin_user()