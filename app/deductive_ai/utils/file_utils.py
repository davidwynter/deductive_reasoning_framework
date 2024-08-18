import json
from pathlib import Path

# Base directory for storing JSON files
JSON_DIR = Path("data_management")

def setup_json_directory():
    """
    Ensure that the directory for storing JSON data exists.
    """
    JSON_DIR.mkdir(parents=True, exist_ok=True)

def save_json(data, filename):
    """
    Save data to a JSON file.
    
    :param data: The data to be saved (as a Python dictionary).
    :param filename: The name of the file to save the data to (without extension).
    """
    setup_json_directory()
    file_path = JSON_DIR / f"{filename}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(filename):
    """
    Load data from a JSON file.
    
    :param filename: The name of the file to load the data from (without extension).
    :return: The loaded data as a Python dictionary.
    """
    file_path = JSON_DIR / f"{filename}.json"
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def delete_json(filename):
    """
    Delete a JSON file.
    
    :param filename: The name of the file to delete (without extension).
    """
    file_path = JSON_DIR / f"{filename}.json"
    if file_path.exists():
        file_path.unlink()
