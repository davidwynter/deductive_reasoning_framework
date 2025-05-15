import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))

# Get storage path from environment variable, or use default paths
STORAGE_PATH = os.environ.get("INFERIQ_STORAGE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Base directory for storing JSON files
JSON_DIR = Path(os.path.join(STORAGE_PATH, "data_management"))
UPLOAD_DIR = Path(os.path.join(STORAGE_PATH, "uploaded_files"))

def setup_directories():
    """
    Create necessary directories if they do not exist.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / "ttl").mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / "n3").mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / "xml").mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / "txt").mkdir(parents=True, exist_ok=True)

def save_uploaded_file(uploaded_file, file_type):
    """
    Save an uploaded file to the appropriate directory.
    
    :param uploaded_file: The uploaded file object.
    :param file_type: The type/format of the file (e.g., "ttl", "n3", "xml", "txt").
    :return: Path to the saved file.
    """
    setup_directories()  # Ensure directories are created
    file_path = UPLOAD_DIR / file_type / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def list_files(file_type=None):
    """
    List files in the upload directory. Optionally filter by file type.
    
    :param file_type: The type/format of the files to list (e.g., "ttl", "n3", "xml").
    :return: List of file paths relative to the upload directory.
    """
    setup_directories()  # Ensure directories are created
    if file_type:
        directory = UPLOAD_DIR / file_type
    else:
        directory = UPLOAD_DIR

    files = [str(file.relative_to(UPLOAD_DIR)) for file in directory.glob("**/*") if file.is_file()]
    return files

def delete_file(file_path):
    """
    Delete a file from the file system.
    
    :param file_path: The relative path to the file to delete.
    """
    full_path = UPLOAD_DIR / file_path
    if full_path.exists():
        full_path.unlink()
        
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
