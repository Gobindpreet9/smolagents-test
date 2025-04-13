import hashlib
import json
import os

def get_file_hash(file_path):
    """Calculates the SHA256 hash of a file's content."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as file:
            while chunk := file.read(8192): # Read in chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error hashing file {file_path}: {e}")
        return None

def load_metadata(metadata_path):
    """Loads filehash metadata from a JSON file."""
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading metadata {metadata_path}: {e}")
        return None

def save_metadata(metadata_path, data):
    """Saves filehash metadata to a JSON file."""
    try:
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving metadata {metadata_path}: {e}")

def load_openapi_docs(docs_path):
    """Load all OpenAPI documents from the specified directory."""
    openapi_docs = {}
    for file_name in os.listdir(docs_path):
        file_path = os.path.join(docs_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                openapi_docs[file_name] = file.read()
    return openapi_docs