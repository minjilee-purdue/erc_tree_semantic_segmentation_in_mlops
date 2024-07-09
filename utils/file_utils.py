# file_utils.py

import os
import shutil

def create_directory(directory_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def copy_file(source_file, destination_file):
    """Copy a file from source to destination."""
    try:
        shutil.copy2(source_file, destination_file)
        print(f"File copied from {source_file} to {destination_file}")
    except FileNotFoundError:
        print(f"Error: File not found - {source_file}")

def move_file(source_file, destination_dir):
    """Move a file from source to destination directory."""
    try:
        shutil.move(source_file, destination_dir)
        print(f"File moved from {source_file} to {destination_dir}")
    except FileNotFoundError:
        print(f"Error: File not found - {source_file}")

def delete_file(file_path):
    """Delete a file if it exists."""
    try:
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")

def list_files(directory):
    """List all files in a directory."""
    files = os.listdir(directory)
    return files

# Example usage
if __name__ == "__main__":
    # Create a directory
    create_directory("data")

    # Copy a file
    copy_file("source.txt", "destination.txt")

    # Move a file
    move_file("file.txt", "new_directory")

    # Delete a file
    delete_file("file_to_delete.txt")

    # List files in a directory
    files_list = list_files("data")
    print("Files in 'data' directory:", files_list)

