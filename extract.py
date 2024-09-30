import tarfile
import os

print(os.getcwd())

file_path = r"rt-polaritydata.tar.gz"
destination_directory = r"Data"

try:
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    # Open the .tar.gz file
    with tarfile.open(file_path, 'r:gz') as tar:
        # Extract all the contents into the destination directory
        tar.extractall(path=destination_directory)
        print("Extraction completed successfully.")

except FileNotFoundError as fnf_error:
    print(f"Error: {fnf_error}")

except tarfile.ReadError:
    print("Error: The file could not be read or is not a valid .tar.gz archive.")

except PermissionError:
    print("Error: You do not have the necessary permissions to extract files to this directory.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
