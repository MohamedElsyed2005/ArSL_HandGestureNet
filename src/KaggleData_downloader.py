# Required packages (install via pip if not already installed):
# pip install kaggle
# pip install kagglehub

import kagglehub  # For downloading datasets directly from Kaggle
import shutil      # For moving/copying files and folders
import os          # For managing file paths and directories

# ----------------------------
# Download dataset
# ----------------------------
path = kagglehub.dataset_download("sabribelmadoui/arabic-sign-language-unaugmented-dataset")
print("Downloaded path:", path)  # Confirm where the dataset is saved

# ----------------------------
# Set base directory
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Path of this script

# ----------------------------
# Define destination folder and ensure it exists
# ----------------------------
destination_folder = os.path.join(BASE_DIR, "..", "data", "raw")
os.makedirs(destination_folder, exist_ok=True)  # Creates folder if it doesn't exist

# ----------------------------
# Move all files from cache to destination
# ----------------------------
for file_name in os.listdir(path):
    full_file_name = os.path.join(path, file_name)
    target_path = os.path.join(destination_folder, file_name)
    
    # Ensure unique filenames to avoid overwriting
    if os.path.exists(target_path):
        base, ext = os.path.splitext(file_name)
        counter = 1
        while os.path.exists(target_path):
            target_path = os.path.join(destination_folder, f"{base}_{counter}{ext}")
            counter += 1
    
    shutil.move(full_file_name, target_path)  # Move file to destination

print("Files moved to:", destination_folder)  # Final confirmation
