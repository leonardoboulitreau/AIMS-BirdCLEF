import os
from dupes_config import dupes

# Define the base directory where the folders like 'asbfly', 'grewar3' etc., are located
base_dir = '../input/birdclef-2024/train_audio/' # May be different depending on how you are working with the di

# Function to delete files
def delete_files(file):
    full_path = os.path.join(base_dir, file)
    try:
        os.remove(full_path)
        print(f"Deleted {full_path}")
    except FileNotFoundError:
        print(f"File not found: {full_path}")
    except Exception as e:
        print(f"Error when deleting {full_path}: {str(e)}")

# Iterate over each tuple in the set
for files_tuple in dupes:
    delete_files(files_tuple[0])
