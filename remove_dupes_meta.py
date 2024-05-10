from dupes_config import dupes
import pandas as pd
import os

# Define the directory where the metadata is and try to transform it into a dataframe
meta_path = '../input/birdclef-2024/train_metadata.csv'
try:
    meta_df = pd.read_csv(meta_path)
except Exception as e:
    print(f"Error reading the metadata CSV: {e}")
    exit()  # Exits the script if the file cannot be read

# Select the first elements of each tuple
files_to_remove = [file_tuple[0] for file_tuple in dupes]

# Create a mask to remove the selected files from the metadata dataframe
try:
    mask = meta_df['filename'].map(lambda x: all(word not in x for word in files_to_remove))
    meta_df_filtered = meta_df[mask]
except Exception as e:
    print(f"Error applying the mask or filtering the DataFrame: {e}")
    exit()

# Define the output directory and save the filtered dataframe
output_directory = '../input/birdclef-2024'
file_name = 'train_metadata_filtered.csv'
full_path = f"{output_directory}/{file_name}"

# Ensure the directory exists before saving
if not os.path.exists(output_directory):
    try:
        os.makedirs(output_directory)
    except Exception as e:
        print(f"Error creating directory {output_directory}: {e}")
        exit()

# Try to save the filtered DataFrame to a CSV file
try:
    meta_df_filtered.to_csv(full_path, index=False)
    print(f"Filtered DataFrame saved successfully to {full_path}")
except Exception as e:
    print(f"Error saving the DataFrame: {e}")
