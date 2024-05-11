import os
import subprocess
import pandas as pd

# Directory containing your audio files
audio_dir_unlabeled = 'input/birdclef2024/birdclef-2024/unlabeled_soundscapes/'

# Directory of the species list
species_dir = 'species_list.txt'

# Paths to the BirdNET scripts
analyze_path = "BirdNET-Analyzer/analyze.py"
embedding_path = "BirdNET-Analyzer/embeddings.py"

# Output directory for analysis results
output_dir = "Pedro/out_analyzer"
embedding_out_dir = "Pedro/out_analyzer/embeddings"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating output directory at {output_dir}")
else:
    print(f"Output directory already exists at {output_dir}")

# Path to the final results file
final_results_path = os.path.join(output_dir, "final_results_unlabeled.csv")
print(f"Results will be saved incrementally to {final_results_path}")

# Initialize a flag to track whether the header should be written
header_written = False

# Walk through the directory structure
for root, dirs, files in sorted(os.walk(audio_dir_unlabeled)):
    for file in sorted(files):
        file_path = os.path.join(root, file)
        print(f"Analyzing: {file_path}")

        # Run the analysis
        subprocess.run(['python', analyze_path, '--i', file_path, '--o', output_dir, '--slist', species_dir, "--min_conf", "0.3", "--threads", "50"], check=True)

        # Generate embeddings
        embeddings_output_path = os.path.join(embedding_out_dir, os.path.splitext(file)[0] + '.embeddings_unlabeled.txt')
        subprocess.run(['python', embedding_path, '--i', file_path, '--o', embeddings_output_path, "--threads", "50"], check=True)

        # Read the embeddings output to get the third column
        try:
            embeddings_df = pd.read_csv(embeddings_output_path, sep="\t", header=None, usecols=[2])
            embeddings_column = embeddings_df.iloc[:, 0]  # Get the third column data
        except Exception as e:
            print(f"Failed to read embeddings from {embeddings_output_path}: {e}")
            embeddings_column = pd.Series([None] * len(df))  # Handle missing data

        # Read analysis results and append embeddings data
        result_file_path = os.path.join(output_dir, ".BirdNET.selection.table.txt")
        print(f"Reading results from {result_file_path}")
        try:
            df = pd.read_csv(result_file_path, sep="\t")
            df['source_file'] = file
            df['embeddings_data'] = embeddings_column  # Add embeddings data as a new column

            # Save each new DataFrame immediately without storing them in final_df
            df.to_csv(final_results_path, mode='a', index=False, header=not header_written)
            if not header_written:
                header_written = True  # Ensure header is not written again
        except Exception as e:
            print(f"Failed to read {result_file_path}: {e}")

print("Analysis complete. All results have been incrementally saved.")





