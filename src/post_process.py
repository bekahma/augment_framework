'''Some paraphrased rows may have been filtered out due to quality issues. To ensure proper alignment for evaluation after LLM inference, we need to restore the original row structure by reinserting empty placeholders where data was removed.'''

import numpy as np
import pickle
import shutil
import pandas as pd
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="./result")
    args = parser.parse_args()

    #Search all files in result_dir
    file_dir = Path(args.result_dir)
    files = file_dir.glob("*/*.txt")

    #Paths to data
    DATA_FOLDER='./data/paraphrases/'
    MAP_IDS_TO_ROW= DATA_FOLDER+"map_idx.pkl"
    
    # Load the mapping from question IDs to the corresponding row indices in the full dataset
    # Each ID corresponds to one or more rows that came from a single source question
    with open(MAP_IDS_TO_ROW, 'rb') as f:
        map_idx=pickle.load(f)

    for file in files:
        file_path = str(file)
        path_parts = file_path.split("_")

        # Extract the useful parts (modification and model used for paraphrase)
        model = path_parts[-1].split(".")[0]  
        modification = path_parts[-2]  
        
        # Skip the original dataset (there are no model so [-1] is the modification and [-2]=="prompt")
        if modification!="prompt": 
            print(file_path)
            print(modification)
            print(model)
            
            #Retrieve the list of indices to keep
            IDS_KEEP= DATA_FOLDER+f"Gender_identity_{modification}_{model}_ids.pkl"
            with open(IDS_KEEP, 'rb') as f:
                ids_to_keep=pickle.load(f)

            # Read lines from the result file
            with open(file, 'r') as f:
                text_lines = [line.strip() for line in f]
            
            # Initialize a full-sized placeholder for all 1872 rows
            # Missing entries (filtered-out paraphrases) are initialized as NaN
            full_series = pd.Series([np.nan] * 1872, dtype=object)

            # Fill in the positions of the kept paraphrases based on the original row mapping
            start = 0
            for i in ids_to_keep: #each i corresponds to a question
                s, e = map_idx[i] # Get the original row range [s, e) of instances for this paraphrased question
                full_series[s:e] = text_lines[start:start + (e - s)]
                start += (e - s)
            
            # Sanity check: make sure all non-NaN entries match the length of the original file
            assert len(text_lines)==(sum(full_series.notna()))

            # Backup the original output file in a 'raw' subfolder
            raw_dir = file.parent / "raw"
            raw_dir.mkdir(exist_ok=True)
            raw_file_path = raw_dir / file.name
            shutil.move(str(file), str(raw_file_path))

            # Save the realigned output: preserve original row structure with blanks where data was filtered
            with open(file, 'w') as f:
                for i, item in enumerate(full_series):
                    f.write("" if pd.isna(item) else str(item))
                    if i < len(full_series) - 1:
                        f.write("\n")


        