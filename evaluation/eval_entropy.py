"""This script is to analyze the distribution of answers regarding different prompts."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from statsmodels.stats.inter_rater import fleiss_kappa

def compute_entropy(probas_list):
    assert all(np.all((0 <= probs) & (probs <= 1)) for probs in probas_list), "Probabilities must be in the range [0,1]"
    return [-np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0))) for probs in probas_list]

def ans_entropy(responses):
    """
    Compute the probability distribution and entropy for each response across multiple files.
    
    Parameters:
    files (list): List of file paths containing responses (assumed to be in 'A', 'B', or 'C').
    
    Returns:
    tuple: A tuple containing:
        - prob_dist_list: List of probability distributions for each response.
        - entropies: List of entropy values for each response.
    """
    prob_dist_list = []
    num_files=len(responses)
    num_responses = len(responses[0])
    assert all(len(r) == num_responses for r in responses), "Files have different response counts!" 

    for i in range(num_responses): # loop through each line (question) to calculate probability distributions and entropy
        response_set = [responses[f][i] for f in range(num_files)]  # Collect responses for the i-th question from all files
        
        # Count occurrences of each possible response ('A', 'B', 'C')
        letter_counts = {letter: 0 for letter in ["A", "B", "C"]}
        for response in response_set:
            if None not in response_set and '' not in response_set:
            #if response !='':
                letter_counts[response] += 1
        
        # Convert letter counts to a normalized probability distribution
        prob_dist = [letter_counts.get(letter, 0) / num_files for letter in ["A", "B", "C"]]
        prob_dist_list.append(prob_dist)
    
    entropies=compute_entropy(np.array(prob_dist_list)) # Compute entropy for each probability distribution
    return prob_dist_list, entropies

def compute_fleiss_kappa(responses):
    """
    responses: a dict of {file_index: list of responses} with categorical answers 'A', 'B', or 'C'
    """
    letter_to_index = {'A': 0, 'B': 1, 'C': 2 }
    num_responses = len(responses[0])  # number of examples
    num_files=len(responses)
    n_categories = 4  # 'A', 'B', 'C'

    fleiss_data = np.zeros((num_responses, n_categories), dtype=int)
    fleiss_data = []

    # Loop through each example (question)
    for i in range(num_responses):  # loop through each example (row)
        response_set = [responses[f][i] for f in range(num_files)]  # Responses for this example
        
        # Check if all models have responded (no missing responses)
        if None not in response_set and '' not in response_set:
            # If all responses are valid, count occurrences of each response ('A', 'B', 'C')
            counts = [0, 0, 0]  # For 'A', 'B', 'C'
            for response in response_set:
                if response in letter_to_index:
                    counts[letter_to_index[response]] += 1
            fleiss_data.append(counts)
    
    fleiss_data = np.array(fleiss_data)
    kappa = fleiss_kappa(fleiss_data)
    return kappa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="./result")
    args = parser.parse_args()

    #Getting all folders that exist in the result directory with a uncertainty subfolder
    file_dir = Path(args.result_dir)
    folders = [
            f.name for f in file_dir.iterdir() 
            if f.is_dir() and 
            #any(subfolder.name == "uncertainty" and subfolder.is_dir() for subfolder in f.iterdir()) and
            any(file.suffix == ".txt" for file in f.glob("*.txt"))
        ]
    
    modifs_list=['', 'prepositions', 'AAE']
    
    dfs = []
    for folder in folders:
        for modif in modifs_list:
            folder_path=Path(args.result_dir)/folder#/"uncertainty"

            #Retrieving the corresponding results per prompt strategy for the current model
            result_folder="./result/"+folder
            files = list(Path(result_folder).glob("*.txt"))

            responses = [path.read_text().split("\n") for path in files if modif in str(path)] # Read and process all files into a list of responses 
            num_files=len(responses)
        
            num_responses = len(responses[0])
            assert all(len(r) == num_responses for r in responses), "Files have different response counts!" 

            #Computing answer distribution and corresponding entropy
            prob_list, ans_entropies = ans_entropy(responses)

            if prob_list is not None:
                #Comparing the distributions
                df = pd.DataFrame({
                    'model': str(folder),
                    "modif": modif if len(modif)>1 else 'all',
                    "distribution":prob_list,
                    'entropy': ans_entropies,
                    'kappa':compute_fleiss_kappa(responses)
                    })
                dfs.append(df)

    # Saving results
    output_path = file_dir / "summary"
    output_path.mkdir(exist_ok=True)
    all_result=pd.concat(dfs)
    all_result.to_csv(file_dir / "summary" / "prompt_distributions.csv", index=False)
    


