"""This script is for preliminary analysis. 
The goal is to compare the distribution of log-likelihoods with the distribution of answers to the different prompts."""

import json
import re
import string
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy.stats import pearsonr

def compute_entropy(probas_list):
    assert all(np.all((0 <= probs) & (probs <= 1)) for probs in probas_list), "Probabilities must be in the range [0,1]"
    return [-np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0))) for probs in probas_list]

def softmax(logits):
    exp_values = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exp_values / np.sum(exp_values)

def log_entropy(file):
    """
    Reads log-likelihood values from a file, computes entropy, and returns the results.

    Parameters:
    file (str): Path to the file containing log-likelihood values. 

    Returns:
    tuple: A tuple containing:
        - entropies (array-like): The computed entropy values.
        - log_likelihoods (list of numpy arrays): The parsed log-likelihood values.

    """
    log_likelihoods = []
    with open(file, "r") as f:
        for line in f:
            log_probs=np.array(eval(line.strip())) # Convert the string to a list of floats
            log_likelihoods.append(softmax(log_probs)) #Apply softmax to have probabilities
    entropies = compute_entropy(log_likelihoods)
    return log_likelihoods, entropies

def ans_entropy(files):
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
    responses = [path.read_text().strip().split("\n") for path in files] # Read and process all files into a list of responses 
    num_files=len(responses)
    
    if num_files==0: # if there are no files, return None
        return None, None
    
    num_responses = len(responses[0])
    assert all(len(r) == num_responses for r in responses), "Files have different response counts!" 

    for i in range(num_responses): # loop through each line (question) to calculate probability distributions and entropy
        response_set = [responses[f][i] for f in range(num_files)]  # Collect responses for the i-th question from all files
        
        # Count occurrences of each possible response ('A', 'B', 'C')
        letter_counts = {letter: 0 for letter in ["A", "B", "C"]}
        for response in response_set:
            letter_counts[response] += 1
        
        # Convert letter counts to a normalized probability distribution
        prob_dist = np.array([letter_counts.get(letter, 0) / num_files for letter in ["A", "B", "C"]])
        prob_dist_list.append(prob_dist)
    
    entropies=compute_entropy(prob_dist_list) # Compute entropy for each probability distribution
    return prob_dist_list, entropies

def compare_distribution(prob_dist, log_likelihoods, epsilon=1e-10):
    '''Compute kl divergence for each answer'''
    prob_dist = np.array(prob_dist)
    log_likelihoods = np.array(log_likelihoods)
    
    # Ensure no zeros in probability distributions
    prob_dist = np.clip(prob_dist, epsilon, 1)  
    log_likelihoods = np.clip(log_likelihoods, epsilon, 1)

    kl_values = np.sum(prob_dist * np.log(prob_dist / log_likelihoods), axis=1)
    
    return np.nan_to_num(kl_values)

def clean_model_name(model_name):
    match = re.search(r"Llama-\d+[a-zA-Z0-9-]*", model_name)
    return match.group(0) if match else model_name

def print_latex_table(correlations, kl):
    print(r"\begin{table}[h]")
    print(r"    \centering")
    print(r"    \begin{tabular}{l c c}")
    print(r"        \toprule")
    print(r"        \textbf{Model} & \textbf{Pearson Correlation} & \textbf{Avg KL Divergence} \\")
    print(r"        \midrule")
    
    for model in correlations.index:
        clean_name = clean_model_name(model)
        print(f"        {clean_name} & {correlations[model]:.3f} & {kl[model]:.3f} \\\\")
    
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"    \caption{Comparing log-likelihood distribution and prompt answers distribution: pearson correlation between both entropies and average KL divergence.}")
    print(r"\end{table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="./result")
    args = parser.parse_args()

    #Getting all folders that exist in the result directory with a uncertainty subfolder
    file_dir = Path(args.result_dir)
    folders = [
            f.name for f in file_dir.iterdir() 
            if f.is_dir() and 
            any(subfolder.name == "uncertainty" and subfolder.is_dir() for subfolder in f.iterdir()) and
            any(file.suffix == ".txt" for file in f.glob("*.txt"))
        ]
    
    dfs = []
    for folder in folders:
        folder_path=Path(args.result_dir)/folder/"uncertainty"

        #Retrieving log-likelihoods and computing entropy
        files_uncertainty = folder_path.glob("*.txt")
        for file in files_uncertainty:
            log_likelihoods, log_entropies=log_entropy(file)

        #Retrieving the corresponding results per prompt strategy for the current model
        result_folder="./result/"+folder
        files = list(Path(result_folder).glob("*.txt"))

        #Computing answer distribution and corresponding entropy
        prob_list, ans_entropies = ans_entropy(files)

        if prob_list is not None:
            #Comparing the distributions
            kl_divergences=compare_distribution(prob_list, log_likelihoods)
            df = pd.DataFrame({
                'model': str(folder),
                'Answer_Entropy': ans_entropies,
                'Log_Entropy': log_entropies,
                'KL': kl_divergences
                })
            dfs.append(df)

    # Saving results
    output_path = file_dir / "summary"
    output_path.mkdir(exist_ok=True)
    all_result=pd.concat(dfs)
    all_result.to_csv(file_dir / "summary" / "entropies.csv", index=False)

    # Computing the correlation and avg KL
    correlations = all_result.groupby("model").apply(lambda g: pearsonr(g["Answer_Entropy"], g["Log_Entropy"])[0])
    kl = all_result.groupby("model")["KL"].mean()

    print(f"Pearson correlation between loglikelihood entropy and prompts entropy:")
    print(correlations)
    
    print(f"\nAvg KL divergence per model")
    print(kl)

    #Printing for latex
    print_latex_table(correlations, kl)
        
    


