"""This script is for preliminary analysis. 
The goal is to compare the answers to one prompt with cyclic permutations of the answer."""

import json
import re
import string
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy.stats import pearsonr

def entropy(counts):
    """
    Computes the normalized entropy of a probability distribution.

    Parameters:
    counts (dict): A dictionary where keys represent categories and values represent counts.

    Returns:
    float: Normalized entropy value between 0 and 1.
    """
    total = sum(counts.values())  # Sum of non-zero counts
    probs = np.array([v / total for v in counts.values() if v > 0])  # Probabilities for non-zero counts
    H=-np.sum(probs * np.log2(probs))

    # Compute maximum entropy for a uniform distribution of 3 categories
    probs_max = np.array([1 /3, 1/3, 1/3])  
    H_max=-np.sum(probs_max * np.log2(probs_max))
    
    # Normalize entropy by dividing by the maximum entropy
    return H/H_max

def eval_fixed_answer(file):
    """
    Evaluates contamination in a dataset by computing entropy of cyclic permutations.

    Parameters:
    file (str): Path to the input file.

    Returns:
    list: A list of normalized entropy values.
    """
    with open(file, "r") as f:
        lines = f.read().split("\n")

    #For each unique example, we look at the answers when permutating the answers
    cyclic_res = {}
    for i in range(len(lines)):
        p = lines[i].strip()
        idx = str(i // 672) # divide by 672 for the 3 cyclic permutations
        if idx not in cyclic_res:
            cyclic_res[idx] = {'A':0, 'B':0, 'C':0}
        cyclic_res[idx][p]+=1
    
    # Compute entropy for each unique example and return the list
    return [entropy(counts) for counts in cyclic_res.values()]

def print_latex_table(cyclic):
    print(r"\begin{table}[h]")
    print(r"    \centering")
    print(r"    \begin{tabular}{l c}")
    print(r"        \toprule")
    print(r"        \textbf{Model} & \textbf{Avg Normalized Entropy} \\")
    print(r"        \midrule")
    
    for model, entropy in cyclic.items():
        print(f"        {model} & {entropy:.4f} \\\\")
    
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"    \caption{Average Normalized Entropy with cyclic permutations per model.}")
    print(r"\end{table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="./result")
    args = parser.parse_args()

    #Getting all folders that exist in the result directory with .txt files
    file_dir = Path(args.result_dir)
    folders = [f.name for f in file_dir.iterdir() if f.is_dir() and any(file.suffix == '.txt' for file in f.glob("*.txt"))]
    
    dfs = []
    for folder in folders:
        folder_path=Path(args.result_dir)/Path(folder)
        files = folder_path.glob("*.txt")
        for file in files:
            entropies=eval_fixed_answer(file)
            break # Only doing one file 

        df = pd.DataFrame({
            'model': str(folder),
            'cyclic_entropy': entropies,
            })
        dfs.append(df)

    #Saving output
    output_path = file_dir / "summary"
    output_path.mkdir(exist_ok=True)
    all_result=pd.concat(dfs)
    all_result.to_csv(file_dir / "summary" / "fixed_answer.csv", index=False)

    # Output the correlation
    cyclic = all_result.groupby("model")["cyclic_entropy"].mean().sort_values(ascending=False)
    print(f"Avg cyclic entropy per model")
    print(cyclic)

    #Printing for latex
    print_latex_table(cyclic)