import json
import re
import string
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy.stats import pearsonr

PUNCS = set(list(string.punctuation))
LABEL_MAP = {"A": "ans0", "B": "ans1", "C": "ans2"}

def entropy(counts):
    # Only consider non-zero counts
    total = sum(counts.values())  # Sum of non-zero counts
    probs = np.array([v / total for v in counts.values() if v > 0])  # Probabilities for non-zero counts
    H=-np.sum(probs * np.log2(probs))
    probs_max = np.array([1 /3, 1/3, 1/3])  # Probabilities for non-zero counts
    H_max=-np.sum(probs_max * np.log2(probs_max))
    return H/H_max

def eval_contamination(file):
    with open(file, "r") as f:
        lines = f.read().split("\n")
    consist_res = {}

    for i in range(len(lines)):
        p = lines[i].strip()
        # for consistency
        idx = str(i // 672)
        # divide by 672 for the 3 cyclic permutations
        if idx not in consist_res:
            consist_res[idx] = {'A':0, 'B':0, 'C':0}
        consist_res[idx][p]+=1
    
    return [entropy(counts) for counts in consist_res.values()]

def print_latex_cyclic_entropy(cyclic):
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

    with open("eval_config.json", "r") as f:
        eval_config = json.load(f)
    eval_base_file = eval_config["eval_base_file"]

    with open(eval_base_file, "r") as f:
        jsonl_data = [json.loads(line) for line in f.readlines()]

    correct_labels = [entry["label"] for entry in jsonl_data]

    file_dir = Path(args.result_dir)
    folders = [f.name for f in file_dir.iterdir() if f.is_dir() and any(file.suffix == '.txt' for file in f.glob("*.txt"))]
    dfs = []
    
    for folder in folders:
        folder_path=Path(args.result_dir)/Path(folder)
        files = folder_path.glob("*.txt")
        for file in files:
            entropies=eval_contamination(file)
            break #only doing one file
        df = pd.DataFrame({
            'model': str(folder),
            'cyclic_entropy': entropies,
            })
        dfs.append(df)

    output_path = file_dir / "summary"
    output_path.mkdir(exist_ok=True)
    all_result=pd.concat(dfs)

    # Output the correlation
    cyclic = all_result.groupby("model")["cyclic_entropy"].mean().sort_values(ascending=False)
    
    print(f"Avg cyclic entropy per model")
    print(cyclic)

    print_latex_cyclic_entropy(cyclic)
        
    all_result.to_csv(file_dir / "summary" / "contamination.csv", index=False)


