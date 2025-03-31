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

def compute_entropy(log_likelihoods):
    entropies = []
    for log_probs in log_likelihoods:
        probs = np.exp(log_probs - np.max(log_probs))  # Apply softmax (stable version)
        probs /= np.sum(probs)
        entropy = -np.sum(probs * np.log(probs))
        entropies.append(entropy)
    return entropies

def eval_entropy(file):
    log_likelihoods = []
    with open(file, "r") as f:
        for line in f:
            # Convert the string to a list of floats
            log_likelihoods.append(np.array(eval(line.strip())))
    entropies = compute_entropy(log_likelihoods)

    return entropies, log_likelihoods

def compute_variance(files, log_likelihoods):
    probs_list = [np.exp(log_probs - np.max(log_probs)) / np.sum(np.exp(log_probs - np.max(log_probs))) for log_probs in log_likelihoods]
    response_variances = []
    kl_divergences = []
    responses = [path.read_text().strip().split("\n") for path in files]
    if len(responses)==0:
        return None, None
    num_responses = len(responses[0])
    assert all(len(r) == num_responses for r in responses), "Files have different response counts!"

    for i in range(num_responses):
        response_set = [responses[f][i] for f in range(len(responses))]  # Collect corresponding responses
        
        # Count occurrences of each letter
        letter_counts = {letter: 0 for letter in ["A", "B", "C"]}
        for response in response_set:
            letter_counts[response] += 1
        
        # Convert to probability distribution (normalized)
        prob_dist = np.array([letter_counts.get(letter, 0) / len(responses) for letter in ["A", "B", "C"]])
        epsilon = 1e-10
        prob_dist = np.maximum(prob_dist, epsilon)

        # Compute entropy
        entropy = -np.sum(prob_dist * np.log2(prob_dist))  # Log base 2 for bits
        response_variances.append(entropy)

        # Compute kl-divergence
        kl_div = np.sum(prob_dist * np.log(prob_dist / probs_list[i]))
        kl_divergences.append(kl_div)

    return response_variances, kl_divergences

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
    parser.add_argument("--result_dir", default="./result_uncertainty")
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
        files_uncertainty = folder_path.glob("*.txt")
        for file in files_uncertainty:
            entropies, log_likelihoods=eval_entropy(file)
        result_folder="./result/"+folder
        files = list(Path(result_folder).glob("*.txt"))
        response_variances, kl_divergences=compute_variance(files, log_likelihoods)
        if response_variances is not None and kl_divergences is not None:
            df = pd.DataFrame({
                'model': str(folder),
                'Variance': response_variances,
                'Entropy': entropies,
                'KL': kl_divergences
                })
            dfs.append(df)

    output_path = file_dir / "summary"
    output_path.mkdir(exist_ok=True)
    all_result=pd.concat(dfs)

    # Output the correlation
    correlations = all_result.groupby("model").apply(lambda g: pearsonr(g["Variance"], g["Entropy"])[0])
    kl = all_result.groupby("model")["KL"].mean()

    print(f"Pearson correlation between loglikelihood entropy and prompts entropy:")
    print(correlations)
    
    print(f"\nAvg KL divergence per model")
    print(kl)

    print_latex_table(correlations, kl)
        
    all_result.to_csv(file_dir / "summary" / "entropies.csv", index=False)


