import os
import pandas as pd
import numpy as np
from collections import defaultdict
import itertools

# Configuration
ANNOTATOR1_DIR = "annotations/annotator1"
ANNOTATOR2_DIR = "annotations/annotator2"
ANNOTATOR3_DIR = "annotations/annotator3"
LLM_DIR = "annotations/llm"
COMBINED_DIR = "annotations/combined"
os.makedirs(COMBINED_DIR, exist_ok=True)

COLUMN_INDEX = ord('E') - ord('A')  # 4
UNCERTAIN_IDX = ord('F') - ord('A')  # 5

# Helper functions
def get_file_map(folder, prefix):
    """Match files by modification type + model"""
    files = [f for f in os.listdir(folder) if f.endswith(".xlsx") and f.startswith(prefix)]
    return {f[len(prefix):]: os.path.join(folder, f) for f in files}

def normalize_labels(series):
    """Normalize T/F labels to boolean"""
    mapping = {
        "TRUE": True, "T": True, "1": True, "YES": True,
        "FALSE": False, "F": False, "0": False, "NO": False
    }
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map(mapping)
        .infer_objects(copy=False)
    )

def tf_to_val(s):
    """Convert T/F to 0/1"""
    if s in ("T", True, "1", 1):
        return 1
    elif s in ("F", False, "0", 0):
        return 0
    else:
        return np.nan

def pairwise_agreement(vec1, vec2):
    """
    Compute agreement between two annotation vectors.
    Returns (agreement_rate, num_valid_comparisons)
    """
    mask = ~(np.isnan(vec1) | np.isnan(vec2))
    if mask.sum() == 0:
        return np.nan, 0
    matches = (vec1[mask] == vec2[mask]).sum()
    total = mask.sum()
    return matches / total, total

def parse_llm_filename(fname):
    """
    Parse LLM filename: llm_{modification}_{generator}_{judged}_{judge}.xlsx
    Returns: (modification, generator, judge)
    """
    base = fname.replace("llm_", "").replace(".xlsx", "")
    parts = base.split("_")
    
    # Find "judged" keyword to split
    if "judged" in parts:
        judged_idx = parts.index("judged")
        modification_parts = parts[:judged_idx-1]
        generator = parts[judged_idx-1]
        judge = "_".join(parts[judged_idx+1:])
    else:
        # Fallback: assume last part is judge, second-to-last is generator
        modification_parts = parts[:-2]
        generator = parts[-2]
        judge = parts[-1]
    
    modification = "_".join(modification_parts)
    return modification, generator, judge

def compute_agreement_matrix(annotations_dict, filter_uncertain=True):
    """
    Compute pairwise agreement matrix for all annotators.
    
    annotations_dict: {judge_name: annotation_vector}
    Returns: DataFrame with agreement rates and counts
    """
    judges = list(annotations_dict.keys())
    n = len(judges)
    
    # Initialize matrices
    agreement_rates = np.zeros((n, n))
    vote_counts = np.zeros((n, n), dtype=int)
    
    for i, judge1 in enumerate(judges):
        for j, judge2 in enumerate(judges):
            if i == j:
                agreement_rates[i, j] = 1.0
                vote_counts[i, j] = len(annotations_dict[judge1])
            else:
                rate, count = pairwise_agreement(
                    annotations_dict[judge1],
                    annotations_dict[judge2]
                )
                agreement_rates[i, j] = rate if not np.isnan(rate) else 0.0
                vote_counts[i, j] = count
    
    return agreement_rates, vote_counts, judges

def create_agreement_table(agreement_rates, vote_counts, judges):
    """
    Create a formatted agreement table with rates and counts.
    Returns a DataFrame formatted like the MT-bench table.
    """
    n = len(judges)
    
    # Create multi-index for displaying both rate and count
    data = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append("-")
            elif i > j:
                # Lower triangle: show agreement rate and count
                rate = agreement_rates[i, j]
                count = vote_counts[i, j]
                row.append(f"{rate:.1%}\n({count})")
            else:
                # Upper triangle: empty (or could mirror)
                row.append("")
        data.append(row)
    
    df = pd.DataFrame(data, index=judges, columns=judges)
    return df

def process_modification_model(modification, model, human_files, llm_files_by_judge):
    """
    Process one modification-model combination and compute all agreements.
    """
    # Load human annotations
    suffix = f"{modification}_{model}.xlsx"
    
    a1_file = human_files['a1'].get(suffix)
    a2_file = human_files['a2'].get(suffix)
    a3_file = human_files['a3'].get(suffix)
    
    if not all([a1_file, a2_file, a3_file]):
        return None
    
    df1 = pd.read_excel(a1_file)
    df2 = pd.read_excel(a2_file)
    df3 = pd.read_excel(a3_file)
    
    # Build human annotations dictionary
    annotations = {
        'Human-A1': normalize_labels(df1.iloc[:, COLUMN_INDEX]).map(tf_to_val).to_numpy(),
        'Human-A2': normalize_labels(df2.iloc[:, COLUMN_INDEX]).map(tf_to_val).to_numpy(),
        'Human-A3': normalize_labels(df3.iloc[:, COLUMN_INDEX]).map(tf_to_val).to_numpy(),
    }
    
    # Compute human majority
    H = np.vstack([annotations['Human-A1'], annotations['Human-A2'], annotations['Human-A3']]).T
    human_majority = []
    for row in H:
        valid = row[~np.isnan(row)]
        if len(valid) == 0:
            human_majority.append(np.nan)
        else:
            human_majority.append(1 if valid.sum() > len(valid)/2 else 0)
    annotations['Human-Majority'] = np.array(human_majority)
    
    # Add LLM judges
    for judge_name, llm_file in llm_files_by_judge.items():
        df_llm = pd.read_excel(llm_file)
        llm_vec = normalize_labels(df_llm.iloc[:, COLUMN_INDEX]).map(tf_to_val).to_numpy()
        annotations[judge_name] = llm_vec
    
    # Filter uncertain if needed
    if 'human_uncertain' in df1.columns:
        uncertain_mask = df1['human_uncertain'] == 1
        annotations_filtered = {
            name: vec[~uncertain_mask] 
            for name, vec in annotations.items()
        }
    else:
        annotations_filtered = annotations
    
    # Compute agreement matrix
    rates, counts, judges = compute_agreement_matrix(annotations_filtered)
    
    return {
        'modification': modification,
        'model': model,
        'agreement_rates': rates,
        'vote_counts': counts,
        'judges': judges,
        'n_rows_total': len(df1),
        #'n_rows_filtered': len(annotations_filtered['Human-A1'])
    }

# Main execution
def main():
    # Load human annotations
    human_files = {
        'a1': get_file_map(ANNOTATOR1_DIR, "a1_"),
        'a2': get_file_map(ANNOTATOR2_DIR, "a2_"),
        'a3': get_file_map(ANNOTATOR3_DIR, "a3_")
    }
    
    # Load and organize LLM annotations
    llm_files = [f for f in os.listdir(LLM_DIR) if f.startswith("llm_") and f.endswith(".xlsx")]
    
    # Group LLM files by (modification, generator)
    llm_grouped = defaultdict(dict)
    for fname in llm_files:
        modification, generator, judge = parse_llm_filename(fname)
        key = (modification, generator)
        llm_grouped[key][f"LLM-{judge}"] = os.path.join(LLM_DIR, fname)
    
    # Process each modification-model combination
    results = []
    #output_file = os.path.join(COMBINED_DIR, "agreement_matrices_all.txt")
    output_file = "llm-judge/agreement_matrices_all.txt"
    
    with open(output_file, 'w') as f:
        for (modification, model), llm_files_dict in llm_grouped.items():
            print(f"\nProcessing: {modification} - {model}")
            result = process_modification_model(
                modification, model, human_files, llm_files_dict
            )
            if result:
                results.append(result)
                
                # Create and display agreement table
                table = create_agreement_table(
                    result['agreement_rates'],
                    result['vote_counts'],
                    result['judges']
                )
                print(f"\nAgreement Matrix for {modification} - {model}:")
                print(f"Total rows: {result['n_rows_total']}")
                print(table.to_string())
                
                # Write to file
                f.write("="*60 + "\n")
                f.write(f"Agreement Matrix for {modification} - {model}\n")
                f.write(f"Total rows: {result['n_rows_total']}\n\n")
                f.write(table.to_string())
                f.write("\n\n")
    
    print("\n" + "="*60)
    print("Agreement analysis complete!")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
