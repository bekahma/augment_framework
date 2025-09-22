"""This script is for the inference with the target LLMs, for any dataset. """

import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
import pandas as pd
from llms import LLMs
from utils import make_prompt_for_chatmodel

def load_existing_results(filepath):
    """Load already computed results from the existing file."""
    if os.path.isfile(filepath):
        with open(filepath, "r") as f:
            responses = [json.loads(line) for line in f.readlines()]
        print(f"Resuming from {len(responses)} existing results.")
        return responses
    return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--is_chatmodel", action="store_true")
    parser.add_argument(
        "--dataset",
        choices=["BBQ", "MMLU"],
        type=str,
        default="BBQ",
        help="Specify the dataset used."
    )
    parser.add_argument("--output_dir", type=str) # added this to specify output 

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_id = args.model.replace("/", "-")
    dataset=args.dataset

    if args.output_dir is None:
        args.output_dir = f"./result/{dataset}/{model_id}"

    # load test data
    file = args.file_name
    file_name = Path(file).stem
    fname = f"result_{model_id}_{file_name}.jsonl"

    with open(file, "r") as f:
        data = [json.loads(line) for line in f.readlines()]

    res_path = Path(args.output_dir) #/ f"{file_name}"
    res_path.mkdir(parents=True, exist_ok=True)

    file_path=res_path / f"{fname}"
    print(file_path)
    existing_responses = load_existing_results(file_path)
    start_index = len(existing_responses)

    if start_index >= len(data):
        print("File already contains all results, skipping computation.")
    else:
        llm = LLMs(args.model, model_id, device)
        with open(file_path, "a") as f:
            # inference
            for jd in tqdm(data[start_index:], miniters=100):
                prompt = jd.get("prompt", "") # sometimes 'prompt' is missing, default to empty
                enum_choices = jd["enum_choices"]

                loglikelihoods = llm.pred_likelihoods(prompt, enum_choices)
                # Determine predicted answer (max likelihood)
                predicted_answer = enum_choices[loglikelihoods.index(max(loglikelihoods))]

                # Create output row
                out_row = {
                    "prompt": prompt,
                    "enum_choices": enum_choices,
                    "loglikelihoods": loglikelihoods,
                    "answer": predicted_answer,
                    "true_answer": jd["label"]
                }

                # Write as JSONL
                f.write(json.dumps(out_row) + "\n")