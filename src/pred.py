import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
from llms import LLMs
from utils import make_prompt_for_chatmodel

with open("./data/debias_prompts.json", "r") as f:
    PROMPTS = json.load(f)

def load_existing_results(filepath):
    """Load already computed results from the existing file."""
    if os.path.isfile(filepath):
        with open(filepath, "r") as f:
            responses = f.read().splitlines()
        print(f"Resuming from {len(responses)} existing results.")
        return responses
    return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--debias_prompt", type=str)
    parser.add_argument("--is_chatmodel", action="store_true")
    parser.add_argument("--output_dir", type=str) # added this to specify output 
    parser.add_argument("--probas", type=str, default="False") 

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_id = args.model.replace("/", "-")

    if args.output_dir is None:
        if args.probas=='False':
            args.output_dir = "./result/"+model_id
        else:
            args.output_dir = "./result_uncertainty/"+model_id

    # load test data
    file = args.file_name
    with open(file, "r") as f:
        jsonl_data = [json.loads(line) for line in f.readlines()]
    file_name = Path(file).stem
    fname = f"result_{model_id}_{file_name}"

    p = None
    if args.debias_prompt:
        p = PROMPTS[args.debias_prompt]
        fname += f"_{args.debias_prompt}"
    fname += ".txt"

    res_path = Path(args.output_dir) #/ f"{file_name}"
    res_path.mkdir(parents=True, exist_ok=True)

    file_path=res_path / f"{fname}"
    print(file_path)
    existing_responses = load_existing_results(file_path)
    start_index = len(existing_responses)

    if start_index >= len(jsonl_data):
        print("File already contains all results, skipping computation.")
    else:
        llm = LLMs(args.model, model_id, device)
        # inference
        for jd in tqdm(jsonl_data[start_index:]):
            prompt = jd.get("prompt", "") # sometimes 'prompt' is missing, default to empty
            enum_choices = jd["enum_choices"]
            if p:
                if args.is_chatmodel:
                    context = make_prompt_for_chatmodel(prompt, "\n".join(p), model_id)
                else:
                    context = p + "\n\n" + prompt
            else:
                context = prompt
            if args.probas=='False':
                pred = llm.pred_MCP(context, enum_choices, ["A", "B", "C"])
            else:
                pred = llm.pred_likelihoods(context, enum_choices)
                pred=str(pred)
            existing_responses.append(pred)

        # save output
        with open(file_path, "w") as f:
            f.write("\n".join(existing_responses))
