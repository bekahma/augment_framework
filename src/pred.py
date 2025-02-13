import json
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
from llms import LLMs
from utils import make_prompt_for_chatmodel

with open("./data/debias_prompts.json", "r") as f:
    PROMPTS = json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--debias_prompt", type=str)
    parser.add_argument("--is_chatmodel", action="store_true")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_id = args.model.replace("/", "-")

    llm = LLMs(args.model, model_id, device)

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

    # inference
    responses = []
    for jd in tqdm(jsonl_data):
        prompt = jd.get("prompt", "") # sometimes 'prompt' is missing, default to empty
        enum_choices = jd["enum_choices"]
        if p:
            if args.is_chatmodel:
                context = make_prompt_for_chatmodel(prompt, "\n".join(p), model_id)
            else:
                context = p + "\n\n" + prompt
        else:
            context = prompt
        pred = llm.pred_MCP(context, enum_choices, ["A", "B", "C"])
        responses.append(pred)

    # save output
    res_path = Path("result") / f"{file_name}"
    res_path.mkdir(parents=True, exist_ok=True)
    with open(res_path / f"{fname}", "w") as f:
        f.write("\n".join(responses))
