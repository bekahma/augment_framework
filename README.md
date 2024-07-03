# Social Bias Evaluation for Large Language Models Requires Prompt Variations
This repository contains the code for social bias evaluation for LLMs using prompt variations.
The dataset is [BBQ dataset](https://github.com/nyu-mll/BBQ).
Prompt variations are comprised of three perspectives.
1. task instruction and prompt
    - see `data/template.tsv`
2. few-shot examples
    - see `data/BBQ_few_shot.jsonl`
3. debias-prompts
    - see `data/debias_prompts.json`

# How to Use
You can run experiments with the following command.

## Dataset Preparation
Before inference, prepare the variation of 1. task instruction and prompt, 2. few-shot settings.
`python3 data/convert_format.py`


## Inference
You can run the inference by each LLM.
`export PYTHONPATH="$pwd/src:$PYTHONPATH"; python3 src/pred.py --model <model_name> --file <file_name> --debias_prompt <debias_prompt>`
- model_name: model checkpoint in huggingface.
- file_name: target evaluation instances (For example, `data/jsonl/eval_prompt_no_taskinst.jsonl`)
- debias_prompt: debias_prompt key. See the above description. When evaluating without debias-prompts, drop this arg. 

## Evaluation
You can calculate task performance and social bias of LLMs.
`python3 evaluation/eval_bbq.py --result_dir <result_folder>`

