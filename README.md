# Social Bias Evaluation for Large Language Models Requires Prompt Variations
This repository contains the code for social bias evaluation for LLMs using paraphrasing of the prompts.
The dataset is [BBQ dataset](https://github.com/nyu-mll/BBQ).

# How to Use
You can run experiments with the following command.

## Dataset Preparation
Before inference, prepare the variation of 1. task instruction and prompt, 2. few-shot settings.
`python3 data/convert_format.py`

## Paraphrasing
The `paraphrasing.py` file outputs a .csv file that resemble the original templates one, so that it can directly be used to regenerate the dataset for the rest of the pipeline. You should run it like this, by specifying the model (TODO also parse the modification?)

```python3 src/paraphrasing.py --use_model chatgpt```

Once the .csv file is generated, to recreate the whole dataset in a format that will be used in the `pred.py` script, run (TODO: should also parse arguments instead of modifying in the file)

```python3 src/generate_BBQ_from_templates.py```

This will save a .jsonl file in the ‘data/jsonl’ folder.

Finally, to perform human annotations on the paraphrased outputs, you can run 

```python3 src/paraphrase_detection.py```

This will take in input the csv file from the paraphrasing script as well as the original csv file and build an excel file with automated metrics for annotation.

## Inference
You can run the inference by each LLM.
`export PYTHONPATH="$pwd/src:$PYTHONPATH"; python3 src/pred.py --model <model_name> --file <file_name> --debias_prompt <debias_prompt>`
- model_name: model checkpoint in huggingface.
- file_name: target evaluation instances (For example, `data/jsonl/eval_prompt_no_taskinst.jsonl`)
- debias_prompt: debias_prompt key. See the above description. When evaluating without debias-prompts, drop this arg. 

## Evaluation
You can calculate task performance and social bias of LLMs.
`python3 evaluation/eval_bbq.py --result_dir <result_folder>`



