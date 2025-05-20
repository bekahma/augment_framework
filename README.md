# Say It Another Way: Auditing LLMs with a User-Grounded Automated Paraphrasing Framework
This repository contains code for evaluating social bias in large language models (LLMs) through prompt paraphrasing.
The dataset is [BBQ dataset](https://github.com/nyu-mll/BBQ).

## Paraphrase

### Paraphrasing
The `paraphrasing.py` file outputs a .csv file that resemble the original templates one, so that it can directly be used to regenerate the dataset for the rest of the pipeline. You should run it like this, by specifying the model 

```python3 src/paraphrasing.py --model chatgpt --modification prepositions```

### Automatic detection and filtering

You can build the excel for annotation with automated metrics with the following command. Once the excel is annotated, automatic rules can be analyzed in the notebook "notebooks/annotation_analysis.ipynb".

```python3 src/paraphrase_detection.py --model chatgpt --modification prepositions --building```

To apply automatic detection and filter paraphrases, run:

```python3 src/paraphrase_detection.py --model chatgpt --modification prepositions --filtering```

### Formatting

Once the .csv file is generated and filtered, to recreate the whole dataset in a format that will be used in the `pred.py` script, run the following command. This will save a .jsonl file in the ‘data/jsonl’ folder.

```python3 src/generate_BBQ.py --model chatgpt --modification prepositions```

## Inference 
You can run the inference with each LLM.
```export PYTHONPATH="$pwd/src:$PYTHONPATH"; python3 src/pred.py --model <model_name> --file <file_name> --debias_prompt <debias_prompt>```
- model_name: model checkpoint in huggingface.
- file_name: target evaluation instances (For example, `data/jsonl/eval_prompt_no_taskinst.jsonl`)
- debias_prompt: debias_prompt key. See the above description. When evaluating without debias-prompts, drop this arg. 

## Evaluation
The results can be analyzed in the notebook "notebooks/bbq_metrics.ipynb".



