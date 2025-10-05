# Say It Another Way: Auditing LLMs with a User-Grounded Automated Paraphrasing Framework
This repository contains code for evaluating social bias in large language models (LLMs) through prompt paraphrasing.
The datasets used are [BBQ](https://github.com/nyu-mll/BBQ) and [MMLU](https://huggingface.co/datasets/cais/mmlu).

## Setup and Environment

To run the code, it is recommended to create a dedicated Python virtual environment:

```bash
# Create virtual environment using venv
python3 -m venv venv

# Activate the environment
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Structure

The repository is organized as follows:
```
├── src/
│   ├── compute_metrics.py
│   ├── configuration.py
│   ├── generate_prompts.py
│   ├── llm_inference.py
│   ├── llms.py
│   ├── paraphrase_detection.py
│   ├── paraphrasing.py
│   └── utils.py
├── data/
│   ├── BBQ/
│   └── MMLU/
├── notebooks/
│   ├── annotation_analysis.ipynb
│   ├── classification_analysis.ipynb
│   ├── downstream_metrics.ipynb
│   └── iaa_scores.ipynb
├── results/
│   ├── BBQ/
│   └── MMLU/
├── requirements.txt
├── .gitignore
└── README.md
```


## Paraphrase

### Paraphrasing
The `src/paraphrasing.py` file outputs a .csv file that resemble the original one, with added columns for paraphrases. You should run it like this, by specifying the dataset, generator model, modification and category of the dataset:

```python3 src/paraphrasing.py --dataset BBQ --model chatgpt --modification prepositions --category Age```

More information on the configurations supported can be found in the `src/configuration.py`. 

### Automatic detection and filtering

Once the paraphrases are generated, you can build excel files for human annotation with the following command.

```python3 src/paraphrase_detection.py --dataset BBQ --model chatgpt --modification prepositions --category Gender_identity --building```

Once the excel is annotated, inter-annotator agreements and ground truth can be computed in "notebooks/iaa_scores.ipynb" and automatic rules can be compared to human ground truth in "notebooks/annotation_analysis.ipynb".

To apply automatic detection and filter paraphrases, run:

```python3 src/paraphrase_detection.py --dataset BBQ --model chatgpt --modification prepositions --category Gender_identity --filtering```

If you are running the paraphrasing process for the random baseline, i.e. the modification "random", the command --building will perform the automatic classification of paraphrases, whereas the command --filtering will flatten the lists of 5 paraphrases per example.

### Formatting

Once the .csv file is filtered or flattened, to create the prompts that will be used for LLM inference, run the following command. This will save a .jsonl file in the ‘data/{dataset}/jsonl’ folder.

```python3 src/generate_prompts.py --dataset $DATASET --modification $MODIF --model $MODEL --category $CAT```

## Inference 
You can run the inference with each LLM with the following command, with a jsonl file generated previously:
```python3 src/llm_inference.py --dataset $DATASET --model $MODEL_NAME --file $file ```

## Evaluation
The downstream metrics can be computed with the following command:
```python3 src/compute_metrics.py --dataset $DATASET```

The results can then be analyzed in the notebook "notebooks/downstream_metrics.ipynb".



