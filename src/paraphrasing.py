"""This script is for the generation of paraphrases, for any dataset."""

import os
import re
import argparse
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import random
from utils import return_list_from_string
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset

def get_openai_client(model_name):
    """
    Returns an OpenAI client and model name based on the specified backend.
    Args:
        model_name (str): The name of the model backend to use ("deepseek" or "chatgpt").

    Returns:
        tuple: (OpenAI client instance, model name string)
    
    Note:
        The API key must be set in the environment variable `OPENAI_API_KEY` before
        calling this function. The client will automatically use this key.
    """
    if model_name == "deepseek":
        return OpenAI(base_url="https://api.deepseek.com"), "deepseek-chat"
    elif model_name == "chatgpt":
        return OpenAI(), "gpt-4o"
    else:
        raise ValueError("Unknown model name: choose 'deepseek' or 'chatgpt'")
    
def load_mistral_model(model_name="mistralai/Mistral-7B-Instruct-v0.3"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    return tokenizer, model

def generate_with_mistral(prompt, tokenizer, model, temperature=1e-5,  max_new_tokens=256):
    prompt=prompt.replace("{{NAME1}}", "{{NOUN1}}")
    prompt=prompt.replace("{{NAME2}}", "{{NOUN2}}")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text=generated_text[len(prompt):].strip() # remove prompt
    generated_text=generated_text.replace("{{NOUN1}}", "{{NAME1}}")
    generated_text=generated_text.replace("{{NOUN2}}", "{{NAME2}}")
    return generated_text
    
def extract_paraphrase_line(text):
    """Extracts only the line starting with 'PARAPHRASE:' from model output."""
    paraphrases = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match 'PARAPHRASE:' or 'PARAPHRASE 1:', etc.
        match = re.search(r"PARAPHRASE(?: \d+)?:\s*(.*)", line, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if len(candidate) >= 5:
                paraphrases.append(candidate)
            else:
                # Try the next line if it's non-empty and not too short
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if len(next_line) >= 5:
                        paraphrases.append(next_line)
                    i += 1  # skip next line since it's consumed

        else:
            # Match numbered list format like '1. paraphrase here'
            list_match = re.match(r"\d+\.\s+(.+)", line)
            if list_match:
                candidate = list_match.group(1).strip()
                if len(candidate) >= 5:
                    paraphrases.append(candidate)
        
        i += 1

    if paraphrases==[]:
        print(text)
    return paraphrases

def paraphrasing(prompt_template, data_list, use_model="deepseek", temperature=0):
    """
    Generate paraphrases for a list of input sentences using a specified model.

    Args:
        prompt_template (str): A template string for prompts.
        data_list (list[str]): List of sentences to paraphrase.
        use_model (str, optional): Which model to use. Options: "mistral", "deepseek" or "chatgpt".
        temperature (float, optional): Sampling temperature for model generation. Defaults to 0 (deterministic output).

    Returns:
        list[str]: A list of paraphrased sentences corresponding to `data_list`.
    """
    #Initialize output list
    paraphrased_data=[]

    # Load paraphrasing model
    if use_model == "mistral":
        tokenizer, model = load_mistral_model()
    else:
        client, model_name = get_openai_client(use_model)

    #Iterating through the list
    for i, sent in tqdm(enumerate(data_list), total=len(data_list)):
        prompt=prompt_template.format(sent) #replace the placeholder {} in the prompt template with the original sentence
        try:
            if use_model == "mistral":
                response_text = generate_with_mistral(prompt, tokenizer, model)
            else:
                #Call the API for the prompt
                response = client.chat.completions.create(
                    model= model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt}], 
                        temperature=temperature,# top_p=1, #we can play with these parameters for more/less diversity
                        stream=False)
                response_text = response.choices[0].message.content
        
        except Exception as e:
            print(f"Failed to generate for sentence n°{i}")
            raise e
        
        paraphrases = extract_paraphrase_line(response_text)
        paraphrased_data.append(paraphrases)
    
    return paraphrased_data

def paraphrasing_df(data, dataset, para_modif, instructions_df, use_model="deepseek"):
    """
    Apply paraphrasing to a dataset DataFrame based on a paraphrase modification type.

    Args:
        data (pd.DataFrame): Input dataset containing text columns to paraphrase.
        dataset (str): Dataset name ('BBQ', 'MMLU', etc).
        para_modif (str): The type of paraphrase modification. 
        instructions_df (pd.DataFrame): DataFrame containing prompt templates for each modification type.
        use_model (str, optional): Model name to use. Defaults to "deepseek".

    Returns:
        pd.DataFrame: A copy of the input DataFrame with new paraphrased columns added.
    """

    #Loading the correct prompt template
    prompt_template=instructions_df.loc[instructions_df.modification==para_modif, "prompt"].values[0] 
    print(prompt_template) #to check if the correct template is being used

    # Output DataFrame
    paraphrase_df=data.copy()

    if dataset=='BBQ':
        amb_ctxt=data.Ambiguous_Context.to_list()
        disamb_ctxt=data.Disambiguating_Context.to_list()
        amb_ctxt_paraphrased=paraphrasing(prompt_template, amb_ctxt, use_model)
        disamb_ctxt_paraphrased=paraphrasing(prompt_template, disamb_ctxt, use_model)
        paraphrase_df["Disambiguating_Paraphrases"] = disamb_ctxt_paraphrased
        paraphrase_df["Ambiguous_Paraphrases"] = amb_ctxt_paraphrased
        return paraphrase_df
    
    elif dataset=="HatEval":
        txt_list=data.text.to_list()
        paraphrased_txt=paraphrasing(prompt_template, txt_list, use_model)
        paraphrase_df["paraphrases"] = paraphrased_txt
        return paraphrase_df
    
    elif dataset=="MMLU":
        question_list=data.question.to_list() #paraphrasing only the question, not the choices
        paraphrased_txt=paraphrasing(prompt_template, question_list, use_model)
        paraphrase_df["paraphrases"] = paraphrased_txt
        return paraphrase_df

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Paraphrase a specified dataset using LLMs.")

    parser.add_argument("--model", choices=["deepseek", "chatgpt", "mistral"], default="deepseek",
                        help="Choose the paraphrasing LLM: 'deepseek' or 'chatgpt'. Default is 'deepseek'.")
    parser.add_argument('--modification', type=str, default='prepositions',
                        help="Type of modification to apply (e.g., 'prepositions')")
    
    parser.add_argument(
        "--dataset",
        choices=["BBQ", "HatEval", "MMLU"],
        type=str,
        default="BBQ",
        help="Specify the dataset to paraphrase."
    )
    
    parser.add_argument(
        "--category",
        type=str,
        default="None",
        help="Specify a single category to paraphrase (e.g., 'Race_ethnicity' for BBQ or 'philosophy' for MMLU)."
    )
    
    args = parser.parse_args()
    model = args.model
    modification = args.modification
    dataset = args.dataset
    category=args.category

    print(f"Paraphrasing the dataset {dataset} with modification {modification} with model {model} for subset {category}")

    #Paths
    INSTRUCTION_FILE = "./data/paraphrase_instructions.tsv"
    DATA_FOLDER=f'./data/{dataset}/paraphrases/'
    
    #Output path
    OUTPUT_FOLDER = f'./data/{dataset}/paraphrases/{modification}/'
    OUTPUT_FILE = OUTPUT_FOLDER+f"{category}_{modification}_{model}.csv"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Loading the dataframes
    instructions_df=pd.read_csv(INSTRUCTION_FILE, sep='\t')

    if dataset=="BBQ":
        TEMPLATE_FILE = DATA_FOLDER+f"{category}_original.csv"
        data=pd.read_csv(TEMPLATE_FILE)

    elif dataset=="HatEval":
        data_raw = load_dataset("valeriobasile/HatEval")
        data=data_raw["test"].to_pandas()
        data=data.iloc[:30, :] #experimenting with only a small subset
    
    elif dataset=="MMLU":
        TEMPLATE_FILE = DATA_FOLDER+f"{category}_original.csv"
        if not os.path.isfile(TEMPLATE_FILE): #if the dataset was not loaded previously
            data_raw = load_dataset("cais/mmlu", category, split='test') 
            data=data_raw.to_pandas()
            data.to_csv(TEMPLATE_FILE, index=False) #saving the original dataset
        else:
            data=pd.read_csv(TEMPLATE_FILE)

    #Paraphrasing
    paraphrase_df=paraphrasing_df(data, dataset, modification, instructions_df, model)

    #Saving output
    paraphrase_df.to_csv(OUTPUT_FILE, index=False)
