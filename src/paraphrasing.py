"""This script is for the generation of paraphrases."""

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
from dotenv import load_dotenv # for loading API key

def choose_vocabulary(bbq_templates):
    """
    This function choose vocabulary words in the BBQ templates.
    Should be performed only once on each stereotypical category before paraphrasing, so that all original templates have the same lexical diversity.
    """

    original_df=bbq_templates.copy()

    #Iterating through BBQ templates
    for idx, row in tqdm(bbq_templates.iterrows(), total=bbq_templates.shape[0]):
        #some templates have lexical diversity, we just pick one word randomly as in the BBQ construction
        lex_div = row['Lexical_diversity']
        if pd.notna(lex_div):
            wrdlist1, wrdlist2 = return_list_from_string(lex_div)
            rand_wrd1 = random.choice(wrdlist1)
            rand_wrd2 = random.choice(wrdlist2) if len(wrdlist2) > 1 else None
        else:
            rand_wrd1 = rand_wrd2 = None

        for _, disambiguated in enumerate([False, True]): #for each row, paraphrase ambiguous context alone or ambiguous+disambiguated
            original_context=row["Ambiguous_Context"]
            if disambiguated:
                original_context+=' '+row["Disambiguating_Context"]

            #some templates have lexical diversity, we just pick one word randomly and remove the placeholders {{WORD1}} and {{WORD2}}
            if rand_wrd1 is not None:
                original_context = original_context.replace("{{WORD1}}", rand_wrd1)
            if rand_wrd2 is not None:
                original_context = original_context.replace("{{WORD2}}", rand_wrd2)

            if disambiguated:
                original_df.loc[idx, "Disambiguating_Context"]=original_context
            else:
                original_df.loc[idx, "Ambiguous_Context"]=original_context
    
    return original_df

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

def paraphrase(para_modif, instructions_df, bbq_templates, use_model="deepseek", temperature=0):
    """
    This function performs paraphrase on the whole Gender identity subset contexts

    Args:
        para_modif (str): the type of paraphrase modification required

    Returns:
        pd.DataFrame
    """

    # Loading the correct prompt template
    prompt_template=instructions_df.loc[instructions_df.modification==para_modif, "prompt"].values[0] 
    print(prompt_template) #to check if the correct template is being used

    # Output DataFrame
    paraphrase_df=bbq_templates.copy()

    # Initialize empty columns for storing the paraphrases
    paraphrase_df["Ambiguous_Paraphrases"] = None
    paraphrase_df["Disambiguating_Paraphrases"] = None

    if use_model == "mistral":
        tokenizer, model = load_mistral_model()
    else:
        # Load model client and model name
        client, model_name = get_openai_client(use_model)

    #Iterating through BBQ templates
    for idx, row in tqdm(bbq_templates.iterrows(), total=bbq_templates.shape[0]):
        for _, disambiguated in enumerate([False, True]): #for each row, paraphrase ambiguous context alone or ambiguous+disambiguated
            original_context = row["Disambiguating_Context"] if disambiguated else row["Ambiguous_Context"]

            prompt=prompt_template.format(original_context) #replace the placeholder {} in the prompt template with the context
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
                print(f"Failed to generate for row {idx}, disambiguated {disambiguated}")
                raise e
            
            paraphrases = extract_paraphrase_line(response_text)
            
            if disambiguated:
                paraphrase_df.at[idx, "Disambiguating_Paraphrases"] = paraphrases
            else:
                paraphrase_df.at[idx, "Ambiguous_Paraphrases"] = paraphrases

    return paraphrase_df

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Paraphrase BBQ templates using LLMs.")
    parser.add_argument("--model", choices=["deepseek", "chatgpt", "mistral"], default="deepseek",
                        help="Choose the LLM backend to use: 'deepseek' or 'chatgpt'. Default is 'deepseek'.")
    parser.add_argument('--modification', type=str, default='prepositions',
                        help="Type of modification to apply (e.g., 'prepositions')")
    
    parser.add_argument(
        "--category",
        type=str,
        default="Gender_identity",
        help="Specify a single category to paraphrase (e.g., 'Race_ethnicity')."
    )
    
    args = parser.parse_args()
    model = args.model
    modification = args.modification
    category=args.category

    print(f"Paraphrasing for modification {modification} with model {model} for subset {category}")

    #Paths
    DATA_FOLDER='./data/paraphrases/'
    TEMPLATE_FILE = DATA_FOLDER+f"{category}_original.csv"
    INSTRUCTION_FILE = DATA_FOLDER+"paraphrase_instructions.tsv"

    #Output path
    OUTPUT_FOLDER = f'./data/paraphrases/{modification}/'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    OUTPUT_FILE = OUTPUT_FOLDER+f"{category}_{modification}_{model}.csv"

    # Loading the dataframes
    instructions_df=pd.read_csv(INSTRUCTION_FILE, sep='\t')
    bbq_templates=pd.read_csv(TEMPLATE_FILE)

    #Paraphrasing
    paraphrase_df=paraphrase(modification, instructions_df, bbq_templates, model)

    #Saving output
    paraphrase_df.to_csv(OUTPUT_FILE, index=False)
