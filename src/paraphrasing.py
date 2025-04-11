"""This script is for the generation of paraphrases."""

import os
import argparse
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import random
from utils import return_list_from_string

def choose_vocabulary(gender_bbq_templates):
    """
    This function choose vocabulary words in the BBQ templates.
    Should be performed only once on each stereotypical category before paraphrasing, so that all original templates have the same lexical diversity.
    """

    original_df=gender_bbq_templates.copy()

    #Iterating through BBQ templates
    for idx, row in tqdm(gender_bbq_templates.iterrows(), total=gender_bbq_templates.shape[0]):
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

from dotenv import load_dotenv # for loading API key

def get_openai_client(model_name, key):
    """
    Returns an OpenAI client and model name based on the specified backend.

    Args:
        model_name (str): The name of the model backend to use ("deepseek" or "chatgpt").
        key (str): API key for the respective service.

    Returns:
        tuple: (OpenAI client instance, model name string)
    """
    if model_name == "deepseek":
        return OpenAI(api_key=key, base_url="https://api.deepseek.com"), "deepseek-chat"
    elif model_name == "chatgpt":
        return OpenAI(api_key=key), "gpt-4o"
    else:
        raise ValueError("Unknown model name: choose 'deepseek' or 'chatgpt'")

def paraphrase(para_modif, instructions_df, gender_bbq_templates, api_key, use_model="deepseek"):
    """
    This function performs paraphrase on the whole Gender identity subset contexts

    Args:
        para_modif (str): the type of paraphrase modification required

    Returns:
        pd.DataFrame
    """

    # Loading the correct prompt template
    prompt_template=instructions_df.loc[instructions_df.modification==para_modif, "prompt"].values[0] 

    # Output DataFrame
    paraphrase_df=gender_bbq_templates.copy()

    # Load model client and model name
    client, model_name = get_openai_client(use_model, api_key)

    #Iterating through BBQ templates
    for idx, row in tqdm(gender_bbq_templates.iterrows(), total=gender_bbq_templates.shape[0]):
        for _, disambiguated in enumerate([False, True]): #for each row, paraphrase ambiguous context alone or ambiguous+disambiguated
            if disambiguated:
                original_context=row["Disambiguating_Context"]
            else: 
                original_context=row["Ambiguous_Context"]

            prompt=prompt_template.format(original_context) #replace the placeholder {} in the prompt template with the context
            try:
                #Call the API for the prompt
                response = client.chat.completions.create(
                    model= model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt}], stream=False)
                
            except Exception as e:
                print(f"Failed to generate for row {idx}, disambiguated {disambiguated}")
                raise e
                
            #Retrieve the raw_response text
            response_text=response.choices[0].message.content
            if disambiguated:
                paraphrase_df.loc[idx, "Disambiguating_Context"]=response_text
            else:
                paraphrase_df.loc[idx, "Ambiguous_Context"]=response_text
    
    return paraphrase_df

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Paraphrase BBQ templates using LLMs.")
    parser.add_argument("--model", choices=["deepseek", "chatgpt"], default="deepseek",
                        help="Choose the LLM backend to use: 'deepseek' or 'chatgpt'. Default is 'deepseek'.")
    args = parser.parse_args()
    
    use_model = args.model
    modification='synonym_substitution'

    #Paths
    DATA_FOLDER='./data/paraphrases/'
    TEMPLATE_FILE = DATA_FOLDER+"Gender_identity_original.csv"
    INSTRUCTION_FILE = DATA_FOLDER+"paraphrase_instructions.tsv"
    OUTPUT_FILE = DATA_FOLDER+f"Gender_identity_{modification}_{use_model}.csv"

    #Loading API key
    load_dotenv()  # load environment variables from .env
    api_key = os.getenv("OPENAI_API_KEY")

    ''' TODO: standardize the api key loading process
    with open(os.path.expanduser(f"~/{use_model}_api.key"), "r") as f:
        api_key=f.read().strip()
    '''

    # Loading the dataframes
    instructions_df=pd.read_csv(INSTRUCTION_FILE, sep='\t')
    gender_bbq_templates=pd.read_csv(TEMPLATE_FILE)

    #Paraphrasing
    paraphrase_df=paraphrase(modification, instructions_df, gender_bbq_templates, api_key, use_model)

    #Saving output
    paraphrase_df.to_csv(OUTPUT_FILE, index=False)
