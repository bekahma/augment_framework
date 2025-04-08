import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import random

def return_list_from_string(inputx):
    """
    From BBQ codebase
    This function takes a string that's formatted as
    WORD1: [word, word]; WORD2: [word, word]
    and separates it into two iterable lists

    Args:
        input: string

    Returns:
        two lists
    """
    x = inputx.split(";")
    for wrd in x:
        if "WORD1" in wrd or "NAME1" in wrd:
            wrd2 = (
                wrd.replace("WORD1:", "")
                .replace("{{WORD1}}:", "")
                .replace("NAME1:", "")
                .replace("{{NAME1}}:", "")
            )
            wrd3 = wrd2.strip()
            wrds = wrd3.replace("[", "").replace("]", "")
            wrds1 = wrds.split(",")
            wrds1 = [w.strip() for w in wrds1]
        if "WORD2" in wrd or "NAME2" in wrd:
            wrd2 = (
                wrd.replace("WORD2:", "")
                .replace("{{WORD2}}:", "")
                .replace("NAME2:", "")
                .replace("{{NAME2}}:", "")
            )
            wrd3 = wrd2.strip()
            wrds = wrd3.replace("[", "").replace("]", "")
            wrds2 = wrds.split(",")
            wrds2 = [w.strip() for w in wrds2]
        else:
            wrds2 = ""
    return wrds1, wrds2


def paraphrase(para_modif, instructions_df, gender_bbq_templates):
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
    paraphrase_df=pd.DataFrame(columns=['modification', 'Q_id', 'disambiguated', 'original', 'raw_answer']) #'explanations'])

    # Initializing DeepSeek client
    client = OpenAI(base_url="https://api.deepseek.com")

    #Iterating through BBQ templates
    for idx, row in tqdm(gender_bbq_templates.iterrows(), total=gender_bbq_templates.shape[0]):
        for _, disambiguated in enumerate([False, True]): #for each row, paraphrase ambiguous context alone or ambiguous+disambiguated
            original_context=row["Ambiguous_Context"]
            if disambiguated:
                original_context+=' '+row["Disambiguating_Context"]

            #some templates have lexical diversity, we just pick one word randomly and remove the placeholders {{WORD1}} and {{WORD2}}
            lex_div = row['Lexical_diversity']
            if pd.notna(lex_div):
                wrdlist1, wrdlist2 = return_list_from_string(lex_div)
                rand_wrd1 = random.choice(wrdlist1)
                original_context=original_context.replace("{{WORD1}}", rand_wrd1)
                if len(wrdlist2) > 1:  # sometimes there's not a WORD2
                    rand_wrd2 = random.choice(wrdlist2)
                    original_context=original_context.replace("{{WORD2}}", rand_wrd2)

            prompt=prompt_template.format(original_context) #replace the placeholder {} in the prompt template with the context
            try:
                #Call the API for the prompt
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt}], stream=False)
                
                #Retrieve the raw_response text
                response_text=response.choices[0].message.content

                # Append to the DataFrame
                paraphrase_df.loc[len(paraphrase_df)] = {
                    "modification":"prepositions",
                    "Q_id":row['Q_id'],
                    "disambiguated":disambiguated,
                    "original": original_context,
                    "raw_answer": response_text.replace('\n', '\\n') #reformat breaking lines for better display
                }

            except Exception as e:
                print(f"Failed to generate for row {idx}, disambiguated {disambiguated}")
                raise e
    
    return paraphrase_df

if __name__ == "__main__":
    #Paths
    DATA_FOLDER='./data/'
    INSTRUCTION_FILE = DATA_FOLDER+"paraphrases/paraphrase_instructions.tsv"
    TEMPLATE_FILE = DATA_FOLDER+"BBQ_templates/Gender_identity.csv"
    OUTPUT_FILE = DATA_FOLDER+"paraphrases/gender_paraphrases.xlsx"

    modification='prepositions'

    # Loading the dataframes
    instructions_df=pd.read_csv(INSTRUCTION_FILE, sep='\t')
    gender_bbq_templates=pd.read_csv(TEMPLATE_FILE)

    #Paraphrasing
    paraphrase_df=paraphrase(modification, instructions_df, gender_bbq_templates)

    #Saving to excel for annotations
    paraphrase_df.to_excel(OUTPUT_FILE, index=False)