"""This script is for automatic detection of paraphrases."""

from tqdm import tqdm
import difflib
import spacy
from bert_score import score
from sentence_transformers import CrossEncoder
import pandas as pd
from transformers import logging
logging.set_verbosity_error() 
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import language_tool_python

#Load grammar tool
tool = language_tool_python.LanguageTool('en-US')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def compute_bert_score(reference, candidate):
    # Calculate BERTScore for reference and candidate sentences
    P, R, F1 = score([candidate], [reference], lang="en")
    return F1.mean().item()

def compute_sbert_cross_encoder(reference, candidate):
    # Calculate SBERTScore for reference and candidate sentences
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    similarity = model.predict([(reference, candidate)])
    return similarity[0]

def compare_sentences(sentence1, sentence2):
    # Compare the words that were removed or added
    tokens1 = [token.text for token in nlp(sentence1)]
    tokens2 = [token.text for token in nlp(sentence2)]
    
    diff = list(difflib.ndiff(tokens1, tokens2))
    
    changes = {
        "added": [token[2:] for token in diff if token.startswith('+ ')],
        "removed": [token[2:] for token in diff if token.startswith('- ')]
    }
    
    return changes

def grammar_errors(sentence1, sentence2):
    '''Checks for new grammar errors in sentence 2'''
    errors1 = tool.check(sentence1) 
    errors2 = tool.check(sentence2) 
            
    # Convert the list of errors to sets
    errors1 = set([str(error) for error in errors1])
    errors2 = set([str(error) for error in errors2])

    return errors2 - errors1

def compute_perplexity(text):
    '''Compute perplexity with gpt-2
        TODO maybe compute perplexity in other ways?
    '''

    # Load pre-trained model and tokenizer from Hugging Face
    model_name = "gpt2"  
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Encode the text and get input tensors
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_likelihood = outputs.loss * inputs["input_ids"].size(1)
    
    # Compute perplexity
    perplexity = torch.exp(log_likelihood / inputs["input_ids"].size(1))
    return perplexity.item()

def build_excel(gender_bbq_templates, paraphrase_df, output_path, modification):
    annotations_df=pd.DataFrame(columns=['idx', 'Q_id', "disambiguated", 'modification',  'original', 'raw_answer', 'nb_modif', 
                                         'wrong_modif', 'realism', 'meaning', #columns for human annotation
                                         'added_words', 'removed_words', 'pos_added', 'pos_removed', 
                                         'wrong_added', "wrong_removed", "bert_score", "sbert_score", 
                                         'grammar', 'perplexity_par', 'perplexity_original']) 
    
    for idx, row in tqdm(paraphrase_df.iterrows(), total=paraphrase_df.shape[0]):
        for _, disambiguated in enumerate([False, True]): 
            if disambiguated:
                original_context=gender_bbq_templates.loc[idx, "Disambiguating_Context"]
                paraphrased=row["Disambiguating_Context"]
            else: 
                original_context=gender_bbq_templates.loc[idx, "Ambiguous_Context"]
                paraphrased=row["Ambiguous_Context"]

            # Run words comparison
            changes = compare_sentences(original_context, paraphrased)

            # POS tagging in context
            doc_original = nlp(original_context)
            doc_paraphrased = nlp(paraphrased)

            # Get POS tags only for changed words in their sentence context
            added_pos_tags = [token.pos_ for token in doc_paraphrased if token.text in changes["added"]]
            removed_pos_tags = [token.pos_ for token in doc_original if token.text in changes["removed"]]

            if modification=='prepositions':
                pos_tags_to_check = {"ADP", "SCONJ", 'ADV', 'CCONJ', 'PART'} #specific to current modification !!
                allowed_deps = {"prep"} #specific to current modification !! TODO: add advmod?

                wrong_added = [
                    token.text for token in doc_paraphrased
                    if token.text in changes["added"]
                    and token.pos_ not in pos_tags_to_check
                    and token.dep_ not in allowed_deps
                ]
                wrong_removed = [
                    token.text for token in doc_original
                    if token.text in changes["removed"] 
                    and token.pos_ not in pos_tags_to_check
                    and token.dep_ not in allowed_deps
                ]
            else:
                #TODO adapt to other modifications
                wrong_added = []
                wrong_removed=[]

            # Append to the DataFrame
            annotations_df.loc[len(annotations_df)] = {
                "idx": idx,
                "Q_id":row['Q_id'],
                "disambiguated":disambiguated,
                "modification":modification,
                "original": original_context,
                "raw_answer": paraphrased.replace('\n', '\\n'), #reformat breaking lines for better display
                'nb_modif':len(changes["added"])+len(changes["removed"]),
                'added_words':changes["added"], 
                'removed_words': changes["removed"],
                'pos_added': added_pos_tags,
                'pos_removed': removed_pos_tags,
                "wrong_added": wrong_added,
                "wrong_removed": wrong_removed,
                "bert_score": compute_bert_score(original_context, paraphrased), 
                "sbert_score":compute_sbert_cross_encoder(original_context, paraphrased),
                'grammar':grammar_errors(original_context, paraphrased),
                "perplexity_par":compute_perplexity(paraphrased),
                "perplexity_original":compute_perplexity(original_context)
            }

    annotations_df.to_excel(output_path, index=False)

if __name__ == "__main__":
    #Config
    # TODO: change every time !!!
    modification='synonym_substitution'
    use_model='deepseek'

    #Paths
    DATA_FOLDER='./data/paraphrases/'
    TEMPLATE_FILE = DATA_FOLDER+"Gender_identity_original.csv"
    PARAPHRASE_FILE=DATA_FOLDER+f"Gender_identity_{modification}_{use_model}.csv"
    OUTPUT_FILE = DATA_FOLDER+f"Gender_identity_{modification}_{use_model}.xlsx"

    #Loading data
    gender_bbq_templates=pd.read_csv(TEMPLATE_FILE)
    paraphrase_df=pd.read_csv(PARAPHRASE_FILE)

    #Building the excel for annotation
    build_excel(gender_bbq_templates, paraphrase_df, OUTPUT_FILE, modification)

