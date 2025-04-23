"""This script is for automatic detection of paraphrases."""

import os
import ast
import pickle
import argparse
import random
from tqdm import tqdm
import difflib
import spacy
from bert_score import score
from sentence_transformers import CrossEncoder
from rouge_score import rouge_scorer
import pandas as pd
from transformers import logging
logging.set_verbosity_error() 
import torch
from transformers import pipeline, DebertaV2ForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import language_tool_python

#LOADING MODELS
#Load grammar tool
tool = language_tool_python.LanguageTool('en-US')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

#Load cross encoder model for sbert
ce_model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

# Load pre-trained model and tokenizer from Hugging Face for perplexity
model_name = "EleutherAI/gpt-neo-1.3B"
perplexity_model = AutoModelForCausalLM.from_pretrained(model_name)
perplexity_tokenizer = AutoTokenizer.from_pretrained(model_name)

#AAE classifier
aae_model = DebertaV2ForSequenceClassification.from_pretrained("webis/acl2024-aae-dialect-classification", subfolder="model")
aae_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
classifier = pipeline("text-classification", model=aae_model, tokenizer=aae_tokenizer)

#UTILS FUNCTIONS
def compute_rouge_l(reference, candidate):
    # Calculate ROUGE-L score for reference and candidate sentences
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, candidate)
    return score['rougeL'].fmeasure

def compute_bert_score(reference, candidate):
    # Calculate BERTScore for reference and candidate sentences
    P, R, F1 = score([candidate], [reference], lang="en")
    return F1.mean().item()

def compute_sbert(reference, candidate):
    # Calculate SBERTScore for reference and candidate sentences
    similarity = ce_model.predict([(reference, candidate)])
    return similarity[0]

def compare_sentences(sentence1, sentence2):
    # Compare the words that were removed or added
    tokens1 = [token.text for token in nlp(sentence1)]
    tokens2 = [token.text for token in nlp(sentence2)]
    diff = list(difflib.ndiff(tokens1, tokens2))
    changes = {
        "added": [],
        "removed": []
    }

    index1 = index2 = 0
    for token in diff:
        if token.startswith('  '):
            index1 += 1
            index2 += 1
        elif token.startswith('- '):
            changes["removed"].append((token[2:], index1))
            index1 += 1
        elif token.startswith('+ '):
            changes["added"].append((token[2:], index2))
            index2 += 1

    return changes

def lemmatize_list(words):
    return [nlp(w)[0].lemma_ for w in words]

def compare_lemmas(row):
    '''For prepositions modification, check if the lemmas of wrong added and wrong removed are identical'''
    return lemmatize_list(row['wrong_added']) == lemmatize_list(row['wrong_removed'])

def detect_AAE(text):
    '''Classify the text as SAE or AAE'''
    text=text.replace('{{NAME1}}', 'woman')
    text=text.replace('{{NAME2}}', 'man')
    return classifier(text)

def grammar_errors(sentence1, sentence2):
    '''Checks for new grammar errors in sentence 2'''
    errors1 = tool.check(sentence1) 
    errors2 = tool.check(sentence2) 
            
    # Convert the list of errors to sets
    errors1 = set([str(error) for error in errors1])
    errors2 = set([str(error) for error in errors2])

    return errors2 - errors1

def compute_perplexity(text):
    '''Compute perplexity'''
    # Encode the text and get input tensors
    inputs = perplexity_tokenizer(text, return_tensors="pt")
    
    # Get the model's output
    with torch.no_grad():
        outputs = perplexity_model(**inputs, labels=inputs["input_ids"])
        log_likelihood = outputs.loss * inputs["input_ids"].size(1)
    
    # Compute perplexity
    perplexity = torch.exp(log_likelihood / inputs["input_ids"].size(1))
    return perplexity.item()

def detect_pos(doc_paraphrased, doc_original, added_tokens, removed_tokens, pos_tags_to_check, allowed_deps):
    """
    Identifies POS tags of added and removed tokens in a paraphrased sentence, 
    and flags any that do not match the expected POS or syntactic dependencies.
    """
    # Get POS tags only for changed words in their sentence context
    added_pos_tags = [
        token.pos_ for i, token in enumerate(doc_paraphrased)
        if (token.text, i) in added_tokens
    ]
    removed_pos_tags = [
        token.pos_ for i, token in enumerate(doc_original)
        if (token.text, i) in removed_tokens
    ]
    #Check if correct POS tags were added/removed
    wrong_added = [
        token.lemma_ for i, token in enumerate(doc_paraphrased)
        if (token.text, i) in added_tokens
        and token.pos_ not in pos_tags_to_check
        and token.dep_ not in allowed_deps
        and not token.is_space and not token.is_punct
    ]
    wrong_removed = [
        token.lemma_ for i, token in enumerate(doc_original)
        if (token.text, i) in removed_tokens
        and token.pos_ not in pos_tags_to_check
        and token.dep_ not in allowed_deps
        and not token.is_space and not token.is_punct
    ]
    return added_pos_tags, removed_pos_tags, wrong_added, wrong_removed

def automatic_detection(original_context, paraphrase, modification, other_metrics=True):
    """
    Automatically analyzes differences between an original sentence and its paraphrase,
    computing both general and modification-specific metrics.

    Args:
        original_context (str): The original input sentence.
        paraphrase (str): The paraphrased version of the sentence.
        modification (str): Type of paraphrasing modification applied (e.g., 'prepositions', 'AAE', etc.).
        other_metrics (bool): Whether to compute additional semantic and fluency metrics (e.g., BERTScore, perplexity).

    Returns:
        dict: A dictionary of metrics
    """
    
    # POS tagging in context
    doc_original = nlp(original_context)
    doc_paraphrased = nlp(paraphrase)
    
    # Run words comparison
    changes = compare_sentences(original_context, paraphrase)

    # Create lookup sets for fast access
    added_tokens= set(changes["added"])
    removed_tokens = set(changes["removed"])

    # Get changed words (ignoring spaces and punctuation)
    added_words =[ token.text for i, token in enumerate(doc_paraphrased)
            if (token.text, i) in added_tokens
            and not token.is_space and not token.is_punct]
    
    removed_words = [ token.text for i, token in enumerate(doc_original)
            if (token.text, i) in removed_tokens
                        and not token.is_space and not token.is_punct]
    
    metrics={'nb_modif':len(added_words)+len(removed_words),
            'added_words': added_words, 
            'removed_words': removed_words}
                
    
    if modification=='prepositions': #Specific metrics for the preposition modification 
        
        pos_tags_to_check = {'DET', "ADP", "SCONJ", 'ADV', 'CCONJ', 'PART'} 
        allowed_deps = {"prep"}

        added_pos_tags, removed_pos_tags, wrong_added, wrong_removed=detect_pos(doc_paraphrased, doc_original, added_tokens, removed_tokens, pos_tags_to_check, allowed_deps)

        metrics.update({
            'pos_added': added_pos_tags, 
            'pos_removed':removed_pos_tags, 
            'wrong_added': wrong_added, 
            "wrong_removed": wrong_removed
        })
    
    elif modification=='AAE': #Specific metrics for the AAE modification 
        pred_original=detect_AAE(original_context)[0]
        pred_par=detect_AAE(paraphrase)[0]
        metrics.update({
            "label_ori": pred_original["label"],
            "proba_ori": pred_original["score"],
            "label_par": pred_par["label"],
            "proba_par": pred_par["score"],
        })

    elif modification=='verb_adjective':
        #TODO finish?
        pos_tags_to_check = {"VERB",  'ADJ', 'DET'} 
        allowed_deps={}
        added_pos_tags, removed_pos_tags, wrong_added, wrong_removed=detect_pos(doc_paraphrased, doc_original, added_tokens, removed_tokens, pos_tags_to_check, allowed_deps)

        metrics.update({
            'pos_added': added_pos_tags, 
            'pos_removed':removed_pos_tags, 
            'wrong_added': wrong_added, 
            "wrong_removed": wrong_removed
        })
    
    else:
        #TODO adapt to other modifications
        wrong_added = []
        wrong_removed=[]
    
    if other_metrics:
        metrics.update({"bert_score": compute_bert_score(original_context, paraphrase), 
                "sbert_score": compute_sbert(original_context, paraphrase),
                "rouge_l": compute_rouge_l(original_context, paraphrase),
                'grammar': grammar_errors(original_context, paraphrase),
                "perplexity_par": compute_perplexity(paraphrase),
                "perplexity_original": compute_perplexity(original_context)})
    
    return metrics


def build_excel(paraphrase_df, output_path, modification):
    """
    Creates an Excel file for human annotation from a DataFrame of paraphrases.

    Args:
        paraphrase_df (pd.DataFrame): The input DataFrame containing paraphrases and original contexts.
        output_path (str): The file path where the Excel file will be saved.
        modification (str): The type of paraphrase modification being filtered (e.g., "prepositions", "AAE").

    Returns:
        None. Writes an Excel file to `output_path` for annotation purposes.
    """
    #Columns per type of modification
    columns_per_modif={'AAE': ["label_ori", "label_par", "proba_ori", "proba_par"],
            'prepositions': ['pos_added', 'pos_removed', 'wrong_added', "wrong_removed"]}
    
    annotations_df=pd.DataFrame(columns=['idx', 'Q_id', "disambiguated", 'modification',  'original', 'raw_answer', 'nb_modif', 
                                         'wrong_modif', 'realism', 'meaning', #columns for human annotation
                                         'added_words', 'removed_words', 'grammar', 
                                         "bert_score", "sbert_score", "rouge_l", 'perplexity_par', 'perplexity_original']
                                         +columns_per_modif[modification] #specific columns to each type of modification
                                         ) 
    
    for idx, row in tqdm(paraphrase_df.iterrows(), total=paraphrase_df.shape[0]): #Iterating for each question ID
        for _, disambiguated in enumerate([False, True]): #Ambiguous or disambiguated contexts
            if disambiguated:
                original_context=row["Disambiguating_Context"] #retrieving original context
                paraphrases=row["Disambiguating_Paraphrases"] #retrieving the list of N paraphrases generated
            else: 
                original_context=row["Ambiguous_Context"] #retrieving original context
                paraphrases=row["Ambiguous_Paraphrases"] #retrieving the list of N paraphrases generated
            
            assert isinstance(paraphrases, list)

            for paraphrase in paraphrases: #iterating through the list of paraphrases

                # Prepare new row
                new_row = {
                    "idx": idx,
                    "Q_id":row['Q_id'],
                    "disambiguated":disambiguated,
                    "modification":modification,
                    "original": original_context,
                    "raw_answer": paraphrase.replace('\n', '\\n'), #reformat breaking lines for better display
                }

                #Add specific metrics
                new_row.update(automatic_detection(original_context, paraphrase, modification))
                
                #Append new row to the dataframe
                annotations_df.loc[len(annotations_df)]=new_row

    #Exporting to excel
    annotations_df.to_excel(output_path, index=False)

def filter_out(paraphrase_df, output_path, modification):
    """
    Filters out rows from a paraphrase DataFrame based on modification-specific heuristics.

    Args:
        paraphrase_df (pd.DataFrame): The input DataFrame containing paraphrases and original contexts.
        output_path (str): Path where a filtered version of the DataFrame is saved (csv format).
        modification (str): The type of paraphrase modification being filtered (e.g., "prepositions", "AAE").

    Returns:
        List[int]: Indices of rows to keep after filtering.
    """
    idx_to_keep=[]
    nb_untouched, nb_wrong=0, 0 #counters for untouched sentences (no modifications) and wrong paraphrases
    for idx, row in tqdm(paraphrase_df.iterrows(), total=paraphrase_df.shape[0]): 
        delete_ctx={}
        for _, disambiguated in enumerate([False, True]): 
            if disambiguated:
                key="Disambiguating_Context"
                original_context=row["Disambiguating_Context"]
                paraphrases=row["Disambiguating_Paraphrases"]
            else: 
                key="Ambiguous_Context"
                original_context=row["Ambiguous_Context"]
                paraphrases=row["Ambiguous_Paraphrases"]
            
            assert isinstance(paraphrases, list)

            for paraphrase in paraphrases:

                metrics_dict=automatic_detection(original_context, paraphrase, modification, other_metrics=False)
                nb_modifs=metrics_dict['nb_modif']

                if nb_modifs==0: 
                    #remove paraphrase if no modification was applied
                    paraphrases.remove(paraphrase)
                    nb_untouched+=1 
                
                else:
                    if modification=='prepositions': 
                        #Automatic detection for prepositions:
                        # Remove paraphrase if incorrect POS tags added/removed
                        # except if same lemmas added/removed 
                        # (for instance "due to ... being" vs "because ... is")
                        wrong_added=metrics_dict['wrong_added']
                        wrong_removed=metrics_dict['wrong_removed']
                        if len(wrong_added)!=0 or len(wrong_removed)!=0: 
                            if wrong_added!=wrong_removed: 
                                paraphrases.remove(paraphrase)
                                nb_wrong+=1
                
                    elif modification=='AAE': 
                        #Automatic detection for AAE
                        pred_label=metrics_dict["label_par"]
                        proba_paraphrase=metrics_dict["proba_par"]
                        proba_original=metrics_dict["proba_ori"]
                        if pred_label=="LABEL_0": #predicted as SAE
                            if proba_paraphrase>0.9 or proba_paraphrase>proba_original:
                                paraphrases.remove(paraphrase)
                                nb_wrong+=1
                    else:
                        #TODO adapt to other modifications
                        raise ValueError(f"No automatic rules were set up for the {modification} modification")
            
            if len(paraphrases)>0: #at least one paraphrase is correct
                row[key]=random.choice(paraphrases)
            else: #all paraphrases were incorrect
                delete_ctx[key]=True

        if not any(delete_ctx.values()): 
            #because of the BBQ structure, if we need to delete the ambiguous context we also need to delete its disambiguated counterpart and vice versa
            idx_to_keep.append(idx)

    print("Number of untouched paraphrases:", nb_untouched) 
    print("Number of wrong paraphrases:", nb_wrong) 
    print("Rows kept out of 60:", len(idx_to_keep))
    paraphrase_df=paraphrase_df.loc[idx_to_keep]
    paraphrase_df.to_csv(output_path, index=False)
    print("Filtered dataframe saved to", output_path)
    return idx_to_keep


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script configuration")

    parser.add_argument('--modification', type=str, default='prepositions',
                        help="Type of modification to apply (e.g., 'prepositions')")
    parser.add_argument('--model', type=str, default='deepseek',
                        help="Model to use (e.g., 'deepseek')")
    parser.add_argument('--building', action='store_true',
                        help="Building the excel file for annotations (default: False)")
    parser.add_argument('--filtering', action='store_true',
                        help="Filtering the dataframe (default: False)")

    args = parser.parse_args()

    modification = args.modification
    model = args.model
    building = args.building
    filtering = args.filtering

    print(f"Results for {modification} with {model}")

    #Paths
    DATA_FOLDER='./data/paraphrases/'
    TEMPLATE_FILE = DATA_FOLDER+"Gender_identity_original.csv"
    PARAPHRASE_FILE=DATA_FOLDER+f"Gender_identity_{modification}_{model}.csv"
    OUTPUT_EXCEL_FILE = DATA_FOLDER+f"Gender_identity_{modification}_{model}.xlsx"
    OUTPUT_FILTERED_FILE = DATA_FOLDER+f"Gender_identity_{modification}_{model}_filtered_new.csv"
    IDS_KEEP= DATA_FOLDER+f"Gender_identity_{modification}_{model}_ids_new.pkl"

    #Loading data
    gender_bbq_templates=pd.read_csv(TEMPLATE_FILE)
    paraphrase_df=pd.read_csv(PARAPHRASE_FILE)
    paraphrase_df["Disambiguating_Paraphrases"]=paraphrase_df["Disambiguating_Paraphrases"].apply(ast.literal_eval)
    paraphrase_df["Ambiguous_Paraphrases"]=paraphrase_df["Ambiguous_Paraphrases"].apply(ast.literal_eval)

    if building:
        #Building the excel for annotation
        build_excel(paraphrase_df, OUTPUT_EXCEL_FILE, modification)
    
    if filtering:
        ids_to_keep=filter_out(paraphrase_df, OUTPUT_FILTERED_FILE, modification)
        with open(IDS_KEEP, 'wb') as f: #save the ids to keep
            pickle.dump(ids_to_keep, f)


