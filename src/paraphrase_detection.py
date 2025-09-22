"""This script is for automatic detection of paraphrases."""

import os
import ast
import pickle
import argparse
import random
import numpy as np
from tqdm import tqdm
import difflib
import spacy
from nltk.stem import PorterStemmer

from bert_score import score
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from rouge_score import rouge_scorer
import pandas as pd
from transformers import logging
logging.set_verbosity_error() 
import torch
from transformers import pipeline, DebertaV2ForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import language_tool_python
from PassivePySrc import PassivePy

#LOADING MODELS
#Load grammar tool
tool = language_tool_python.LanguageTool('en-US')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

#Load cross encoder model for sbert
ce_model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

# Load pre-trained model and tokenizer from Hugging Face for perplexity
model_name = "EleutherAI/gpt-neo-2.7B"
#model_name = "mistralai/Mistral-7B-v0.1"
perplexity_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
perplexity_tokenizer = AutoTokenizer.from_pretrained(model_name)

#AAE classifier
aae_model = DebertaV2ForSequenceClassification.from_pretrained("webis/acl2024-aae-dialect-classification", subfolder="model")
aae_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
aae_classifier = pipeline("text-classification", model=aae_model, tokenizer=aae_tokenizer)

#Formal classifier
formal_classifier = pipeline("text-classification", model="LenDigLearn/formality-classifier-mdeberta-v3-base")

#Synonym cosine similarity model
syn_model = SentenceTransformer('all-MiniLM-L6-v2')

#Passive voice detector
passivepy = PassivePy.PassivePyAnalyzer(spacy_model = "en_core_web_lg")

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

def detect_AAE(text):
    '''Classify the text as SAE or AAE'''
    text=text.replace('{{NAME1}}', 'woman')
    text=text.replace('{{NAME2}}', 'man')
    return aae_classifier(text)

def detect_formal(text):
    '''Classify the text as formal, neutral, or informal'''
    text=text.replace('{{NAME1}}', 'woman')
    text=text.replace('{{NAME2}}', 'man')
    return formal_classifier(text)

def grammar_errors(sentence1, sentence2, excluded_categories={"TYPOS", "PUNCTUATION", "REDUNDANCY"}):
    '''Checks for new grammar errors in sentence 2'''
    errors1 = [e for e in tool.check(sentence1) if e.category != None and e.category not in excluded_categories]
    errors2 = [e for e in tool.check(sentence2) if e.category != None and e.category not in excluded_categories]

    def extract_error_signature(error):
        return (error.ruleId, error.message)

    error_set1 = set(extract_error_signature(e) for e in errors1)
    error_set2 = set(extract_error_signature(e) for e in errors2)

    new_errors = error_set2 - error_set1
    return [e for e in errors2 if extract_error_signature(e) in new_errors]

def compute_perplexity(text):
    '''Compute perplexity'''
    #text=text.replace('{{NAME1}}', 'woman')
    #text=text.replace('{{NAME2}}', 'man')

    # Encode the text and get input tensors
    inputs = perplexity_tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    device = next(perplexity_model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Get the model's output
    with torch.no_grad():
        outputs = perplexity_model(input_ids=input_ids, labels=input_ids)
        logits=outputs.logits
        loss = outputs.loss

    '''# Calculate token-level perplexity
    # We calculate perplexity for each token based on the corresponding logits and the input_ids
    token_perplexities = []
    for i in range(input_ids.size(1)):  # iterate over each token in the sequence
        # Get the logits for the current token
        logit = logits[0, i]
        target_id = input_ids[0, i]
        
        # Calculate the log probability for the token
        log_prob = torch.log_softmax(logit, dim=-1)
        
        # Perplexity for this token is exp(-log_prob for the correct token)
        token_perplexity = torch.exp(-log_prob[target_id]).item()
        token_perplexities.append(token_perplexity)'''
    
    perplexity = torch.exp(loss)
    #tokens = perplexity_tokenizer.convert_ids_to_tokens(input_ids[0])
    return perplexity.item()

def detect_pos(doc_paraphrased, doc_original, added_tokens, removed_tokens, pos_tags_to_check, allowed_deps):
    """
    Identifies POS tags of added and removed tokens in a paraphrased sentence, 
    and flags any that do not match the expected POS or syntactic dependencies.
    """
    added_pos_tags = []
    wrong_added = []

    for i, token in enumerate(doc_paraphrased):
        key = (token.text, i)
        if key in added_tokens:
            added_pos_tags.append(token.pos_)
            if (
                token.pos_ not in pos_tags_to_check and
                token.dep_ not in allowed_deps and
                not token.is_space and
                not token.is_punct 
            ):
                wrong_added.append(token.text)
    
    removed_pos_tags = []
    wrong_removed = []

    for i, token in enumerate(doc_original):
        key = (token.text, i)
        if key in removed_tokens:
            removed_pos_tags.append(token.pos_)
            if (
                token.pos_ not in pos_tags_to_check and
                token.dep_ not in allowed_deps and
                not token.is_space and
                not token.is_punct
            ):
                wrong_removed.append(token.text)

    return added_pos_tags, removed_pos_tags, wrong_added, wrong_removed

def compute_cos_similarity(added_words, removed_words):
    """
    ...
    """
    added_embeddings = syn_model.encode(added_words, convert_to_tensor=True)
    removed_embeddings = syn_model.encode(removed_words, convert_to_tensor=True)
    
    # Compute cosine similarity matrix
    cosine_scores = util.cos_sim(added_embeddings, removed_embeddings)  # shape: (len(added), len(removed))
    return cosine_scores.mean().item() # note: computing average

def ngrams(seq, n):
    return set(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))

def jaccard_similarity(seq1, seq2, n=2):
    """Jaccard similarity between POS n-grams."""
    ngrams1, ngrams2 = ngrams(seq1, n), ngrams(seq2, n)
    if not ngrams1 and not ngrams2:
        return 1.0
    return len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)

def is_passive(sentence):
    """Check if a sentence is written in passive voice using spaCy dependency parse."""
    match=passivepy.match_text(sentence, full_passive=True, truncated_passive=True)
    return match["binary"]

def check_for_voice_changes(doc_paraphrased, doc_original):
    """
    Compare two texts sentence by sentence.
    Returns a list of results about active/passive shifts.
    """
    sentences1 = [sent.text.strip() for sent in doc_original.sents]
    sentences2 = [sent.text.strip() for sent in doc_paraphrased.sents]

    for i, (s1, s2) in enumerate(zip(sentences1, sentences2), start=1):
        passive1 = is_passive(s1).any()
        passive2 = is_passive(s2).any()

        if not passive1 and passive2:
            return True
        elif passive1 and not passive2:
            return True
    return False

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

    pos_original=[token.pos_ for token in doc_original if not token.is_stop and token.is_alpha]
    pos_paraphrase= [token.pos_ for token in doc_paraphrased if not token.is_stop and token.is_alpha]
    
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
                
    
    if modification=='prepositions' or modification=='random': #Specific metrics for the preposition modification 
        
        pos_tags_to_check = {'DET', 'ADV', "ADP", "SCONJ", 'CCONJ', 'PART', 'PRON'} 
        allowed_deps = {"prep"}

        added_pos_tags, removed_pos_tags, wrong_added, wrong_removed=detect_pos(doc_paraphrased, doc_original, added_tokens, removed_tokens, pos_tags_to_check, allowed_deps)

        metrics.update({
            'pos_added': added_pos_tags, 
            'pos_removed':removed_pos_tags, 
            'wrong_added': wrong_added, 
            "wrong_removed": wrong_removed
        })
    
    if modification=='AAE' or modification=='random': #Specific metrics for the AAE modification 
        aae_pred_original=detect_AAE(original_context)[0]
        aae_pred_par=detect_AAE(paraphrase)[0]
        metrics.update({
            "aae_label_ori": aae_pred_original["label"],
            "aae_proba_ori": aae_pred_original["score"],
            "aae_label_par": aae_pred_par["label"],
            "aae_proba_par": aae_pred_par["score"],
        })
    
    if modification=='formal' or modification=='random':
        formal_pred_original=detect_formal(original_context)[0]
        formal_pred_par=detect_formal(paraphrase)[0]
        metrics.update({
            "formal_label_ori": formal_pred_original["label"],
            "formal_proba_ori": formal_pred_original["score"],
            "formal_label_par": formal_pred_par["label"],
            "formal_proba_par": formal_pred_par["score"],
        })
    
    if modification=='synonym_substitution' or modification=='random':
        #cos_score=compute_cos_similarity(added_words, removed_words)
        #matcher = difflib.SequenceMatcher(None, pos_original, pos_paraphrase)
        #seq_ratio = matcher.ratio()
        
        metrics.update({
            #"seq_ratio": seq_ratio,
            "jac_pos_sim":jaccard_similarity(pos_original, pos_paraphrase)
        })
    
    if modification=='change_voice' or modification=='random':
        metrics.update({
            "voice_changed": check_for_voice_changes(doc_paraphrased, doc_original)
        })
    
    if other_metrics:
        ppl_ori=compute_perplexity(original_context)
        ppl_par=compute_perplexity(paraphrase)
        metrics.update({"bert_score": compute_bert_score(original_context, paraphrase), 
                "sbert_score": compute_sbert(original_context, paraphrase),
                "rouge_l": compute_rouge_l(original_context, paraphrase),
                'grammar': grammar_errors(original_context, paraphrase),
                "perplexity_par": ppl_par,
                "perplexity_original": ppl_ori,
                "perplexity_ratio":ppl_par/ppl_ori})
    
    return metrics

def build_excel(paraphrase_df, output_path, modification, dataset):
    """
    Creates an Excel file for human annotation from a DataFrame of paraphrases.

    Args:
        paraphrase_df (pd.DataFrame): The input DataFrame containing paraphrases and original contexts.
        output_path (str): The file path where the Excel file will be saved.
        modification (str): The type of paraphrase modification being filtered (e.g., "prepositions", "AAE").
        dataset (str): Dataset name ('BBQ', 'MMLU', etc).

    Returns:
        None. Writes an Excel file to `output_path` for annotation purposes.
    """
    #Columns per type of modification
    columns_per_modif={'AAE': ["label_ori", "label_par", "proba_ori", "proba_par"],
                       'formal': ["label_ori", "label_par", "proba_ori", "proba_par"],
            'prepositions': ['pos_added', 'pos_removed', 'wrong_added', "wrong_removed"],
            "synonym_substitution": ["seq_ratio", "jac_pos_sim"],
            "change_voice":["voice_changed"]}

    columns_per_dataset={'BBQ': ['Q_id', "disambiguated"],
                         "MMLU":[],
                         "HatEval":[]} #specific columns if needed per dataset
    
    annotations_df=pd.DataFrame(columns=columns_per_dataset[dataset]
                                        +['idx', 'modification',  'original', 'raw_answer', 'nb_modif', 
                                         'wrong_modif', 'realism', 'meaning', #columns for human annotation
                                         'added_words', 'removed_words', 'grammar', 
                                         "bert_score", "sbert_score", "rouge_l", 
                                         'perplexity_par', 'perplexity_original', "perplexity_ratio"]
                                         +columns_per_modif[modification] #specific columns to each type of modification
                                         ) 
    
    for idx, row in tqdm(paraphrase_df.iterrows(), total=paraphrase_df.shape[0]): #Iterating over the dataset
        if dataset == "BBQ":
            # iterate over ambiguous/disambiguated contexts
            for disambiguated in [False, True]:
                if disambiguated:
                    original_context = row["Disambiguating_Context"]
                    paraphrases = row["Disambiguating_Paraphrases"]
                else:
                    original_context = row["Ambiguous_Context"]
                    paraphrases = row["Ambiguous_Paraphrases"]

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

        elif dataset == "MMLU":
            paraphrases = row["paraphrases"]
            assert isinstance(paraphrases, list)

            for paraphrase in paraphrases:
                new_row = {
                    "idx": idx,
                    "original": row["question"], 
                    "raw_answer": paraphrase.replace("\n", "\\n"),
                }
                new_row.update(automatic_detection(row["question"], paraphrase, modification))
                annotations_df.loc[len(annotations_df)] = new_row

    #Exporting to excel
    annotations_df.to_excel(output_path, index=False)

def evaluate_paraphrase(original_text, paraphrase, modification):
    """
    Compute automatic detection metrics (modification-specific) and apply heuristic rules.
    Returns nb_modifs and a dict of boolean flags.
    """
    metrics_dict = automatic_detection(original_text, paraphrase, modification, other_metrics=True)
    nb_modifs = metrics_dict['nb_modif']
    perplexity_ratio = metrics_dict["perplexity_ratio"]
    sbert_score = metrics_dict["sbert_score"]
    bert_score = metrics_dict.get("bert_score")  # some mods may not provide it
    
    flags = {
        "prepositions": False,
        "AAE": False,
        "formal": False,
        "synonym_substitution": False,
        "change_voice": False,
    }

    if nb_modifs==0:
        return nb_modifs, flags

    if modification == "prepositions" or modification=='random':
        if perplexity_ratio < 1.85 and sbert_score > 0.8:
            wrong_added, wrong_removed = metrics_dict['wrong_added'], metrics_dict['wrong_removed']
            if not wrong_added and not wrong_removed:
                flags["prepositions"] = True
            else:
                wrong_added_lem = [nlp(w)[0].lemma_ for w in wrong_added]
                wrong_removed_lem = [nlp(w)[0].lemma_ for w in wrong_removed]
                wrong_added_stem = [stemmer.stem(w) for w in wrong_added]
                wrong_removed_stem = [stemmer.stem(w) for w in wrong_removed]
                if wrong_added_lem == wrong_removed_lem or wrong_added_stem == wrong_removed_stem:
                    flags["prepositions"] = True
    
    if modification == "AAE" or modification=='random':
        if sbert_score > 0.75:
            pred_label = metrics_dict["aae_label_par"]
            proba_par, proba_ori = metrics_dict["aae_proba_par"], metrics_dict["aae_proba_ori"]
            if pred_label == "LABEL_1" or (proba_par < proba_ori and proba_par <= 0.9):
                flags["AAE"] = True
    
    if modification == "formal" or modification=='random':
        if perplexity_ratio < 2 and sbert_score > 0.75:
            pred_label, pred_ori = metrics_dict["formal_label_par"], metrics_dict["formal_label_ori"]
            proba_par, proba_ori = metrics_dict["formal_proba_par"], metrics_dict["formal_proba_ori"]
            if pred_label == "formal" or (pred_label == "neutral" and pred_ori =="neutral" and proba_par < proba_ori):
                flags["formal"] = True
    
    if modification == "synonym_substitution" or modification=='random':
        if perplexity_ratio < 2.5 and sbert_score > 0.85 and metrics_dict["jac_pos_sim"]>0.8: #and metrics_dict["seq_ratio"] > 0.80:
            flags["synonym_substitution"] = True
    
    if modification == "change_voice" or modification=='random':
        if perplexity_ratio < 1.8 and bert_score > 0.93 and sbert_score > 0.90 and metrics_dict["voice_changed"]:
            flags["change_voice"] = True
    
    return nb_modifs, flags

def filter_paraphrases(original_text, paraphrases, modification, nb_untouched, nb_wrong):
    """
    Apply modification-specific heuristics to filter out low-quality paraphrases.

    Args:
        original_text (str): The source/original text to compare against.
        paraphrases (List[str]): List of candidate paraphrases to filter.
        modification (str): The type of modification applied (e.g., "prepositions").
        nb_untouched (int): Counter for paraphrases that contained no modifications.
        nb_wrong (int): Counter for paraphrases that failed automatic filtering rules.

    Returns:
        Tuple[List[str], int, int]:
            - Filtered list of paraphrases (only those passing all heuristics).
            - Updated `nb_untouched` count.
            - Updated `nb_wrong` count.
    """
    filtered = []
    for paraphrase in paraphrases:
        nb_modifs, flags = evaluate_paraphrase(original_text, paraphrase, modification)
        
        if nb_modifs == 0:
            nb_untouched += 1
            continue
        
        if flags.get(modification, True):
            filtered.append(paraphrase)
        else:
            nb_wrong += 1
    
    return filtered, nb_untouched, nb_wrong

def classify_paraphrases(original_text, paraphrases):
    """ 
    Apply modification-specific heuristics to identify the type of paraphrase used. 
    
    Args: 
        original_text (str): The source/original text to compare against. 
        paraphrases (List[str]): List of candidate paraphrases to filter.  

    Returns: 
        classifications (dict): Dictionary where keys are modifications and 
            values are counts of paraphrases classified as that modification type.

    """
    classifications = None
    for paraphrase in list(paraphrases):
        nb_modifs, mod_flags = evaluate_paraphrase(original_text, paraphrase, "random")
        if nb_modifs == 0:
            continue  # untouched case handled outside

        if classifications is None:
            classifications = {mod: 0 for mod in mod_flags.keys()}
        
        for mod, flag in mod_flags.items():
            if flag:
                classifications[mod] += 1
        
    return classifications

def filter_out(paraphrase_df, output_path, modification, dataset):
    """
    Filters out rows from a paraphrase DataFrame based on modification-specific heuristics.

    Args:
        paraphrase_df (pd.DataFrame): The input DataFrame containing paraphrases and original contexts.
        output_path (str): Path where a filtered version of the DataFrame is saved (csv format).
        modification (str): The type of paraphrase modification being filtered (e.g., "prepositions", "AAE").
        dataset (str): Dataset name ('BBQ', 'MMLU', etc).

    Returns:
        List[int]: Indices of rows to keep after filtering.
    """
    nb_replaced, nb_untouched, nb_wrong=0, 0, 0 #counters for untouched sentences (no modifications) and wrong paraphrases
    for idx, row in tqdm(paraphrase_df.iterrows(), total=paraphrase_df.shape[0]): 
        if dataset == "BBQ":
            for _, disambiguated in enumerate([False, True]): 
                if disambiguated:
                    key="Disambiguating_Context"
                    original_text=row["Disambiguating_Context"]
                    paraphrases=row["Disambiguating_Paraphrases"]
                else: 
                    key="Ambiguous_Context"
                    original_text=row["Ambiguous_Context"]
                    paraphrases=row["Ambiguous_Paraphrases"]
                
                assert isinstance(paraphrases, list)

                paraphrases, nb_untouched, nb_wrong=filter_paraphrases(original_text, paraphrases, modification, nb_untouched, nb_wrong)
            
                if len(paraphrases)>0: #at least one paraphrase is correct
                    paraphrase_df.loc[idx, key] = paraphrases[0] #we take the first one
                else: #all paraphrases were incorrect
                    paraphrase_df.loc[idx, key]=original_text #we keep original context
                    nb_replaced+=1
        elif dataset == "MMLU":
            original_text=row["question"]
            paraphrases=row["paraphrases"]

            assert isinstance(paraphrases, list)

            paraphrases, nb_untouched, nb_wrong=filter_paraphrases(original_text, paraphrases, modification, nb_untouched, nb_wrong)
            
            if len(paraphrases)>0: #at least one paraphrase is correct
                paraphrase_df.loc[idx, "question"] = paraphrases[0] #we take the first one
            else: #all paraphrases were incorrect
                paraphrase_df.loc[idx, "question"]=original_text #we keep original text
                nb_replaced+=1

    total = len(paraphrase_df) * 2 if dataset == "BBQ" else len(paraphrase_df)
    print(f"Number of replaced contexts out of {total}:", nb_replaced) 
    print("Number of untouched paraphrases:", nb_untouched) 
    print("Number of wrong paraphrases:", nb_wrong) 
    paraphrase_df.to_csv(output_path, index=False)
    print("Filtered dataframe saved to", output_path)

def classify_data(paraphrase_df, output_data_path, output_classification_path, dataset, export=True, classify=True):
    """
    Classify paraphrases from a paraphrase DataFrame based on modification-specific heuristics.

    Args:
        paraphrase_df (pd.DataFrame): The input DataFrame containing paraphrases and original contexts.
        output_path (str): Path where a filtered version of the DataFrame is saved (csv format).
        dataset (str): Dataset name ('BBQ', 'MMLU', etc).

    Returns:
        List[int]: Indices of rows to keep after filtering.
    """
    classification_rows = []
    expanded_rows = []
    for idx, row in tqdm(paraphrase_df.iterrows(), total=paraphrase_df.shape[0]): 
        if dataset == "BBQ":
            for _, disambiguated in enumerate([False, True]): 
                if disambiguated:
                    key="Disambiguating_Context"
                    original_text=row["Disambiguating_Context"]
                    paraphrases=row["Disambiguating_Paraphrases"]
                else: 
                    key="Ambiguous_Context"
                    original_text=row["Ambiguous_Context"]
                    paraphrases=row["Ambiguous_Paraphrases"]
                
                assert isinstance(paraphrases, list)

                if classify:
                    classifications=classify_paraphrases(original_text, paraphrases)
                    classification_rows.append(classifications)

                for p in paraphrases:
                    new_row = row.copy()
                    new_row[key] = p
                    expanded_rows.append(new_row)

        elif dataset == "MMLU":
            original_text=row["question"]
            paraphrases=row["paraphrases"]

            assert isinstance(paraphrases, list)

            if classify:
                classifications=classify_paraphrases(original_text, paraphrases)
                classification_rows.append(classifications)

            for p in paraphrases:
                new_row = row.copy()
                new_row["question"] = p
                expanded_rows.append(new_row)
    
    if classify:
        classification_df = pd.DataFrame(classification_rows)
        classification_df.to_csv(output_classification_path, index=False)
    if export:
        # save expanded paraphrase dataset
        expanded_df = pd.DataFrame(expanded_rows)
        expanded_df.to_csv(output_data_path, index=False)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--modification', type=str, default='prepositions',
                        help="Type of modification to apply (e.g., 'prepositions')")
    parser.add_argument('--model', type=str, default='deepseek',
                        help="Model to use (e.g., 'deepseek')")
    parser.add_argument(
        "--dataset",
        choices=["BBQ", "HatEval", "MMLU"],
        type=str,
        default="BBQ",
        help="Specify the dataset used to paraphrase."
    )
    
    parser.add_argument(
        "--category",
        type=str,
        default="None",
        help="Specify a single category to paraphrase (e.g., 'Race_ethnicity' for BBQ or 'philosophy' for MMLU)."
    )

    parser.add_argument('--building', action='store_true',
                        help="Building the excel file for annotations (default: False)")
    parser.add_argument('--filtering', action='store_true',
                        help="Filtering the dataframe (default: False)")

    args = parser.parse_args()

    modification = args.modification
    model = args.model
    dataset = args.dataset
    category=args.category
    building = args.building
    filtering = args.filtering

    print(f"Results for dataset {dataset} for the subset {category} modified with {modification} generated by {model}")

    #Paths
    DATA_FOLDER=f'./data/{dataset}/paraphrases/{modification}/'
    PARAPHRASE_FILE=DATA_FOLDER+f"{category}_{modification}_{model}.csv"
    OUTPUT_EXCEL_FILE = DATA_FOLDER+f"{category}_{modification}_{model}.xlsx"
    OUTPUT_FILTERED_FILE = DATA_FOLDER+f"{category}_{modification}_{model}_filtered.csv"
    OUTPUT_CLASSIFICATION_FILE = DATA_FOLDER+f"{category}_{modification}_{model}_classified.csv"

    #Loading data
    paraphrase_df=pd.read_csv(PARAPHRASE_FILE)
    if dataset=='BBQ':
        paraphrase_df["Disambiguating_Paraphrases"]=paraphrase_df["Disambiguating_Paraphrases"].apply(ast.literal_eval)
        paraphrase_df["Ambiguous_Paraphrases"]=paraphrase_df["Ambiguous_Paraphrases"].apply(ast.literal_eval)
    else:
        paraphrase_df["paraphrases"]=paraphrase_df["paraphrases"].apply(ast.literal_eval)
    
    if modification=='random':
        classify_data(paraphrase_df, OUTPUT_FILTERED_FILE, OUTPUT_CLASSIFICATION_FILE, dataset, filtering, building)

    else:
        if building:
            #Building the excel for annotation
            build_excel(paraphrase_df, OUTPUT_EXCEL_FILE, modification, dataset)
        
        if filtering:
            #Filtering low quality paraphrases
            filter_out(paraphrase_df, OUTPUT_FILTERED_FILE, modification, dataset)