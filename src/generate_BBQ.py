"""This script is to generate the whole BBQ dataset, once contexts have been rephrased and filtered. 
Adapted from https://github.com/nyu-mll/BBQ and https://github.com/rem-h4/llm_socialbias_prompts"""

import pandas as pd
import io
import json
import random
import argparse
import ast
from utils import *
from typing import List, Set, Union

# Constants
CATEGORIES = [
    "Disability_status", "Age", "Physical_appearance", "SES", "Gender_identity",
    "Race_ethnicity", "Religion", "Nationality", "Sexual_orientation"
]

UNKNOWN_OPTIONS = [
    "Unknown", "Cannot be determined", "Can't be determined", "Not answerable",
    "Not known", "Not enough info", "Not enough information", "Cannot answer",
    "Can't answer", "Undetermined"
]

NEED_STEREOTYPING_SUBSET = ["Race_ethnicity", "Religion", "Nationality", "Sexual_orientation"]

#Utils functions

def downsample(lst, n=5):
    return random.sample(lst, min(len(lst), n))

def should_use_proper_names(row):
    return row.get("Proper_nouns_only", "") == "TRUE"

def get_proper_name_list(category, bias_targets, names_vocab):
    '''Comment this function'''
    if category == "Race_ethnicity":
        names = names_vocab[names_vocab.First_last == "first"]
        if len(bias_targets) >= 1:
            names = names[names.ethnicity.isin(bias_targets)]
        return downsample(names.Name.tolist())

    if category == "Gender_identity":
        names = names_vocab[(names_vocab.First_last == "first_only") & (names_vocab.gender == "F")]
        return downsample(names.Name.tolist())

    names = names_vocab[names_vocab.First_last == "first_only"]
    return downsample(names.Name.tolist(), 6)

def get_name_info(first_names, name, field):
    try:
        return first_names.loc[first_names["Name"] == name, field].iloc[0]
    except IndexError:
        return name

def get_new_word_list(
        cat: str,
        this_word: str,
        word_list: List[str],
        possible_word_list: List[str],
        bias_targets: Set[str],
        need_stereotyping_subset: Set[str],
        has_proper_name: bool,
        words: pd.DataFrame,
        first_names: pd.DataFrame,
        first_names_full: pd.DataFrame,
        names_vocab: pd.DataFrame,
    ) -> Union[List[str], tuple[List[str], str, dict]]:
        """
        Generate a new list of word options based on the category and current word, avoiding bias targets.

        Args:
            cat: The category (e.g. "SES", "Gender_identity", "Race_ethnicity").
            this_word: The current word being considered.
            word_list: Original list of words.
            possible_word_list: List of potential replacement words.
            bias_targets: Set of bias-targeted words for the template.
            need_stereotyping_subset: Categories that need bias-target filtering.
            has_proper_name: Whether the word is a proper name.
            words: DataFrame with columns ["Name", "Info"] for SES.
            first_names: DataFrame with columns ["Name", "gender", "ethnicity"].
            first_names_full: Full version of first_names (same structure).
            names_vocab: DataFrame with columns ["Name", "First_last", "ethnicity"].

        Returns:
            A list of new word candidates or a tuple of (new_word_list, Name1_info, Name2_info_dict)
            when gender or race/ethnicity information is needed.
        """
        Name1_info=None
        Name2_info_dict = {}

        if (len(bias_targets) > 0) and (cat in need_stereotyping_subset):
            new_word_list = [n for n in possible_word_list if n not in bias_targets]
            if len(new_word_list) > 4:
                new_word_list = random.sample(new_word_list, 5)
        else:
            new_word_list = [n for n in word_list if n != this_word]

        # For SES
        if cat == "SES" and not has_proper_name:
            the_word_cat = words.loc[words["Name"] == this_word, "Info"].iloc[0]
            Name1_info = the_word_cat
            new_options = words[words.Info != the_word_cat]
            new_word_list = new_options.Name.unique().tolist()
            return new_word_list, Name1_info, Name2_info_dict

        # For Gender Identity
        if cat == "Gender_identity" and has_proper_name:
            the_word_gender = first_names.loc[
                first_names["Name"] == this_word, "gender"
            ].iloc[0]
            Name1_info = the_word_gender
            new_options = first_names[first_names.gender != the_word_gender]
            new_word_list = new_options.Name.unique().tolist()
            new_word_list = random.sample(new_word_list, 5)
            return new_word_list, Name1_info, Name2_info_dict

        # For Race/Ethnicity
        if cat == "Race_ethnicity" and has_proper_name:
            the_word_gender = first_names.loc[
                first_names["Name"] == this_word, "gender"
            ].iloc[0]
            the_word_eth = first_names.loc[
                first_names["Name"] == this_word, "ethnicity"
            ].iloc[0]
            Name1_info = f"{the_word_gender}-{the_word_eth}"

            # Choose a last name for this_word
            last_names = names_vocab[
                (names_vocab.First_last == "last") & (names_vocab.ethnicity == the_word_eth)
            ]
            last_names_list = last_names.Name.unique().tolist()
            this_last_name = random.choice(last_names_list)
            this_word = f"{this_word} {this_last_name}"

            # Get second names with same gender but non-bias ethnicities
            other_first_names = first_names_full[
                (~first_names_full.ethnicity.isin(bias_targets)) &
                (first_names_full.gender == the_word_gender)
            ].reset_index(drop=True)
            other_first_names = other_first_names.sample(n=5, replace=False)

            other_last_names = names_vocab[
                (names_vocab.First_last == "last") &
                (~names_vocab.ethnicity.isin(bias_targets))
            ]

            new_word_list = []

            for nam in range(len(other_first_names)):
                frst = other_first_names.Name[nam]
                eth = other_first_names.ethnicity[nam]
                gen = other_first_names.gender[nam]
                Name2_info = f"{gen}-{eth}"
                lst_list = other_last_names[other_last_names.ethnicity == eth].Name.unique().tolist()
                lst = random.choice(lst_list)
                full_name = f"{frst} {lst}"
                new_word_list.append(full_name)
                Name2_info_dict[full_name] = Name2_info

            return new_word_list, Name1_info, Name2_info_dict

        return new_word_list, Name1_info, Name2_info_dict


def process_pair(dat_file, category, frame_row, name1, name2, frame_cols, unknown_options, bias_targets, subcat, info1, info2, item_id):
    slotted_frame = do_slotting(
        frame_row, frame_cols, name1, None, name2, None, [], "", ""
    )

    formatted_data = create_templating_dicts(
        category, slotted_frame, subcat, unknown_options, frame_cols,
        bias_targets, name1, name2, info1, info2, item_id
    )

    for item in formatted_data:
        dat_file.write(json.dumps(item, default=str) + "\n")
    dat_file.flush()
    return item_id + 4

def generate_data_for_category(category, data_path, output_path, vocab, names_vocab):
    #Reading templates
    frames = pd.read_csv(data_path, na_filter=False)
    frames=frames[frames.Ambiguous_Context != ""].reset_index()
    frame_cols = frames.columns

    #Writing data
    with io.open(output_path, "w") as dat_file:
        item_id = 0

        for idx, row in frames.iterrows():
            words = vocab[vocab.Category == category]
            possible_words = words.Name.unique().tolist()

            bias_targets = ast.literal_eval(row.Known_stereotyped_groups)
            targeted_words = [
                w for w in possible_words if w in bias_targets
            ] if bias_targets and category in NEED_STEREOTYPING_SUBSET else possible_words

            # if the list of bias targets is too big, downsample
            word_list = downsample(targeted_words) #TODO what if possible_wordl_list<2?
            has_proper_name = should_use_proper_names(row)
            subcat = row.get("Subcategory", "None")
            if len(subcat) > 1:
                words = words[words.SubCat == subcat]
                word_list = words.Name.unique().tolist()
                if len(word_list) > 5:
                    word_list = random.sample(word_list, 5) 
            else:
                subcat = "None"

            custom_names = row.get("Names", "")
            if len(custom_names) > 1:
                word_list, new_word_list = return_list_from_string(custom_names)

            if has_proper_name:
                word_list, first_names, first_names_full = get_proper_name_list(category, bias_targets, names_vocab)
            else:
                first_names, first_names_full=None, None

            for name1 in word_list:
                if len(custom_names)<2:
                    new_word_list, Name1_info, Name2_info_dict=get_new_word_list(category, name1, word_list, 
                                                                                 possible_words, bias_targets, NEED_STEREOTYPING_SUBSET, 
                                                                                 has_proper_name, words, first_names, first_names_full, 
                                                                                 names_vocab)

                for name2 in new_word_list:
                    item_id = process_pair(
                        dat_file, category, frames.iloc[[idx]].reset_index(), name1, name2,
                        frame_cols, UNKNOWN_OPTIONS, bias_targets, subcat,
                        Name1_info, Name2_info_dict.get(name2, name2), item_id
                    )

                    if category in NEED_STEREOTYPING_SUBSET or has_proper_name or (category == "SES" and subcat == "Occupation"):
                        item_id = process_pair(
                            dat_file, category, frames.iloc[[idx]].reset_index(), name2, name1,
                            frame_cols, UNKNOWN_OPTIONS, bias_targets, subcat,
                            Name2_info_dict.get(name2, name2), Name1_info, item_id
                        )

        print(f"Generated {item_id} items for category: {category}")

def convert_to_prompt(output_path):
        with open(output_path, "r") as f:
            jsonl_data = [json.loads(line) for line in f.readlines()]
        
        context_text = "Context: "
        question_text = "Question: "
        answer_text = "Answer: "
        a_text, b_text, c_text = "A: ", "B: ", "C: "

        for jd in jsonl_data:
            context = jd["context"]
            question = jd["question"]
            ans0, ans1, ans2 = jd["ans0"], jd["ans1"], jd["ans2"]
            tmp = f"{context_text}{context}\n{question_text}{question}\n{a_text} {ans0}\n{b_text} {ans1}\n{c_text} {ans2}\n{answer_text}"

            jd["prompt"] = tmp
            if "A" in a_text:
                jd["enum_choices"] = ["A", "B", "C"]
            elif "a" in a_text:
                jd["enum_choices"] = ["a", "b", "c"]
        
        with open(output_path, "w") as f:
            for jd in jsonl_data:
                f.write(json.dumps(jd) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BBQ dataset for bias evaluation.")
    parser.add_argument(
        "--category",
        type=str,
        default="Gender_identity",
        help="Specify a single category to generate (e.g., 'Race_ethnicity'), or 'all' to process everything."
    )

    parser.add_argument('--modification', type=str, default='prepositions',
                        help="Type of modification to apply (e.g., 'prepositions')")
    parser.add_argument('--model', type=str, default='deepseek',
                        help="Model to use (e.g., 'deepseek')")
    
    args = parser.parse_args()

    #Paths
    VOC_FOLDER='./data/BBQ_templates/'
    DATA_FOLDER='./data/paraphrases/'
    modification = args.modification
    model = args.model

    # read in vocabulary files
    vocab = pd.read_csv(VOC_FOLDER+"vocabulary.csv")
    vocab = vocab[vocab.Pilot_include != "No"]
    names_vocab = pd.read_csv(VOC_FOLDER+"vocabulary_proper_names.csv")

    if args.category == "all":
        for category in CATEGORIES:
            data_path=DATA_FOLDER+"new_templates - %s.csv" % category
            generate_data_for_category(category)
    else:
        if args.category not in CATEGORIES:
            raise(f"{args.category}' is not a valid category.")
        DATA_PATH=DATA_FOLDER+f"{args.category}_{modification}_{model}.csv" 
        OUTPUT_PATH=f"./data/jsonl/eval_prompt_{modification}_{model}.jsonl"
        generate_data_for_category(args.category, DATA_PATH, OUTPUT_PATH, vocab, names_vocab)
        convert_to_prompt(OUTPUT_PATH)