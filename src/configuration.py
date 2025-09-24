#Configuration

subsets={'BBQ':["Disability_status", "Age", "Physical_appearance", 
                "SES", "Gender_identity", "Race_ethnicity", "Religion", 
                "Nationality", "Sexual_orientation"],
        'MMLU':['abstract_algebra', 'college_chemistry', 
                'computer_security', 'econometrics', 'us_foreign_policy',
                'international_law', 'philosophy', 'global_facts']} 

modifications={"original":["None"],
                'prepositions': ["chatgpt", "deepseek"], 
                'synonym_substitution': ["chatgpt", "deepseek"], 
                'change_voice': ["chatgpt", "deepseek"], 
                'AAE': ["chatgpt", "deepseek"], 
                'formal': ["chatgpt", "deepseek"],
                'random': ["chatgpt", 'deepseek']
            } 

target_models=["Llama-3-8B", "Llama-3-8B-Instruct", 
                "tiiuae-falcon-7b", "tiiuae-falcon-7b-instruct",
                "mosaicml-mpt-7b", "mosaicml-mpt-7b-instruct",
                "google-gemma-3-1b-it",
                "google-gemma-3-4b-it", 
                "google-gemma-3-12b-it"
            ]

metrics = {'BBQ':{
    "overall_acc": "Overall Accuracy",
    "ambig_acc": "Accuracy in Ambiguous Contexts",
    "disambig_acc": "Accuracy in Disambiguated Contexts",
    "ambig_bias": "Bias Scores in Ambiguous Contexts",
    "disambig_bias": "Bias Scores in Disambiguated Contexts"
    },
    'MMLU': {
    "overall_acc": "Overall Accuracy",}}

letters_index={'BBQ': {"A": 0, "B": 1, "C": 2},
               'MMLU': {"A": 0, "B": 1, "C": 2, "D":3}}