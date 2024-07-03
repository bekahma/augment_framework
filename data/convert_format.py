import json
import pandas as pd
import argparse
from pathlib import Path

df = pd.read_table("./data/template.tsv")
with open("eval_config.json", "r") as f:
    eval_config = json.load(f)

eval_base_file = eval_config["eval_base_file"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--few_shot", action="store_true")

    args = parser.parse_args()

    if args.few_shot:
        with open("./data/BBQ_few_shot.jsonl", "r") as f:
            few_shot_jsonl_data = [json.loads(line) for line in f.readlines()]

    def make_few_shot_samples(few_shot_jd, row):
        result = []
        context_text = row["context"]
        question_text = row["question"]
        a_text, b_text, c_text = row["a"], row["b"], row["c"]
        answer_text = row["answer"]

        for jd in few_shot_jd:
            context = jd["context"]
            question = jd["question"]
            ans0, ans1, ans2 = jd["ans0"], jd["ans1"], jd["ans2"]
            if "A" in a_text:
                answer = ["A", "B", "C"][jd["label"]]
            elif "a" in a_text:
                answer = ["a", "b", "c"][jd["label"]]

            tmp = f"{context_text}{context}\n{question_text}{question}\n{a_text} {ans0}\n{b_text} {ans1}\n{c_text} {ans2}\n{answer_text} {answer}"
            result.append(tmp)

        return "\n\n".join(result)

    for i, row in df.iterrows():
        with open(eval_base_file, "r") as f:
            jsonl_data = [json.loads(line) for line in f.readlines()]
        new_jsonl_data = []
        if args.few_shot:
            few_shot = make_few_shot_samples(few_shot_jsonl_data, row)
        instruction_text = row["instruction"]
        context_text = row["context"]
        question_text = row["question"]
        answer_text = row["answer"]
        prompt_id = row["prompt_id"]
        a_text, b_text, c_text = row["a"], row["b"], row["c"]

        answer_map = {}
        for jd in jsonl_data:
            context = jd["context"]
            question = jd["question"]
            ans0, ans1, ans2 = jd["ans0"], jd["ans1"], jd["ans2"]
            if args.few_shot:
                if isinstance(instruction_text, str):
                    tmp = f"{instruction_text}\n\n{few_shot}\n\n{context_text}{context}\n{question_text}{question}\n{a_text} {ans0}\n{b_text} {ans1}\n{c_text} {ans2}\n{answer_text}"
                else:
                    tmp = f"{few_shot}\n\n{context_text}{context}\n{question_text}{question}\n{a_text} {ans0}\n{b_text} {ans1}\n{c_text} {ans2}\n{answer_text}"
            else:
                if isinstance(instruction_text, str):
                    tmp = f"{instruction_text}\n\n{context_text}{context}\n{question_text}{question}\n{a_text} {ans0}\n{b_text} {ans1}\n{c_text} {ans2}\n{answer_text}"
                else:
                    tmp = f"{context_text}{context}\n{question_text}{question}\n{a_text} {ans0}\n{b_text} {ans1}\n{c_text} {ans2}\n{answer_text}"

            jd["prompt"] = tmp
            if "A" in a_text:
                jd["enum_choices"] = ["A", "B", "C"]
            elif "a" in a_text:
                jd["enum_choices"] = ["a", "b", "c"]

            new_jsonl_data.append(jd)

        if args.few_shot:
            output_fname = f"./data/jsonl/eval_prompt_fewshot_{prompt_id}.jsonl"
        else:
            output_fname = f"./data/jsonl/eval_prompt_{prompt_id}.jsonl"

        output_folder = Path("./data/jsonl/")
        output_folder.mkdir(exist_ok=True)

        with open(output_fname, "w") as f:
            for n_jd in new_jsonl_data:
                f.write(json.dumps(n_jd) + "\n")
