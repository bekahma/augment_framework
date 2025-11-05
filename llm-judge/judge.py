"""
Script for judging paraphrases using LLM models.
Supports both simple TRUE/FALSE judgments and reasoning-based evaluations.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic


# Global variables for Llama model caching
llama_model = None
llama_tokenizer = None


def get_client(model_name):
    """Initialize client for specified model."""
    if model_name == "deepseek":
        return OpenAI(base_url="https://api.deepseek.com"), "deepseek-chat"
    elif model_name == "chatgpt":
        return OpenAI(), "gpt-4o-mini"
    elif model_name == "claude":
        return anthropic.Anthropic(), "claude-sonnet-4-5-20250929"
    elif model_name == "llama":
        return None, "llama"
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_claude_response(prompt, system_msg=""):
    """Generate response from Claude."""
    client, model = get_client("claude")
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system=system_msg or "You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text.strip()


def load_llama_model():
    """Load Llama 3 8B model and tokenizer (cached globally)."""
    global llama_model, llama_tokenizer
    
    if llama_model is None:
        print("Loading Llama 3 8B model...")
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
        llama_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Model loaded successfully!")
    
    return llama_model, llama_tokenizer


def get_llama_response(prompt, system_msg=""):
    """Generate response from Llama 3 8B."""
    model, tokenizer = load_llama_model()
    
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    return response.strip()


def build_judge_prompt(instruction_text, original_sentence, paraphrased_sentence, with_reasoning=False):
    """
    Build a prompt for judging paraphrases.
    
    Args:
        instruction_text: The paraphrasing instruction to evaluate against
        original_sentence: The original sentence
        paraphrased_sentence: The paraphrased sentence to judge
        with_reasoning: If True, asks for reasoning; otherwise just TRUE/FALSE
    """
    base_prompt = f"""
You are a strict judge evaluating paraphrases.

Paraphrasing instructions:
{instruction_text}

Determine whether the following paraphrased sentence correctly follows the above instructions.

Original sentence:
"{original_sentence}"

Paraphrased sentence:
"{paraphrased_sentence}"
"""

    if with_reasoning:
        base_prompt += """
Respond with TRUE or FALSE prefixed with 'DECISION:'.
Explain your reasoning in a concise manner prefixed with 'REASON:'.
"""
    else:
        base_prompt += """
If it follows the instruction, respond with "TRUE".
If it does not, respond with "FALSE".
Do not include explanations or additional text.
"""
    
    return base_prompt.strip()


def get_llm_judgment(prompt, model_choice, system_msg, with_reasoning=False):
    """
    Get judgment from specified LLM model.
    
    Args:
        prompt: The judgment prompt
        model_choice: Model to use ('chatgpt', 'deepseek', 'llama', 'claude')
        system_msg: System message for the model
        with_reasoning: If True, parse both decision and reasoning
    
    Returns:
        If with_reasoning: (decision, reason)
        Otherwise: decision
    """
    try:
        if model_choice == "llama":
            content = get_llama_response(prompt, system_msg)
        elif model_choice == "claude":
            content = get_claude_response(prompt, system_msg)
        else:
            client, model_name = get_client(model_choice)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                stream=False,
            )
            content = (response.choices[0].message.content or "").strip()

        if with_reasoning:
            # Parse decision
            m_dec = re.search(r'\bDECISION:\s*(TRUE|FALSE)\b', content, re.IGNORECASE)
            decision = m_dec.group(1).upper() if m_dec else (
                content.upper() if content.upper() in {"TRUE", "FALSE"} else "UNKNOWN"
            )

            # Parse reasoning
            m_rea = re.search(r'\bREASON:\s*(.*)', content, re.IGNORECASE | re.DOTALL)
            reason = m_rea.group(1).strip() if m_rea else (
                content if decision == "UNKNOWN" else ""
            )
            
            return decision, reason
        else:
            # Simple TRUE/FALSE parsing
            verdict = content.strip()
            if verdict not in ["TRUE", "FALSE"]:
                verdict = "UNKNOWN"
            return verdict

    except Exception as e:
        if with_reasoning:
            return "UNKNOWN", f"Error: {e}"
        else:
            return "UNKNOWN"


def judge_paraphrases_simple(df, instruction_text, model_choice="chatgpt"):
    """
    Simple judging: adds only 'llm_TF' column.
    Used for mode='simple' where all rows use the same instruction.
    """
    results = []
    system_msg = "Please act as an impartial judge and evaluate the correctness of the responses based on the instructions."

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
        original = row.get("original") or row.iloc[0]
        paraphrase = row.get("paraphrase") or row.get("raw_answer") or row.iloc[1]

        prompt = build_judge_prompt(instruction_text, original, paraphrase, with_reasoning=False)
        verdict = get_llm_judgment(prompt, model_choice, system_msg, with_reasoning=False)
        results.append(verdict)

    df["llm_TF"] = results
    return df


def judge_paraphrases_with_reasoning(df, instructions_df, model_choice="chatgpt"):
    """
    Reasoning-based judging: adds both 'llm_TF' and 'llm_reason' columns.
    Used for mode='reasoning' where each row (could potentially have) its own modification type.
    """
    instr_map = dict(zip(instructions_df["modification"], instructions_df["prompt"]))
    decisions, reasons = [], []
    system_msg = "Please act as an impartial judge and evaluate the correctness of the responses based on the instructions."

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging with reasoning"):
        mod = (row.get("modification") or "").strip()
        instruction_text = instr_map.get(mod, "")
        
        if not instruction_text:
            decisions.append("UNKNOWN")
            reasons.append(f"No instruction found for modification='{mod}'")
            continue

        original = row.get("original") or row.iloc[0]
        paraphrase = row.get("paraphrase") or row.get("raw_answer") or row.iloc[1]

        prompt = build_judge_prompt(instruction_text, original, paraphrase, with_reasoning=True)
        decision, reason = get_llm_judgment(prompt, model_choice, system_msg, with_reasoning=True)
        
        decisions.append(decision)
        reasons.append(reason)

    df["llm_TF"] = decisions
    df["llm_reason"] = reasons
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Judge paraphrases using LLM models with optional reasoning."
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["simple", "reasoning"], 
        required=True,
        help="Mode: 'simple' for TRUE/FALSE only, 'reasoning' for TRUE/FALSE + explanation"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="chatgpt",
        choices=["chatgpt", "deepseek", "llama", "claude"],
        help="Model to use for judging"
    )
    
    # Simple mode arguments
    parser.add_argument(
        "--llm_folder", 
        type=str, 
        default="annotations/llm",
        help="[Simple mode] Folder containing annotated Excel files"
    )
    parser.add_argument(
        "--modif", 
        type=str,
        help="[Simple mode] Modification type (e.g., formal, prepositions)"
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        help="[Simple mode] Model used to generate paraphrases (for filename)"
    )
    
    # Reasoning mode arguments
    parser.add_argument(
        "--input", 
        type=str,
        help="[Reasoning mode] Path to the combined Excel file"
    )
    
    args = parser.parse_args()

    # Load instruction map
    instruction_file = "llm-judge/judge_instructions.tsv"
    if not os.path.exists(instruction_file):
        raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
    instructions_df = pd.read_csv(instruction_file, sep="\t")

    if args.mode == "simple":
        if not args.modif:
            raise ValueError("--modif is required for simple mode")
        if not args.generator_model:
            raise ValueError("--generator_model is required for simple mode")
        
        if args.modif not in instructions_df["modification"].values:
            raise ValueError(f"Modification '{args.modif}' not found in instruction file.")

        instruction_text = instructions_df.loc[
            instructions_df["modification"] == args.modif, "prompt"
        ].values[0]

        filename = f"llm_{args.modif}_{args.generator_model}.xlsx"
        filepath = os.path.join(args.llm_folder, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        print(f"Evaluating {filename} using {args.model} judge...")
        df = pd.read_excel(filepath)
        judged_df = judge_paraphrases_simple(df, instruction_text, args.model)
        
        # Save with judge model in filename
        output_filename = f"llm_{args.modif}_{args.generator_model}_judged_{args.model}.xlsx"
        output_path = os.path.join(args.llm_folder, output_filename)
        judged_df.to_excel(output_path, index=False)
        print(f"Finished: saved to {output_path}\n")

    elif args.mode == "reasoning":
        # Original judge_paraphrase_reasoning.py functionality
        if not args.input:
            raise ValueError("--input is required for reasoning mode")
        
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")

        df = pd.read_excel(args.input)

        if "modification" not in df.columns:
            raise ValueError("Input sheet must contain a 'modification' column.")

        print(f"Judging {args.input} with reasoning using {args.model}...")
        judged_df = judge_paraphrases_with_reasoning(df, instructions_df, args.model)
        
        output_path = args.input.replace(".xlsx", f"_judged_{args.model}.xlsx")
        judged_df.to_excel(output_path, index=False)
        print(f"Finished: saved to {output_path}\n")


if __name__ == "__main__":
    main()