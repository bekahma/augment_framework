"""
Script for judging paraphrases using LLM models.
Supports both simple TRUE/FALSE judgments and reasoning-based evaluations.
"""

import os
import argparse
from typing import Dict

import pandas as pd

from models.llm_judge import LLMJudge
from judging.paraphrase_judger import ParaphraseJudger


def load_instructions(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Instruction file not found: {filepath}")
    return pd.read_csv(filepath, sep="\t")


def run_simple_mode(args, judger: ParaphraseJudger, instructions_df: pd.DataFrame):
    if not args.modif or not args.generator_model:
        raise ValueError("--modif and --generator_model are required for simple mode")
    
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
    judged_df = judger.judge_simple(df, instruction_text)
    
    output_filename = f"llm_{args.modif}_{args.generator_model}_judged_{args.model}.xlsx"
    output_path = os.path.join(args.llm_folder, output_filename)
    judged_df.to_excel(output_path, index=False)
    print(f"Finished: saved to {output_path}\n")


def run_reasoning_mode(args, judger: ParaphraseJudger, instructions_df: pd.DataFrame):
    if not args.input:
        raise ValueError("--input is required for reasoning mode")
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    df = pd.read_excel(args.input)
    
    # this handles multiple modification types in the same file
    if "modification" not in df.columns:
        raise ValueError("Input sheet must contain a 'modification' column.")
    
    instruction_map = dict(zip(
        instructions_df["modification"],
        instructions_df["prompt"]
    ))
    
    print(f"Judging {args.input} with reasoning using {args.model}...")
    judged_df = judger.judge_with_reasoning(df, instruction_map)
    
    output_path = args.input.replace(".xlsx", f"_judged_{args.model}.xlsx")
    judged_df.to_excel(output_path, index=False)
    print(f"Finished: saved to {output_path}\n")


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
        help="Model to use for judging (NOT GENERATOR LLM)"
    )
    parser.add_argument(
        "--llama_model",
        type=str,
        default="Meta-Llama-3-8B-Instruct",
        help="Specific Llama model to use (only applicable if --model is 'llama')"
    )
    # only change this for another cluster, don't include
    parser.add_argument(
        "--model_weights_dir",
        type=str,
        default="/model-weights",
        help="Directory containing local model weights (for Llama models)"
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
    parser.add_argument(
        "--instruction_file",
        type=str,
        default="llm-judge/judge_instructions.tsv",
        help="Path to instruction file"
    )
    
    args = parser.parse_args()
    
    # Load instructions
    instructions_df = load_instructions(args.instruction_file)
    
    # Initialize judge
    llm_judge = LLMJudge(
        args.model, 
        llama_model_id=args.llama_model,
        model_weights_dir=args.model_weights_dir
    )
    judger = ParaphraseJudger(llm_judge)
    
    # Run appropriate mode
    if args.mode == "simple":
        run_simple_mode(args, judger, instructions_df)
    elif args.mode == "reasoning":
        run_reasoning_mode(args, judger, instructions_df)


if __name__ == "__main__":
    main()