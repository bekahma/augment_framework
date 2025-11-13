"""
Main judging logic for paraphrases.
Returns df of corresponding judging columns.
"""

from typing import Dict
import pandas as pd
from tqdm import tqdm

from models.llm_judge import LLMJudge
from .prompt_builder import PromptBuilder
from .response_parser import ResponseParser


class ParaphraseJudger:
    def __init__(self, judge: LLMJudge):
        self.judge = judge
        self.prompt_builder = PromptBuilder()
        self.parser = ResponseParser()
        self.system_msg = (
            "Please act as an impartial judge and evaluate the correctness of the responses based on the instructions."
        )
    
    def judge_simple(
        self,
        df: pd.DataFrame,
        instruction_text: str
    ) -> pd.DataFrame:
        """
        Simple judging: adds only 'llm_TF' column.
        
        Args:
            df: DataFrame with paraphrases
            instruction_text: Instruction to evaluate against
        """
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
            original = row.get("original", row.iloc[0])
            paraphrase = row.get("paraphrase") or row.get("raw_answer", row.iloc[1])
            
            prompt = self.prompt_builder.build_judge_prompt(
                instruction_text, original, paraphrase, with_reasoning=False
            )
            
            try:
                content = self.judge.get_response(prompt, self.system_msg)
                verdict = self.parser.parse_simple(content)
            except Exception as e:
                print(f"Error processing row: {e}")
                verdict = "UNKNOWN"
            
            results.append(verdict)
        
        df["llm_TF"] = results
        return df
    
    def judge_with_reasoning(
        self,
        df: pd.DataFrame,
        instruction_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Reasoning-based judging: adds both 'llm_TF' and 'llm_reason' columns.
        
        Args:
            df: DataFrame with paraphrases and 'modification' column
            instruction_map: Map of modification types to instructions
        """
        decisions, reasons = [], []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging with reasoning"):
            mod = str(row.get("modification", "")).strip()
            instruction_text = instruction_map.get(mod, "")
            
            if not instruction_text:
                decisions.append("UNKNOWN")
                reasons.append(f"No instruction found for modification='{mod}'")
                continue
            
            original = row.get("original", row.iloc[0])
            paraphrase = row.get("paraphrase") or row.get("raw_answer", row.iloc[1])
            
            prompt = self.prompt_builder.build_judge_prompt(
                instruction_text, original, paraphrase, with_reasoning=True
            )
            
            try:
                content = self.judge.get_response(prompt, self.system_msg)
                decision, reason = self.parser.parse_with_reasoning(content)
            except Exception as e:
                print(f"Error processing row: {e}")
                decision = "UNKNOWN"
                reason = f"Error: {e}"
            
            decisions.append(decision)
            reasons.append(reason)
        
        df["llm_TF"] = decisions
        df["llm_reason"] = reasons
        return df