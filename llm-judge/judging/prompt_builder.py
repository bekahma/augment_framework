"""
Prompt building utilities for paraphrase judgment.
"""

class PromptBuilder:
    """Builds prompts for paraphrase judgment."""
    
    @staticmethod
    def build_judge_prompt(
        instruction_text: str,
        original_sentence: str,
        paraphrased_sentence: str,
        with_reasoning: bool = False
    ) -> str:
        """
        Build a prompt for judging paraphrases.
        
        Args:
            instruction_text: The paraphrasing instruction to evaluate against
            original_sentence: The original sentence
            paraphrased_sentence: The paraphrased sentence to judge
            with_reasoning: If True, asks for reasoning; otherwise just TRUE/FALSE
        """
        base_prompt = f"""Paraphrasing instructions:
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
Explain your reasoning in a concise manner prefixed with 'REASON:'."""
        else:
            base_prompt += """
If it follows the instruction, respond with "TRUE".
If it does not, respond with "FALSE".
Do not include explanations or additional text."""
        
        return base_prompt.strip()