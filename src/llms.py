import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import log_likelihood


class LLMs:
    def __init__(self, model_path, model_id, device, dtype=None):
        if dtype:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_id = model_id
        self.device = device
        self.is_chatmodel = hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None

    def format_prompt(self, prompt, system_prompt="You are a helpful assistant."):
        if self.is_chatmodel:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            return prompt
        
    def calculate_loglikelihood(self, context, ans):
        return log_likelihood(self.model, self.tokenizer, context, ans, self.device)

    def pred_MCP(self, prompt, answer_candidates, return_candidates, system_prompt="You are a helpful assistant."):
        context = self.format_prompt(prompt, system_prompt)
        pred = [self.calculate_loglikelihood(context, ans) for ans in answer_candidates]
        return return_candidates[pred.index(max(pred))]
    
    def pred_likelihoods(self, prompt, answer_candidates, system_prompt="You are a helpful assistant."):
        context = self.format_prompt(prompt, system_prompt)
        return [self.calculate_loglikelihood(context, ans) for ans in answer_candidates]
