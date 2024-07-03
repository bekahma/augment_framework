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

    def calculate_loglikelihood(self, context, ans):
        return log_likelihood(self.model, self.tokenizer, context, ans, self.device)

    def pred_MCP(self, context, answer_candidates, return_candidates):
        pred = [self.calculate_loglikelihood(context, ans) for ans in answer_candidates]
        return return_candidates[pred.index(max(pred))]
