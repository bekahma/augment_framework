"""
Caching system for Llama models.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LlamaModelCache:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
    
    def load(self, model_name: str, model_weights_dir: str = "/model-weights"): # directory from cluster
        """
        Load Llama model from local weights directory.
        
        Args:
            model_name: Model name (e.g., "Meta-Llama-3-8B-Instruct")
            model_weights_dir: Directory containing local model weights
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model_path = os.path.join(model_weights_dir, model_name)
        
        if self.model is None or self.current_model_name != model_path:
            print(f"Loading model from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.current_model_name = model_path
            print("Model loaded successfully!")
        
        return self.model, self.tokenizer


# Global cache instance
_llama_cache = LlamaModelCache()