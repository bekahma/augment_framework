"""
LLM judge implementation supporting multiple model providers.
"""

from typing import Optional
import torch
from openai import OpenAI
import anthropic

from .model_config import MODEL_CONFIGS, ModelConfig
from .llama_cache import _llama_cache


class LLMJudge:
    """Handles LLM-based judgment of paraphrases."""
    
    def __init__(
        self, 
        model_name: str, 
        llama_model_id: Optional[str] = None,
    ):
        """
        Initialize LLM judge.
        
        Args:
            model_name: Name of the model (chatgpt, deepseek, claude, llama)
            llama_model_id: Specific Llama model ID
            model_weights_dir: Directory containing local model weights
        """
        self.model_name = model_name
        self.llama_model_id = llama_model_id or "Meta-Llama-3-8B-Instruct"
        
        if model_name == "llama":
            self.config = ModelConfig(
                name="llama",
                model_id=self.llama_model_id,
                provider="llama"
            )
        elif model_name in MODEL_CONFIGS:
            self.config = MODEL_CONFIGS[model_name]
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate API client."""
        if self.config.provider == "openai":
            return OpenAI()
        elif self.config.provider == "deepseek":
            return OpenAI(base_url="https://api.deepseek.com")
        elif self.config.provider == "anthropic":
            return anthropic.Anthropic()
        elif self.config.provider == "llama":
            return None  # Llama uses local model
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def get_response(self, prompt: str, system_msg: str = "") -> str:
        """
        Get response from the LLM.
        
        Args:
            prompt: User prompt
            system_msg: System message
            
        Returns:
            Model response as string
        """
        if self.config.provider == "llama":
            return self._get_llama_response(prompt, system_msg)
        elif self.config.provider == "anthropic":
            return self._get_claude_response(prompt, system_msg)
        else:
            return self._get_openai_response(prompt, system_msg)
    
    def _get_llama_response(self, prompt: str, system_msg: str) -> str:
        model, tokenizer = _llama_cache.load(self.llama_model_id) # default directory is /model-weights
        
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
    
    def _get_claude_response(self, prompt: str, system_msg: str) -> str:
        message = self.client.messages.create(
            model=self.config.model_id,
            max_tokens=1024,
            temperature=0,
            system=system_msg,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    
    def _get_openai_response(self, prompt: str, system_msg: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            stream=False,
        )
        return (response.choices[0].message.content or "").strip()