"""
Model configurations and constants.
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    model_id: str
    provider: str

MODEL_CONFIGS = {
    "deepseek": ModelConfig(
        name="deepseek",
        model_id="deepseek-chat",
        provider="deepseek"
    ),
    "chatgpt": ModelConfig(
        name="chatgpt",
        #model_id="gpt-4o-mini",
        model_id="gpt-4.1",
        provider="openai"
    ),
    "claude": ModelConfig(
        name="claude",
        model_id="claude-sonnet-4-5-20250929",
        provider="anthropic"
    ),
}