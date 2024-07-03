import torch
import numpy as np


def log_likelihood(model, tokenizer, context, ans, device):
    answer_probability = []
    tokens = tokenizer.encode(" ".join([context, ans]))
    tokens_context = tokenizer.encode(context)
    length = len(tokens_context)

    tokens_tensor = torch.tensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        output = torch.softmax(model(tokens_tensor)[0], dim=-1)

    for idx in range(length, len(tokens)):
        answer_probability.append(output[0, idx - 1, tokens[idx]].item())

    assert len(tokens) - len(tokens_context) == len(answer_probability)
    assert tokenizer.decode(tokens[length:]).strip() == ans.strip()

    score = np.average([np.log2(ap) for ap in answer_probability])
    return score


def make_prompt_for_chatmodel(text, prompt, model_name):
    if "Llama-2" in model_name:
        return f"<s>[INST] <<SYS>>\n{prompt}\n<</SYS>>\n\n{text} [/INST]"
    elif "falcon" in model_name:
        return f"{prompt}\n{text}"
    elif "mpt" in model_name:
        return f"{prompt}\n### Instruction:\n{text}\n### Response:\n"
