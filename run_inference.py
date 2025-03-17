import os
import subprocess
import glob

# Configuration
MODEL_NAME = "/network/weights/llama.var/llama_2/Llama-2-13b-hf"  # Replace with the actual model
# DEBIAS_PROMPT = "debias_prompts"
DATA_DIR = "data/jsonl"

# Get all .jsonl files
jsonl_files = glob.glob(os.path.join(DATA_DIR, "*.jsonl"))

# Run inference on each file
for file in jsonl_files:
    print(f"Running inference on {file}...")
    command = [
        "python3", "src/pred.py",
        "--model", MODEL_NAME,
        "--file", file,
        # "--debias_prompt", DEBIAS_PROMPT  # Remove this line if not needed
    ]
    
    subprocess.run(command)