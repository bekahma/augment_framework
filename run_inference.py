import os
import subprocess
import glob

# Configuration
MODEL_NAME = "facebook/opt-1.3b" # change as required
# DEBIAS_PROMPT = "debias_prompts"  # Remove if not needed
DATA_DIRS = ["data", "data/jsonl"]  # Search both directories

# Collect all .jsonl files
jsonl_files = []
for directory in DATA_DIRS:
    jsonl_files.extend(glob.glob(os.path.join(directory, "*.jsonl")))

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
