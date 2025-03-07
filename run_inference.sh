#!/bin/bash
#SBATCH --job-name=model_inference      # Job name
#SBATCH --output=logs/inference_%j.out  # Output log file
#SBATCH --error=logs/inference_%j.err   # Error log file
#SBATCH --time=02:00:00                 # Time limit (adjust as needed)
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --mem=16G                        # Memory per node
#SBATCH --cpus-per-task=4                # Number of CPU cores
#SBATCH --partition=a40                   # Change based on your cluster

# Check if the model name and results directory are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch run_inference.sh <MODEL_NAME> <OUTPUT_DIR>"
    exit 1
fi

MODEL_NAME="$1"
OUTPUT_DIR="$2"
DATA_DIR="data/jsonl"

# Set Python path
# export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export PYTHONPATH="$HOME/llm_socialbias_prompts/src:$PYTHONPATH"

# Load Hugging Face credentials automatically
export HF_HOME="$HOME/.cache/huggingface"

echo "Running inference for model: $MODEL_NAME"
echo "Results will be saved in: $OUTPUT_DIR"

# Iterate over all JSONL files and run inference
for file in "$DATA_DIR"/*.jsonl; do
    echo "Processing $file with $MODEL_NAME..."
    python3 src/pred.py --model "$MODEL_NAME" --file "$file" --output_dir "$OUTPUT_DIR"
done

echo "Finished processing all JSONL files with model $MODEL_NAME."
