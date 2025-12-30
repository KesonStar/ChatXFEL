#!/bin/bash
# Evaluation runner script for XFELBench
# Usage: bash run_eval.sh <config_file> <question_file>
#
# Example:
#   bash run_eval.sh configs/experiments/baseline.yaml problem_sets/xfel_qa_basic.json

if [ "$#" -ne 2 ]; then
    echo "Usage: bash run_eval.sh <config_file> <question_file>"
    echo ""
    echo "Example:"
    echo "  bash run_eval.sh configs/experiments/baseline.yaml problem_sets/xfel_qa_basic.json"
    exit 1
fi

CONFIG_FILE=$1
QUESTION_FILE=$2

# Check if files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$QUESTION_FILE" ]; then
    echo "Error: Question file not found: $QUESTION_FILE"
    exit 1
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rag

# Run evaluation
cd "$(dirname "$0")"
python eval_generator.py --config "$CONFIG_FILE" --questions "$QUESTION_FILE"
