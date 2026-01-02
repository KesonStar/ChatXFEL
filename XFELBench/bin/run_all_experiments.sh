#!/bin/bash
# Run all experiment configurations on a question set
# Usage: bash run_all_experiments.sh [question_file]
#
# Example:
#   bash run_all_experiments.sh problem_sets/xfel_qa_basic.json

QUESTION_FILE=${1:-"problem_sets/xfel_qa_basic.json"}

if [ ! -f "$QUESTION_FILE" ]; then
    echo "Error: Question file not found: $QUESTION_FILE"
    exit 1
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rag

cd "$(dirname "$0")"

echo "======================================"
echo "Running all experiments"
echo "Question file: $QUESTION_FILE"
echo "======================================"
echo ""

# Run each experiment configuration
for config in configs/experiments/*.yaml; do
    echo "--------------------------------------"
    echo "Running: $config"
    echo "--------------------------------------"
    python eval_generator.py --config "$config" --questions "$QUESTION_FILE"

    if [ $? -eq 0 ]; then
        echo "✓ Completed: $config"
    else
        echo "✗ Failed: $config"
    fi
    echo ""
done

echo "======================================"
echo "All experiments completed!"
echo "Check outputs/ directory for results"
echo "======================================"
