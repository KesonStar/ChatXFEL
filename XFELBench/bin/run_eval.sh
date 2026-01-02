#!/bin/bash
# Evaluation runner script for XFELBench
# Usage: bash run_eval.sh <config_file> <question_file>
#
# Example:
#   bash run_eval.sh configs/experiments/baseline.yaml problem_sets/xfel_qa_basic.json

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# XFELBench root directory (parent of bin/)
XFELBENCH_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: bash run_eval.sh <config_file> <question_file>"
    echo ""
    echo "Example:"
    echo "  bash run_eval.sh configs/experiments/baseline.yaml problem_sets/xfel_qa_basic.json"
    echo ""
    echo "Note: Paths should be relative to XFELBench/ directory or absolute paths"
    exit 1
fi

CONFIG_FILE=$1
QUESTION_FILE=$2

# Change to XFELBench root directory
cd "$XFELBENCH_ROOT"
echo "[INFO] Working directory: $(pwd)"

# Resolve relative paths
if [[ ! "$CONFIG_FILE" = /* ]]; then
    CONFIG_FILE="$XFELBENCH_ROOT/$CONFIG_FILE"
fi

if [[ ! "$QUESTION_FILE" = /* ]]; then
    QUESTION_FILE="$XFELBENCH_ROOT/$QUESTION_FILE"
fi

# Check if files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$QUESTION_FILE" ]; then
    echo "[ERROR] Question file not found: $QUESTION_FILE"
    exit 1
fi

echo "[INFO] Config file: $CONFIG_FILE"
echo "[INFO] Question file: $QUESTION_FILE"
echo ""

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "[INFO] Activating conda environment 'rag'..."
    eval "$(conda shell.bash hook)"
    if conda activate rag 2>/dev/null; then
        echo "[INFO] Conda environment 'rag' activated"
    else
        echo "[WARNING] Failed to activate conda environment 'rag', using current environment"
    fi
else
    echo "[INFO] Conda not found, using current Python environment"
fi

echo ""
echo "========================================"
echo "Running RAG Evaluation"
echo "========================================"
echo ""

# Run evaluation with correct path
python scripts/evaluation/eval_generator.py \
    --config "$CONFIG_FILE" \
    --questions "$QUESTION_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Evaluation completed successfully!"
    echo "========================================"
    echo ""
    echo "Results saved in outputs/ directory"
else
    echo ""
    echo "========================================"
    echo "✗ Evaluation failed!"
    echo "========================================"
    exit 1
fi
