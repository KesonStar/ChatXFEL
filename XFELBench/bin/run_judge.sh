#!/bin/bash
# LLM Judge evaluation script for XFELBench
# Usage: bash run_judge.sh <results_file> <output_dir> [problem_set]
#
# Example:
#   bash run_judge.sh outputs/20260102_174732_baseline/results.jsonl evaluations/20260102_174732_baseline

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# XFELBench root directory (parent of bin/)
XFELBENCH_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: bash run_judge.sh <results_file> <output_dir> [problem_set]"
    echo ""
    echo "Arguments:"
    echo "  results_file  - Path to results.jsonl from eval_generator"
    echo "  output_dir    - Directory to save evaluation results"
    echo "  problem_set   - (Optional) Path to problem_set.md with ground truth"
    echo "                  Default: problem_sets/problem_set.md"
    echo ""
    echo "Example:"
    echo "  bash run_judge.sh outputs/20260102_174732_baseline/results.jsonl evaluations/20260102_174732_baseline"
    echo ""
    echo "Note: Paths should be relative to XFELBench/ directory or absolute paths"
    exit 1
fi

RESULTS_FILE=$1
OUTPUT_DIR=$2
PROBLEM_SET=${3:-"problem_sets/problem_set.md"}

# Change to XFELBench root directory
cd "$XFELBENCH_ROOT"
echo "[INFO] Working directory: $(pwd)"

# Resolve relative paths
if [[ ! "$RESULTS_FILE" = /* ]]; then
    RESULTS_FILE="$XFELBENCH_ROOT/$RESULTS_FILE"
fi

if [[ ! "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR="$XFELBENCH_ROOT/$OUTPUT_DIR"
fi

if [[ ! "$PROBLEM_SET" = /* ]]; then
    PROBLEM_SET="$XFELBENCH_ROOT/$PROBLEM_SET"
fi

# Check if results file exists
if [ ! -f "$RESULTS_FILE" ]; then
    echo "[ERROR] Results file not found: $RESULTS_FILE"
    exit 1
fi

# Check if problem set exists (warning only)
if [ ! -f "$PROBLEM_SET" ]; then
    echo "[WARNING] Problem set file not found: $PROBLEM_SET"
    echo "[WARNING] Evaluation will proceed without ground truth answers"
fi

echo "[INFO] Results file: $RESULTS_FILE"
echo "[INFO] Output directory: $OUTPUT_DIR"
echo "[INFO] Problem set: $PROBLEM_SET"
echo ""

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "[ERROR] OPENAI_API_KEY environment variable not set"
    echo ""
    echo "Please set it with:"
    echo "  export OPENAI_API_KEY='your-openrouter-api-key'"
    echo ""
    echo "You can get an API key from: https://openrouter.ai/"
    exit 1
fi

echo "[INFO] API Key: ${OPENAI_API_KEY:0:10}..."
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
echo "Running LLM Judge Evaluation"
echo "========================================"
echo ""

# Run LLM judge evaluation
python scripts/evaluation/llm_judge.py \
    --results "$RESULTS_FILE" \
    --output "$OUTPUT_DIR" \
    --problem-set "$PROBLEM_SET"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Evaluation completed successfully!"
    echo "========================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""

    # Show summary if available
    SUMMARY_FILE="$OUTPUT_DIR/evaluation_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary:"
        python -c "
import json
with open('$SUMMARY_FILE', 'r') as f:
    data = json.load(f)
    print(f\"  Total evaluated: {data['total_evaluated']}\")
    scores = data['average_scores']
    print(f\"  Factual Accuracy: {scores['factual_accuracy']:.2f}/5\")
    print(f\"  Groundedness: {scores['groundedness']:.2f}/5\")
    print(f\"  Coverage & Specificity: {scores['coverage_specificity']:.2f}/5\")
    print(f\"  Overall: {scores['overall']:.2f}/5\")
"
    fi
else
    echo ""
    echo "========================================"
    echo "✗ Evaluation failed!"
    echo "========================================"
    exit 1
fi
