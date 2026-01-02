#!/bin/bash
# Batch LLM Judge evaluation script for all results in outputs/
# Usage: bash run_judge_all.sh [problem_set]
#
# Example:
#   bash run_judge_all.sh
#   bash run_judge_all.sh problem_sets/problem_set.md

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# XFELBench root directory (parent of bin/)
XFELBENCH_ROOT="$(dirname "$SCRIPT_DIR")"

PROBLEM_SET=${1:-"problem_sets/problem_set.md"}

# Change to XFELBench root directory
cd "$XFELBENCH_ROOT"

# Resolve problem set path
if [[ ! "$PROBLEM_SET" = /* ]]; then
    PROBLEM_SET="$XFELBENCH_ROOT/$PROBLEM_SET"
fi

echo "========================================"
echo "Batch LLM Judge Evaluation"
echo "========================================"
echo ""
echo "[INFO] Working directory: $(pwd)"
echo "[INFO] Problem set: $PROBLEM_SET"
echo ""


export OPENAI_API_KEY="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"

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "[ERROR] OPENAI_API_KEY environment variable not set"
    echo ""
    echo "Please set it with:"
    echo "  export OPENAI_API_KEY='your-openrouter-api-key'"
    exit 1
fi

# Count results files
RESULTS_COUNT=$(find outputs -name "results.jsonl" 2>/dev/null | wc -l | tr -d ' ')

if [ "$RESULTS_COUNT" -eq 0 ]; then
    echo "[ERROR] No results.jsonl files found in outputs/ directory"
    echo ""
    echo "Please run eval_generator first to generate results"
    exit 1
fi

echo "[INFO] Found $RESULTS_COUNT result file(s) to evaluate"
echo ""

# Ask for confirmation
read -p "Proceed with evaluating all $RESULTS_COUNT result files? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "[INFO] Evaluation cancelled"
    exit 0
fi

echo ""

# Counter for progress
COUNTER=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# Process each results.jsonl file
for RESULTS_FILE in outputs/*/results.jsonl; do
    COUNTER=$((COUNTER + 1))

    # Extract experiment name from path (e.g., outputs/20260102_174732_baseline/)
    EXP_DIR=$(dirname "$RESULTS_FILE")
    EXP_NAME=$(basename "$EXP_DIR")

    # Output directory for this evaluation
    OUTPUT_DIR="evaluations/$EXP_NAME"

    echo "--------------------------------------"
    echo "[$COUNTER/$RESULTS_COUNT] Evaluating: $EXP_NAME"
    echo "--------------------------------------"
    echo "  Input:  $RESULTS_FILE"
    echo "  Output: $OUTPUT_DIR"
    echo ""

    # Run evaluation
    if python scripts/evaluation/llm_judge.py \
        --results "$RESULTS_FILE" \
        --output "$OUTPUT_DIR" \
        --problem-set "$PROBLEM_SET"; then

        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo ""
        echo "✓ Completed: $EXP_NAME"

        # Show quick summary
        SUMMARY_FILE="$OUTPUT_DIR/evaluation_summary.json"
        if [ -f "$SUMMARY_FILE" ]; then
            OVERALL_SCORE=$(python -c "import json; print(f\"{json.load(open('$SUMMARY_FILE'))['average_scores']['overall']:.2f}\")")
            echo "  Overall Score: $OVERALL_SCORE/5"
        fi
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        echo "✗ Failed: $EXP_NAME"
    fi

    echo ""
done

echo "========================================"
echo "Batch Evaluation Complete"
echo "========================================"
echo ""
echo "Summary:"
echo "  Total: $RESULTS_COUNT"
echo "  Success: $SUCCESS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo ""
echo "Results saved in evaluations/ directory"
echo ""

# Offer to generate comparison report
if [ $SUCCESS_COUNT -gt 1 ]; then
    echo "To compare results across experiments, run:"
    echo "  python scripts/evaluation/compare_results.py"
fi
