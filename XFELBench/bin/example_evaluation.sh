#!/bin/bash
# Example script for running LLM-as-Judge evaluation on XFELBench results

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Configuration
RESULTS_FILE="outputs/20251230_230056_baseline/results.jsonl"
OUTPUT_DIR="evaluations/example_$(date +%Y%m%d_%H%M%S)"
PROBLEM_SET="problem_sets/problem_set.md"
MODEL="gpt-4o-mini"

# Check if results file exists
if [ ! -f "$RESULTS_FILE" ]; then
    echo "[ERROR] Results file not found: $RESULTS_FILE"
    echo "Please run eval_generator.py first to generate results."
    exit 1
fi

# Check if OpenAI API key is set
if [ "$OPENAI_API_KEY" = "your-api-key-here" ]; then
    echo "[ERROR] Please set your OpenAI API key in this script or as environment variable"
    exit 1
fi

echo "=========================================="
echo "XFELBench LLM-as-Judge Evaluation"
echo "=========================================="
echo "Results file: $RESULTS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Problem set: $PROBLEM_SET"
echo "Model: $MODEL"
echo "=========================================="

# Run evaluation
python scripts/evaluation/llm_judge.py \
    --results "$RESULTS_FILE" \
    --output "$OUTPUT_DIR" \
    --problem-set "$PROBLEM_SET" \
    --model "$MODEL"

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=========================================="
    echo ""
    echo "View summary:"
    echo "  cat $OUTPUT_DIR/evaluation_summary.json | python -m json.tool"
    echo ""
    echo "View detailed results:"
    echo "  head -n 1 $OUTPUT_DIR/evaluation_results.jsonl | python -m json.tool"
else
    echo ""
    echo "[ERROR] Evaluation failed. Please check the error messages above."
    exit 1
fi
