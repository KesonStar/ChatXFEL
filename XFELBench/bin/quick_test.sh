#!/bin/bash
# Quick Test Script - Test the pipeline with minimal configs
# Useful for debugging and verifying the setup

set -e

echo "========================================"
echo "XFELBench Pipeline Quick Test"
echo "========================================"
echo ""

# Check prerequisites
echo "[INFO] Checking prerequisites..."

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "[WARNING] OPENAI_API_KEY not set. LLM judge will be skipped."
    SKIP_LLM_JUDGE="--skip-llm-judge"
else
    echo "[OK] OpenAI API key found"
    SKIP_LLM_JUDGE=""
fi

# Check if question file exists
QUESTION_FILE="problem_sets/xfel_qa_basic.json"
if [ ! -f "$QUESTION_FILE" ]; then
    echo "[ERROR] Question file not found: $QUESTION_FILE"
    exit 1
fi
echo "[OK] Question file found"

echo ""
echo "========================================"
echo "Step 1: Generate Test Configs"
echo "========================================"

# Generate only baseline and hybrid_search for quick test
python scripts/generation/generate_configs.py --configs baseline hybrid_search

echo ""
echo "========================================"
echo "Step 2: Run Evaluation (2 configs)"
echo "========================================"

python scripts/orchestration/run_full_evaluation.py \
    --questions "$QUESTION_FILE" \
    --configs baseline hybrid_search \
    $SKIP_LLM_JUDGE

echo ""
echo "========================================"
echo "Step 3: Compare Results"
echo "========================================"

if [ -z "$SKIP_LLM_JUDGE" ]; then
    python scripts/evaluation/compare_results.py --compare baseline hybrid_search
else
    echo "[SKIPPED] LLM judge was skipped, no scores to compare"
    echo "To test with LLM judge, set OPENAI_API_KEY and run again"
fi

echo ""
echo "========================================"
echo "Quick Test Complete!"
echo "========================================"
echo ""
echo "If this test passed, you can run the full evaluation:"
echo "  ./run_all.sh"
