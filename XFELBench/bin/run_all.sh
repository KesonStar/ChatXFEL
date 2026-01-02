#!/bin/bash
# One-Click Evaluation Script for XFELBench
# Generates all configs, runs RAG evaluation, and LLM judge evaluation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "========================================"
echo "XFELBench Full Evaluation Pipeline"
echo "========================================"
echo -e "${NC}"

# Configuration
QUESTION_FILE="${1:-problem_sets/xfel_qa_basic.json}"
PROBLEM_SET="${2:-problem_sets/problem_set.md}"
CONFIGS="${3:-all}"  # Default: all configs

export OPENAI_API_KEY="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"

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}[ERROR] OPENAI_API_KEY environment variable not set${NC}"
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Check if question file exists
if [ ! -f "$QUESTION_FILE" ]; then
    echo -e "${RED}[ERROR] Question file not found: $QUESTION_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}[INFO] Configuration:${NC}"
echo "  Question file: $QUESTION_FILE"
echo "  Problem set: $PROBLEM_SET"
echo "  Configs: $CONFIGS"
echo "  API Key: ${OPENAI_API_KEY:0:8}..."
echo ""

# Ask for confirmation
read -p "Start full evaluation pipeline? This may take a while. (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}[INFO] Evaluation cancelled${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}Starting evaluation pipeline...${NC}"
echo ""

# Run the pipeline
if [ "$CONFIGS" = "all" ]; then
    # Run all configs
    python scripts/orchestration/run_full_evaluation.py \
        --questions "$QUESTION_FILE" \
        --problem-set "$PROBLEM_SET"
else
    # Run specific configs
    python scripts/orchestration/run_full_evaluation.py \
        --questions "$QUESTION_FILE" \
        --problem-set "$PROBLEM_SET" \
        --configs $CONFIGS
fi

# Check if pipeline succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}"
    echo "========================================"
    echo "Pipeline Completed Successfully!"
    echo "========================================"
    echo -e "${NC}"
    echo ""
    echo "Results are saved in:"
    echo "  - outputs/     (RAG responses)"
    echo "  - evaluations/ (LLM judge scores)"
    echo ""
    echo "View the summary report:"
    LATEST_SUMMARY=$(ls -t evaluations/summary_*/EVALUATION_REPORT.md 2>/dev/null | head -1)
    if [ -n "$LATEST_SUMMARY" ]; then
        echo "  cat $LATEST_SUMMARY"
    fi
else
    echo ""
    echo -e "${RED}"
    echo "========================================"
    echo "Pipeline Failed!"
    echo "========================================"
    echo -e "${NC}"
    echo "Please check the error messages above"
    exit 1
fi
