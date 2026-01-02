#!/bin/bash
# Ablation Study Runner - Incremental Feature Testing
# Runs experiments in order: baseline -> +rerank -> +hybrid -> +routing -> +query_rewrite
#
# Usage: bash run_ablation.sh [question_file] [problem_set]
#
# Example:
#   bash run_ablation.sh problem_sets/xfel_qa_basic.json problem_sets/problem_set.md

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color


# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# XFELBench root directory (parent of bin/)
XFELBENCH_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
QUESTION_FILE=${1:-"problem_sets/xfel_benchmark.json"}
PROBLEM_SET=${2:-"problem_sets/problem_set.md"}

export OPENAI_API_KEY="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"   

# Change to XFELBench root directory
cd "$XFELBENCH_ROOT"

echo -e "${BLUE}"
echo "========================================"
echo "XFELBench Ablation Study"
echo "========================================"
echo -e "${NC}"
echo ""
echo -e "${CYAN}Incremental Feature Testing:${NC}"
echo "  1. Baseline (dense search only)"
echo "  2. +Reranking"
echo "  3. +Hybrid Search (dense + sparse)"
echo "  4. +Routing (two-stage retrieval)"
echo "  5. +Query Rewrite (all features)"
echo ""
echo -e "${GREEN}[INFO] Configuration:${NC}"
echo "  Question file: $QUESTION_FILE"
echo "  Problem set: $PROBLEM_SET"
echo "  Working directory: $(pwd)"
echo ""

# Check if question file exists
if [ ! -f "$QUESTION_FILE" ]; then
    echo -e "${RED}[ERROR] Question file not found: $QUESTION_FILE${NC}"
    exit 1
fi

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}[WARNING] OPENAI_API_KEY not set. LLM judge evaluation will fail.${NC}"
    echo "Set it with: export OPENAI_API_KEY='your-key'"
    echo ""
fi

# Ask for confirmation
read -p "Start ablation study? This will run 5 experiments + evaluations. (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}[INFO] Ablation study cancelled${NC}"
    exit 0
fi

echo ""

# Define experiment configs in order
CONFIGS=(
    "configs/experiments/1_baseline_no_rerank.yaml"
    "configs/experiments/2_baseline_with_rerank.yaml"
    "configs/experiments/3_with_hybrid_search.yaml"
    "configs/experiments/4_with_routing.yaml"
    "configs/experiments/5_full_features.yaml"
)

FEATURES=(
    "Baseline (Dense)"
    "+Rerank"
    "+Hybrid Search"
    "+Routing"
    "+Query Rewrite"
)

# Track results
TOTAL=${#CONFIGS[@]}
SUCCESS_COUNT=0
FAIL_COUNT=0

# Arrays to store output paths
declare -a OUTPUT_DIRS
declare -a EVAL_DIRS

echo -e "${BLUE}========================================"
echo "Phase 1: Generating RAG Responses"
echo "========================================${NC}"
echo ""

# Run each experiment
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    FEATURE="${FEATURES[$i]}"
    STEP=$((i + 1))

    echo -e "${CYAN}--------------------------------------"
    echo "[$STEP/$TOTAL] ${FEATURE}"
    echo "--------------------------------------${NC}"
    echo "Config: $CONFIG"
    echo ""

    if [ ! -f "$CONFIG" ]; then
        echo -e "${RED}[ERROR] Config file not found: $CONFIG${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # Run evaluation generator
    if python scripts/evaluation/eval_generator.py \
        --config "$CONFIG" \
        --questions "$QUESTION_FILE"; then

        echo -e "${GREEN}✓ Generation completed${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # Find the latest output directory for this experiment
        EXP_NAME=$(basename "$CONFIG" .yaml)
        LATEST_OUTPUT=$(ls -dt outputs/*_${EXP_NAME} 2>/dev/null | head -1)
        if [ -n "$LATEST_OUTPUT" ]; then
            OUTPUT_DIRS+=("$LATEST_OUTPUT")
            echo "  Output: $LATEST_OUTPUT"
        fi
    else
        echo -e "${RED}✗ Generation failed${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    echo ""
done

echo ""
echo -e "${BLUE}========================================"
echo "Phase 1 Summary"
echo "========================================${NC}"
echo "  Total: $TOTAL"
echo "  Success: $SUCCESS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo ""

if [ $SUCCESS_COUNT -eq 0 ]; then
    echo -e "${RED}[ERROR] All experiments failed. Exiting.${NC}"
    exit 1
fi

# Phase 2: LLM Judge Evaluation (if API key is set)
if [ -n "$OPENAI_API_KEY" ]; then
    echo -e "${BLUE}========================================"
    echo "Phase 2: LLM Judge Evaluation"
    echo "========================================${NC}"
    echo ""

    EVAL_SUCCESS=0
    EVAL_FAIL=0

    for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
        EXP_NAME=$(basename "$OUTPUT_DIR")
        RESULTS_FILE="$OUTPUT_DIR/results.jsonl"
        EVAL_OUTPUT="evaluations/$EXP_NAME"

        echo -e "${CYAN}Evaluating: $EXP_NAME${NC}"
        echo "  Input:  $RESULTS_FILE"
        echo "  Output: $EVAL_OUTPUT"
        echo ""

        if [ ! -f "$RESULTS_FILE" ]; then
            echo -e "${RED}[ERROR] Results file not found: $RESULTS_FILE${NC}"
            EVAL_FAIL=$((EVAL_FAIL + 1))
            continue
        fi

        if python scripts/evaluation/llm_judge.py \
            --results "$RESULTS_FILE" \
            --output "$EVAL_OUTPUT" \
            --problem-set "$PROBLEM_SET"; then

            echo -e "${GREEN}✓ Evaluation completed${NC}"
            EVAL_SUCCESS=$((EVAL_SUCCESS + 1))
            EVAL_DIRS+=("$EVAL_OUTPUT")

            # Show quick summary
            SUMMARY_FILE="$EVAL_OUTPUT/evaluation_summary.json"
            if [ -f "$SUMMARY_FILE" ]; then
                OVERALL_SCORE=$(python -c "import json; print(f\"{json.load(open('$SUMMARY_FILE'))['average_scores']['overall']:.2f}\")" 2>/dev/null || echo "N/A")
                echo "  Overall Score: $OVERALL_SCORE/5"
            fi
        else
            echo -e "${RED}✗ Evaluation failed${NC}"
            EVAL_FAIL=$((EVAL_FAIL + 1))
        fi

        echo ""
    done

    echo -e "${BLUE}========================================"
    echo "Phase 2 Summary"
    echo "========================================${NC}"
    echo "  Total: ${#OUTPUT_DIRS[@]}"
    echo "  Success: $EVAL_SUCCESS"
    echo "  Failed: $EVAL_FAIL"
    echo ""
else
    echo -e "${YELLOW}[INFO] Skipping LLM judge evaluation (OPENAI_API_KEY not set)${NC}"
    echo ""
fi

# Final Summary
echo -e "${GREEN}"
echo "========================================"
echo "Ablation Study Complete!"
echo "========================================${NC}"
echo ""
echo "Results saved in:"
echo "  - outputs/     (RAG responses)"
if [ ${#EVAL_DIRS[@]} -gt 0 ]; then
    echo "  - evaluations/ (LLM judge scores)"
fi
echo ""

# Show all experiment results
if [ ${#OUTPUT_DIRS[@]} -gt 0 ]; then
    echo "Generated experiments:"
    for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
        echo "  - $OUTPUT_DIR"
    done
    echo ""
fi

# Suggest next steps
if [ ${#EVAL_DIRS[@]} -gt 1 ]; then
    echo "Next steps:"
    echo "  1. Compare results across experiments:"
    echo "     python scripts/evaluation/compare_results.py"
    echo ""
    echo "  2. View individual summaries:"
    for EVAL_DIR in "${EVAL_DIRS[@]}"; do
        if [ -f "$EVAL_DIR/evaluation_summary.json" ]; then
            echo "     cat $EVAL_DIR/evaluation_summary.json"
        fi
    done
fi
