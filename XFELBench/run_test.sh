#!/bin/bash
# Test script for XFELBench
# Usage: bash run_test.sh

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rag

# Run tests
cd "$(dirname "$0")"
python test_generator.py
