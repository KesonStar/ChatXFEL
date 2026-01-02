#!/bin/bash
# Setup Verification Script
# Verifies that all paths and imports are working correctly after reorganization

set -e

echo "=========================================="
echo "XFELBench Setup Verification"
echo "=========================================="
echo ""

PASSED=0
FAILED=0

# Test 1: Config generator list
echo "[TEST 1] Testing config generator --list..."
if python scripts/generation/generate_configs.py --list > /dev/null 2>&1; then
    echo "  ✅ PASSED"
    ((PASSED++))
else
    echo "  ❌ FAILED"
    ((FAILED++))
fi

# Test 2: Config generator create
echo "[TEST 2] Testing config generation..."
if python scripts/generation/generate_configs.py --configs baseline --output-dir configs/test_generated > /dev/null 2>&1; then
    echo "  ✅ PASSED"
    ((PASSED++))
    rm -rf configs/test_generated
else
    echo "  ❌ FAILED"
    ((FAILED++))
fi

# Test 3: Full evaluation pipeline --list-configs
echo "[TEST 3] Testing run_full_evaluation --list-configs..."
if python scripts/orchestration/run_full_evaluation.py --list-configs > /dev/null 2>&1; then
    echo "  ✅ PASSED"
    ((PASSED++))
else
    echo "  ❌ FAILED"
    ((FAILED++))
fi

# Test 4: Check if question files exist
echo "[TEST 4] Checking question files..."
if [ -f "problem_sets/xfel_qa_basic.json" ]; then
    echo "  ✅ PASSED"
    ((PASSED++))
else
    echo "  ❌ FAILED - problem_sets/xfel_qa_basic.json not found"
    ((FAILED++))
fi

# Test 5: Check if problem set exists
echo "[TEST 5] Checking problem set..."
if [ -f "problem_sets/problem_set.md" ]; then
    echo "  ✅ PASSED"
    ((PASSED++))
else
    echo "  ❌ FAILED - problem_sets/problem_set.md not found"
    ((FAILED++))
fi

# Test 6: Check directory structure
echo "[TEST 6] Checking directory structure..."
if [ -d "bin" ] && [ -d "scripts/evaluation" ] && [ -d "scripts/generation" ] && [ -d "scripts/orchestration" ] && [ -d "docs" ]; then
    echo "  ✅ PASSED"
    ((PASSED++))
else
    echo "  ❌ FAILED - Missing directories"
    ((FAILED++))
fi

# Test 7: Check executable permissions
echo "[TEST 7] Checking executable permissions on shell scripts..."
if [ -x "bin/run_all.sh" ] && [ -x "bin/quick_test.sh" ]; then
    echo "  ✅ PASSED"
    ((PASSED++))
else
    echo "  ❌ FAILED - Shell scripts not executable"
    ((FAILED++))
fi

# Test 8: Check Python __init__.py files
echo "[TEST 8] Checking Python package structure..."
if [ -f "scripts/__init__.py" ] && [ -f "scripts/evaluation/__init__.py" ] && [ -f "scripts/generation/__init__.py" ] && [ -f "scripts/orchestration/__init__.py" ]; then
    echo "  ✅ PASSED"
    ((PASSED++))
else
    echo "  ❌ FAILED - Missing __init__.py files"
    ((FAILED++))
fi

# Test 9: Check import paths (eval_generator can find ChatXFEL modules)
echo "[TEST 9] Checking import paths to ChatXFEL..."
if python -c "from pathlib import Path; import sys; CHATXFEL_ROOT = Path('scripts/evaluation/eval_generator.py').parent.parent.parent.parent; print(CHATXFEL_ROOT.exists())" 2>/dev/null | grep -q "True"; then
    echo "  ✅ PASSED"
    ((PASSED++))
else
    echo "  ❌ FAILED - Cannot locate ChatXFEL root"
    ((FAILED++))
fi

# Test 10: Check prompt template path resolution
echo "[TEST 10] Checking prompt template accessibility..."
if [ -f "../prompts/naive.pt" ]; then
    echo "  ✅ PASSED"
    ((PASSED++))
else
    echo "  ❌ FAILED - prompts/naive.pt not found in ChatXFEL root"
    ((FAILED++))
fi

echo ""
echo "=========================================="
echo "Verification Results"
echo "=========================================="
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "✅ All tests passed! Setup is correct."
    echo ""
    echo "You can now run:"
    echo "  ./bin/run_all.sh           # Full evaluation"
    echo "  ./bin/quick_test.sh        # Quick test"
    exit 0
else
    echo ""
    echo "❌ Some tests failed. Please check the errors above."
    exit 1
fi
