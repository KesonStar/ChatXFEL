# XFELBench Ablation Study Guide

## Overview

This guide explains how to run an incremental ablation study to evaluate the impact of each feature in the ChatXFEL RAG system.

## Experiment Sequence

The ablation study runs **5 experiments** in the following order:

| Step | Name | Features Enabled | Description |
|------|------|------------------|-------------|
| 1 | `1_baseline_no_rerank` | Dense search only | Pure baseline with no advanced features |
| 2 | `2_baseline_with_rerank` | Dense + **Rerank** | Adds cross-encoder reranking |
| 3 | `3_with_hybrid_search` | Dense + Rerank + **Hybrid** | Adds sparse vector search (BM25-like) |
| 4 | `4_with_routing` | Dense + Rerank + Hybrid + **Routing** | Adds two-stage retrieval (abstract→fulltext) |
| 5 | `5_full_features` | All above + **Query Rewrite** | All features enabled |

## Quick Start

### 1. Set API Key (for LLM Judge evaluation)

```bash
export OPENAI_API_KEY="your-openrouter-api-key"
```

### 2. Run Full Ablation Study

```bash
cd XFELBench
bash bin/run_ablation.sh
```

This will:
- Generate RAG responses for all 5 experiments
- Run LLM judge evaluation for each experiment
- Save results to `outputs/` and `evaluations/`

### 3. Compare Results

```bash
python scripts/evaluation/compare_results.py
```

## Step-by-Step Execution

If you prefer to run experiments manually:

### Generate RAG Responses

```bash
cd XFELBench

# Step 1: Baseline (no rerank)
python scripts/evaluation/eval_generator.py \
  --config configs/experiments/1_baseline_no_rerank.yaml \
  --questions problem_sets/xfel_qa_basic.json

# Step 2: +Rerank
python scripts/evaluation/eval_generator.py \
  --config configs/experiments/2_baseline_with_rerank.yaml \
  --questions problem_sets/xfel_qa_basic.json

# Step 3: +Hybrid Search
python scripts/evaluation/eval_generator.py \
  --config configs/experiments/3_with_hybrid_search.yaml \
  --questions problem_sets/xfel_qa_basic.json

# Step 4: +Routing
python scripts/evaluation/eval_generator.py \
  --config configs/experiments/4_with_routing.yaml \
  --questions problem_sets/xfel_qa_basic.json

# Step 5: +Query Rewrite (Full)
python scripts/evaluation/eval_generator.py \
  --config configs/experiments/5_full_features.yaml \
  --questions problem_sets/xfel_qa_basic.json
```

### Run LLM Judge Evaluation

```bash
# Evaluate all experiments at once
bash bin/run_judge_all.sh

# Or evaluate individually
bash bin/run_judge.sh \
  outputs/YYYYMMDD_HHMMSS_1_baseline_no_rerank/results.jsonl \
  evaluations/YYYYMMDD_HHMMSS_1_baseline_no_rerank
```

## Understanding Results

### Output Structure

```
XFELBench/
├── outputs/
│   ├── 20260102_180000_1_baseline_no_rerank/
│   │   ├── config.yaml           # Experiment configuration
│   │   ├── results.jsonl         # Generated responses
│   │   └── summary.json          # Generation statistics
│   ├── 20260102_180500_2_baseline_with_rerank/
│   └── ...
└── evaluations/
    ├── 20260102_180000_1_baseline_no_rerank/
    │   ├── evaluation_results.jsonl   # Detailed scores per question
    │   └── evaluation_summary.json    # Average scores
    └── ...
```

### Evaluation Metrics

Each experiment is scored on three dimensions (1-5 scale):

1. **Factual Accuracy**: Are the facts correct?
2. **Groundedness**: Is the answer supported by retrieved sources?
3. **Coverage & Specificity**: Does it fully address the question with sufficient detail?

**Overall Score**: Average of the three dimensions

### View Summary

```bash
# View evaluation summary for one experiment
cat evaluations/YYYYMMDD_HHMMSS_1_baseline_no_rerank/evaluation_summary.json

# Quick view with formatting
python -c "
import json
with open('evaluations/YYYYMMDD_HHMMSS_1_baseline_no_rerank/evaluation_summary.json') as f:
    data = json.load(f)
    print(f\"Total evaluated: {data['total_evaluated']}\")
    scores = data['average_scores']
    print(f\"Factual Accuracy: {scores['factual_accuracy']:.2f}/5\")
    print(f\"Groundedness: {scores['groundedness']:.2f}/5\")
    print(f\"Coverage: {scores['coverage_specificity']:.2f}/5\")
    print(f\"Overall: {scores['overall']:.2f}/5\")
"
```

## Expected Insights

The ablation study helps answer:

1. **Does reranking improve results?** (Compare Step 1 vs Step 2)
2. **Does hybrid search (dense + sparse) help?** (Compare Step 2 vs Step 3)
3. **Does two-stage routing improve context quality?** (Compare Step 3 vs Step 4)
4. **Does query rewriting enhance retrieval?** (Compare Step 4 vs Step 5)

## Troubleshooting

### Error: "OPENAI_API_KEY environment variable not set"

```bash
export OPENAI_API_KEY="your-openrouter-api-key"
```

Get an API key from: https://openrouter.ai/

### Error: "Question file not found"

Make sure you're running from the XFELBench directory:

```bash
cd XFELBench
bash bin/run_ablation.sh
```

### Conda environment activation fails

The script will use your current Python environment if conda is not available. Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

## Advanced Usage

### Use Different Question Set

```bash
bash bin/run_ablation.sh custom_questions.json
```

### Skip Specific Steps

Edit `bin/run_ablation.sh` and comment out unwanted configs in the `CONFIGS` array.

### Modify Feature Parameters

Edit the config files in `configs/experiments/`:
- Adjust reranking `top_n`
- Change hybrid search weights (`dense_weight`, `sparse_weight`)
- Modify routing `fulltext_top_k`

## Next Steps

After running the ablation study:

1. **Generate comparison report**:
   ```bash
   python scripts/evaluation/compare_results.py
   ```

2. **Analyze specific questions**:
   - Check `evaluation_results.jsonl` for per-question scores
   - Identify which questions benefit most from each feature

3. **Iterate on configurations**:
   - Fine-tune parameters based on results
   - Create new configs to test alternative settings
