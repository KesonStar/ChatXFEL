# XFELBench Quick Start Guide

## Prerequisites

1. **Activate conda environment**:
   ```bash
   conda activate rag
   ```

2. **Ensure services are running**:
   - Ollama server (http://10.15.102.186:9000)
   - Milvus server (10.19.48.181:19530)

## Running Tests

Test the evaluation framework (no service dependencies):

```bash
cd XFELBench
bash run_test.sh
```

## Running a Single Evaluation

Evaluate with baseline configuration:

```bash
bash run_eval.sh \
    configs/experiments/baseline.yaml \
    problem_sets/xfel_qa_basic.json
```

Or manually with conda:

```bash
conda activate rag
python eval_generator.py \
    --config configs/experiments/baseline.yaml \
    --questions problem_sets/xfel_qa_basic.json
```

## Running All Experiments

Compare all configurations:

```bash
bash run_all_experiments.sh problem_sets/xfel_qa_basic.json
```

This will run:
1. **baseline**: Dense search + reranking only
2. **hybrid_search**: Dense + sparse hybrid search
3. **full_features**: All features enabled (query rewrite, hybrid, rerank, routing)

## Checking Results

Results are saved in `outputs/<timestamp>_<experiment_name>/`:

```bash
# List all evaluation runs
ls -lt outputs/

# View summary of latest run
cat outputs/$(ls -t outputs/ | head -1)/summary.json | python -m json.tool

# Count completed questions
cat outputs/$(ls -t outputs/ | head -1)/results.jsonl | wc -l

# View first result
cat outputs/$(ls -t outputs/ | head -1)/results.jsonl | head -1 | python -m json.tool
```

## Example Workflow

```bash
# 1. Activate environment
conda activate rag

# 2. Test the framework
cd XFELBench
bash run_test.sh

# 3. Run baseline evaluation
bash run_eval.sh \
    configs/experiments/baseline.yaml \
    problem_sets/xfel_qa_basic.json

# 4. Check results
ls -l outputs/

# 5. Run all experiments for comparison
bash run_all_experiments.sh problem_sets/xfel_qa_basic.json
```

## Creating Custom Configurations

Copy an existing config and modify:

```bash
cp configs/experiments/baseline.yaml configs/experiments/my_experiment.yaml
# Edit my_experiment.yaml
# Change experiment name and toggle features
```

Then run:

```bash
bash run_eval.sh \
    configs/experiments/my_experiment.yaml \
    problem_sets/xfel_qa_basic.json
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'langchain_community'`
```bash
Solution: Make sure you activated the rag environment:
  conda activate rag
```

**Issue**: `Connection refused` to Milvus/Ollama
```bash
Solution: Check service status and network connectivity
  # Test Ollama
  curl http://10.15.102.186:9000/api/tags

  # Check Milvus connection in your config file
```

**Issue**: Script permission denied
```bash
Solution: Make scripts executable
  chmod +x XFELBench/*.sh
```

## Next Steps

After generating answers, you can:

1. **Analyze results** using the generated JSONL files
2. **Implement LLM-as-Judge** evaluation (eval_judge.py - to be created)
3. **Compare configurations** by analyzing summary.json files
4. **Add more questions** to problem_sets/

For more details, see [README.md](README.md).
