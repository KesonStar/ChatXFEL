# XFELBench - ChatXFEL Evaluation Framework

XFELBench is a comprehensive evaluation framework for the ChatXFEL RAG system. It allows you to systematically test different RAG configurations and measure their performance on curated question sets.

## Features

- **Configurable RAG Pipeline**: Toggle features like query rewrite, hybrid search, reranking, and routing
- **Batch Evaluation**: Process multiple questions with automatic checkpointing
- **Detailed Results**: Save answers, sources, rewritten queries, and generation times
- **Multiple Experiments**: Compare different configurations side-by-side

## Directory Structure

```
XFELBench/
├── configs/                    # Configuration files
│   ├── default.yaml           # Default baseline configuration
│   └── experiments/           # Experiment-specific configs
│       ├── baseline.yaml      # Minimal features (dense + rerank)
│       ├── hybrid_search.yaml # Hybrid search enabled
│       └── full_features.yaml # All features enabled
├── problem_sets/              # Question datasets
│   └── xfel_qa_basic.json    # Basic XFEL Q&A (10 questions)
├── outputs/                   # Evaluation results (auto-generated)
│   └── <timestamp>_<exp_name>/
│       ├── config.yaml       # Configuration used for this run
│       ├── results.jsonl     # Per-question results (JSONL format)
│       └── summary.json      # Summary statistics
├── eval_generator.py          # Answer generation script
└── README.md                  # This file
```

## Quick Start

### 1. Run a Single Evaluation

Generate answers using the baseline configuration:

```bash
cd XFELBench
python eval_generator.py \
    --config configs/experiments/baseline.yaml \
    --questions problem_sets/xfel_qa_basic.json
```

### 2. Run Multiple Experiments

Compare different configurations:

```bash
# Baseline (dense + rerank only)
python eval_generator.py \
    --config configs/experiments/baseline.yaml \
    --questions problem_sets/xfel_qa_basic.json

# Hybrid search
python eval_generator.py \
    --config configs/experiments/hybrid_search.yaml \
    --questions problem_sets/xfel_qa_basic.json

# All features
python eval_generator.py \
    --config configs/experiments/full_features.yaml \
    --questions problem_sets/xfel_qa_basic.json
```

### 3. Check Results

Results are saved in `outputs/<timestamp>_<experiment_name>/`:

```bash
# View summary statistics
cat outputs/20250101_120000_baseline/summary.json

# View detailed results (one question per line)
cat outputs/20250101_120000_baseline/results.jsonl
```

## Configuration Guide

### Configuration File Structure

A configuration file defines all aspects of the RAG pipeline:

```yaml
# Experiment metadata
experiment:
  name: "my_experiment"
  description: "Description of what this config tests"
  version: "1.0"

# Model settings
model:
  llm_name: "Qwen3-30B-Instruct"  # Ollama model name
  embedding_model: "BGE-M3"
  temperature: 0.1
  num_predict: 2048
  num_ctx: 8192

# Database connections
database:
  milvus:
    host: "10.19.48.181"
    port: 19530
    username: "cs286_2025_group8"
    password: "Group8"
    db_name: "cs286_2025_group8"

# Collection to search
collection:
  name: "xfel_bibs_collection_with_abstract"

# Year filtering
year_filter:
  enabled: true
  start_year: 2000
  end_year: 2025

# Feature toggles
features:
  query_rewrite:
    enabled: true  # Rewrite queries using LLM

  hybrid_search:
    enabled: true  # Use dense + sparse vectors
    dense_weight: 0.5
    sparse_weight: 0.5

  rerank:
    enabled: true  # Rerank with cross-encoder
    model: "BAAI/bge-reranker-v2-m3"
    top_n: 6

  routing:
    enabled: true  # Two-stage retrieval (abstract -> fulltext)
    fulltext_top_k: 6

  chat_history:
    enabled: false  # Include conversation context (for future use)

# Retrieval settings
retrieval:
  top_k: 10  # Initial retrieval count
  search_params:
    ef: 20  # HNSW parameter

# Prompt template
prompt:
  template_file: "prompts/naive.pt"

# Evaluation settings
evaluation:
  batch_size: 10  # Checkpoint frequency
  save_sources: true  # Save retrieved documents
  save_rewritten_queries: true  # Save rewritten queries
```

### Feature Toggle Combinations

Here are recommended feature combinations to test:

| Configuration | Query Rewrite | Hybrid Search | Rerank | Routing | Purpose |
|--------------|---------------|---------------|---------|---------|---------|
| Baseline | ❌ | ❌ | ✅ | ❌ | Minimal features |
| Hybrid Search | ❌ | ✅ | ✅ | ❌ | Test hybrid retrieval |
| Query Rewrite | ✅ | ❌ | ✅ | ❌ | Test query optimization |
| Routing | ❌ | ❌ | ✅ | ✅ | Test two-stage retrieval |
| Full Features | ✅ | ✅ | ✅ | ✅ | Maximum performance |

## Question Set Format

Question sets are defined in JSON format:

```json
{
  "metadata": {
    "name": "Question Set Name",
    "description": "Description",
    "version": "1.0",
    "num_questions": 10
  },
  "questions": [
    {
      "id": "q_001",
      "question": "What is XFEL?",
      "category": "basic",
      "difficulty": "easy",
      "expected_topics": ["XFEL", "free electron laser"],
      "reference_answer": null
    }
  ]
}
```

## Output Format

### results.jsonl

Each line contains a JSON object for one question:

```json
{
  "question_id": "basic_001",
  "question": "What is Serial Femtosecond Crystallography?",
  "answer": "Serial Femtosecond Crystallography (SFX) is...",
  "sources": [
    {
      "title": "Paper title",
      "doi": "10.1234/example",
      "journal": "Nature",
      "year": "2023",
      "page": "1",
      "content": "Retrieved text..."
    }
  ],
  "rewritten_query": "serial femtosecond crystallography technique XFEL",
  "generation_time": 2.34,
  "timestamp": "2025-01-01T12:00:00",
  "metadata": {
    "category": "technique",
    "difficulty": "basic",
    "expected_topics": ["SFX", "crystallography"]
  }
}
```

### summary.json

Summary statistics for the entire run:

```json
{
  "experiment_name": "baseline",
  "total_questions": 10,
  "completed_questions": 10,
  "failed_questions": 0,
  "average_generation_time": 2.45,
  "config": { ... },
  "timestamp": "2025-01-01T12:30:00"
}
```

## Next Steps: LLM-as-Judge Evaluation

After generating answers, you can evaluate them using:

1. **Automated Metrics**: BLEU, ROUGE, BERTScore (if reference answers available)
2. **LLM-as-Judge**: Use an LLM to score answer quality, relevance, and correctness
3. **Human Evaluation**: Manual review and scoring

The `eval_judge.py` script (to be implemented) will handle automated evaluation.

## Tips

### Running Multiple Experiments

Create a batch script to run all experiments:

```bash
#!/bin/bash
QUESTIONS="problem_sets/xfel_qa_basic.json"

for config in configs/experiments/*.yaml; do
    echo "Running experiment: $config"
    python eval_generator.py --config "$config" --questions "$QUESTIONS"
done
```

### Monitoring Progress

The generator uses tqdm for progress bars and prints regular status updates:

```
[INFO] Loaded configuration: baseline
[INFO] Description: Baseline with only dense search and reranking
[INFO] Loading embedding model: BGE-M3
[INFO] Loading LLM: Qwen3-30B-Instruct
[INFO] Initializing DENSE-ONLY retriever...
[INFO] Adding reranker: BAAI/bge-reranker-v2-m3 (top_n=6)
[INFO] Loaded 10 questions from problem_sets/xfel_qa_basic.json
Generating answers: 100%|██████████| 10/10 [00:23<00:00,  2.34s/it]
[INFO] Summary: 10/10 completed, 0 failed
[INFO] Average generation time: 2.45s
```

### Troubleshooting

**Issue**: Model not found
```
Solution: Check that Ollama is running and the model is available:
  ollama list
  ollama pull qwen3:30b-a3b-instruct-2507-q8_0
```

**Issue**: Milvus connection failed
```
Solution: Verify connection parameters in config and check network access
```

**Issue**: Out of memory
```
Solution: Reduce batch_size in config or use a smaller model
```

## License

Part of the ChatXFEL project.
