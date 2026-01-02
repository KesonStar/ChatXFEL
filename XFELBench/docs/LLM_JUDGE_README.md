# LLM-as-Judge Evaluator for XFELBench

This module provides automated evaluation of RAG responses using GPT-4o-mini as a judge across three dimensions:

1. **Factual Accuracy** (1-5): Evaluates correctness of facts and concepts
2. **Groundedness / Evidence Use** (1-5): Assesses how well responses are supported by source documents
3. **Coverage & Specificity** (1-5): Measures comprehensiveness and detail level

## Features

- **Conditional Ground Truth Handling**: Automatically uses different evaluation prompts depending on whether ground truth answers are available
- **Multi-dimensional Scoring**: Separate prompts and scores for each evaluation dimension
- **Batch Processing**: Evaluates entire result sets with progress tracking
- **Detailed Feedback**: Provides reasoning, identified errors, and missing aspects for each dimension
- **Summary Statistics**: Generates aggregate scores across all questions

## Installation

```bash
# Install OpenAI Python library
pip install openai

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

Evaluate a results file from `eval_generator.py`:

```bash
python llm_judge.py \
  --results outputs/20251230_230056_baseline/results.jsonl \
  --output evaluations/baseline_evaluation \
  --problem-set problem_sets/problem_set.md
```

### Advanced Options

```bash
python llm_judge.py \
  --results outputs/YOUR_RUN/results.jsonl \
  --output evaluations/YOUR_EVAL_NAME \
  --problem-set problem_sets/problem_set.md \
  --model gpt-4o-mini \
  --api-key YOUR_API_KEY
```

**Parameters:**
- `--results`: Path to results.jsonl file from eval_generator
- `--output`: Directory to save evaluation results
- `--problem-set`: Path to problem_set.md containing ground truth answers (default: `problem_sets/problem_set.md`)
- `--model`: OpenAI model to use (default: `gpt-4o-mini`)
- `--api-key`: OpenAI API key (optional if `OPENAI_API_KEY` env var is set)

## Output Files

The evaluator creates two files in the output directory:

### 1. `evaluation_results.jsonl`

JSONL file with detailed evaluation for each question:

```json
{
  "question_id": "basic_001",
  "question": "What is Serial Femtosecond Crystallography?",
  "has_ground_truth": true,
  "evaluation": {
    "factual_accuracy": {
      "score": 5,
      "reasoning": "Response accurately describes SFX principles...",
      "key_errors": []
    },
    "groundedness": {
      "score": 4,
      "reasoning": "Most claims well-supported by sources...",
      "unsupported_claims": ["minor detail about timing"]
    },
    "coverage_specificity": {
      "score": 5,
      "reasoning": "Comprehensive coverage of all key aspects...",
      "missing_aspects": []
    },
    "average_score": 4.67
  },
  "timestamp": "2025-01-02T10:30:00"
}
```

### 2. `evaluation_summary.json`

Aggregate statistics across all evaluated questions:

```json
{
  "total_evaluated": 50,
  "average_scores": {
    "factual_accuracy": 4.2,
    "groundedness": 3.8,
    "coverage_specificity": 4.0,
    "overall": 4.0
  },
  "timestamp": "2025-01-02T11:00:00"
}
```

## Evaluation Dimensions

### 1. Factual Accuracy

Evaluates whether the response contains factually correct information.

**With Ground Truth:**
- Compares response against reference answer
- Identifies specific factual errors
- Checks for contradictions with ground truth

**Without Ground Truth:**
- Assesses scientific soundness based on XFEL domain knowledge
- Checks proper use of technical terminology
- Identifies obvious errors or misconceptions

### 2. Groundedness / Evidence Use

Evaluates how well the response is supported by retrieved source documents.

**Criteria:**
- Are claims backed by evidence in sources?
- Are there hallucinated facts not present in sources?
- How well are multiple sources integrated?
- Quality of citation and attribution

### 3. Coverage & Specificity

Evaluates comprehensiveness and level of detail.

**With Ground Truth:**
- Compares scope against reference answer
- Identifies missing key concepts
- Assesses appropriate level of technical detail

**Without Ground Truth:**
- Evaluates whether all aspects of question are addressed
- Checks for sufficient technical specificity
- Identifies vague or overly generic responses

## Scoring Rubric

All dimensions use a 1-5 scale:

| Score | Interpretation |
|-------|----------------|
| 5 | Excellent: Meets all criteria, no significant issues |
| 4 | Good: Meets most criteria, minor issues only |
| 3 | Moderate: Mix of strengths and weaknesses |
| 2 | Poor: Significant issues, limited quality |
| 1 | Very Poor: Fails to meet basic criteria |

## Integration with XFELBench Workflow

### Complete Evaluation Pipeline

```bash
# Step 1: Generate responses using RAG system
python eval_generator.py \
  --config configs/baseline.yaml \
  --questions problem_sets/xfel_qa_basic.json

# Step 2: Evaluate responses using LLM judge
python llm_judge.py \
  --results outputs/20251230_230056_baseline/results.jsonl \
  --output evaluations/baseline_20251230 \
  --problem-set problem_sets/problem_set.md

# Step 3: Analyze results (use existing analyze_results.py or custom analysis)
python analyze_results.py --eval-dir evaluations/baseline_20251230
```

## Ground Truth Answer Format

The evaluator parses ground truth answers from `problem_set.md` in the following format:

```markdown
QA-1 (Topic)
Question
What is the question text?
Answer
This is the ground truth answer that will be used for comparison.

QA-2 (Another Topic)
Question
Another question?
Answer
Another answer.
```

Questions without answers (e.g., numbered questions 1-5) will be evaluated without ground truth comparison.

## API Usage and Costs

The evaluator uses OpenAI's GPT-4o-mini model:
- **Model**: `gpt-4o-mini` (cost-effective, good performance)
- **Temperature**: 0.0 (deterministic evaluation)
- **Structured Output**: JSON mode for reliable parsing

**Estimated Cost** (as of 2025):
- ~3 API calls per question (one per dimension)
- GPT-4o-mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- Typical cost: $0.01-0.02 per question
- 100 questions: ~$1-2 total

## Troubleshooting

### API Key Issues

If you get "API key not provided" error:

```bash
export OPENAI_API_KEY="your-key-here"
# Or pass it directly
python llm_judge.py --api-key your-key-here ...
```

### JSON Parsing Errors

The evaluator uses `response_format={"type": "json_object"}` to ensure valid JSON. If parsing fails:
- The system retries up to 3 times
- Falls back to error record with score 0
- Check OpenAI API status if persistent

### Rate Limiting

If you hit OpenAI rate limits:
- The evaluator implements exponential backoff
- Consider adding delays between evaluations
- Use batch processing during off-peak hours

## Customization

### Using Different Models

```bash
# Use GPT-4 for higher quality evaluation
python llm_judge.py --model gpt-4 ...

# Use GPT-4-turbo for balance
python llm_judge.py --model gpt-4-turbo ...
```

### Modifying Evaluation Prompts

Edit prompts in `llm_judge.py`:
- `FACTUAL_ACCURACY_PROMPT_WITH_ANSWER`
- `FACTUAL_ACCURACY_PROMPT_NO_ANSWER`
- `GROUNDEDNESS_PROMPT`
- `COVERAGE_SPECIFICITY_PROMPT_WITH_ANSWER`
- `COVERAGE_SPECIFICITY_PROMPT_NO_ANSWER`

### Adding New Dimensions

1. Add new prompt template in `llm_judge.py`
2. Create evaluation method in `LLMJudge` class
3. Call from `evaluate_response()` method
4. Update score aggregation logic

## Example: Programmatic Usage

```python
from llm_judge import LLMJudge, load_ground_truth_answers

# Initialize judge
judge = LLMJudge(api_key="your-key", model="gpt-4o-mini")

# Load ground truth
ground_truth = load_ground_truth_answers("problem_sets/problem_set.md")

# Evaluate single response
question = "What is SFX?"
response = "Serial Femtosecond Crystallography is..."
sources = [{"title": "Paper 1", "content": "...", ...}]
gt_answer = ground_truth.get("QA-1")

evaluation = judge.evaluate_response(
    question=question,
    response=response,
    sources=sources,
    ground_truth=gt_answer
)

print(f"Factual Accuracy: {evaluation['factual_accuracy']['score']}/5")
print(f"Groundedness: {evaluation['groundedness']['score']}/5")
print(f"Coverage: {evaluation['coverage_specificity']['score']}/5")
print(f"Average: {evaluation['average_score']:.2f}/5")
```

## Citation

If you use this evaluator in your research, please cite:

```bibtex
@software{xfelbench_llm_judge,
  title={LLM-as-Judge Evaluator for XFELBench},
  author={ChatXFEL Team},
  year={2025},
  url={https://github.com/your-repo/ChatXFEL}
}
```

## Support

For issues or questions:
1. Check this README
2. Review example outputs in `evaluations/` directory
3. Open an issue on GitHub
4. Contact the ChatXFEL development team
