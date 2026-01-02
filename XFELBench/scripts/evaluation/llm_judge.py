#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XFELBench LLM-as-Judge Evaluator
Evaluates RAG responses using GPT-4o-mini across three dimensions:
1. Factual Accuracy
2. Groundedness / Evidence Use
3. Coverage & Specificity
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import time

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] OpenAI library not installed. Install with: pip install openai")
    exit(1)


# Evaluation prompts for each dimension
FACTUAL_ACCURACY_PROMPT_WITH_ANSWER = """You are an expert evaluator assessing the factual accuracy of a response to a scientific question about X-ray Free-Electron Lasers (XFELs).

**Question:**
{question}

**Ground Truth Answer:**
{ground_truth}

**Generated Response:**
{response}

**Task:**
Evaluate the factual accuracy of the Generated Response by comparing it with the Ground Truth Answer. Consider:
- Are the key facts and concepts correctly stated?
- Are there any factual errors or misconceptions?
- Does the response contradict the ground truth in any significant way?
- Are technical terms and scientific principles used correctly?

**Scoring Criteria (1-5):**
5 - Completely accurate: All facts align with ground truth, no errors
4 - Mostly accurate: Minor imprecisions but no significant errors
3 - Partially accurate: Some correct information but notable errors or gaps
2 - Minimally accurate: Several significant errors, limited correct information
1 - Inaccurate: Predominantly incorrect or misleading information

Provide your evaluation in the following JSON format:
{{
  "score": <float 1-10>,
  "reasoning": "<brief explanation of your score>",
  "key_errors": ["<list of any factual errors found>"]
}}
"""

FACTUAL_ACCURACY_PROMPT_NO_ANSWER = """You are an expert evaluator assessing the factual accuracy of a response to a scientific question about X-ray Free-Electron Lasers (XFELs).

**Question:**
{question}

**Generated Response:**
{response}

**Task:**
Evaluate the factual accuracy of the Generated Response based on your knowledge of XFEL science. Consider:
- Are the scientific concepts and principles correctly explained?
- Are technical terms used accurately and appropriately?
- Are there any obvious factual errors or misconceptions?
- Is the information scientifically sound and consistent with established XFEL knowledge?

**Scoring Criteria (1-5):**
5 - Highly accurate: Information is scientifically sound and technically correct
4 - Mostly accurate: Generally correct with minor imprecisions
3 - Partially accurate: Mix of correct and questionable information
2 - Minimally accurate: Contains notable errors or misconceptions
1 - Inaccurate: Predominantly incorrect or misleading information

Provide your evaluation in the following JSON format:
{{
  "score": <float 1-10>,
  "reasoning": "<brief explanation of your score>",
  "key_errors": ["<list of any factual errors found>"]
}}
"""

GROUNDEDNESS_PROMPT = """You are an expert evaluator assessing how well a response is grounded in the provided source documents.

**Question:**
{question}

**Generated Response:**
{response}

**Source Documents:**
{sources}

**Task:**
Evaluate how well the Generated Response is supported by the Source Documents. Consider:
- Are the claims in the response directly supported by evidence in the sources?
- Does the response cite or reference specific information from the sources?
- Are there unsupported claims or hallucinated information not found in the sources?
- How well does the response integrate and synthesize information from multiple sources?

**Scoring Criteria (1-5):**
5 - Fully grounded: All claims directly supported by sources, excellent citation
4 - Well grounded: Most claims supported, minor unsupported details
3 - Moderately grounded: Mix of supported and unsupported claims
2 - Poorly grounded: Many claims lack source support, significant hallucination
1 - Not grounded: Response largely ignores sources or invents information

Provide your evaluation in the following JSON format:
{{
  "score": <float 1-10>,
  "reasoning": "<brief explanation of your score>",
  "unsupported_claims": ["<list of claims not supported by sources>"]
}}
"""

COVERAGE_SPECIFICITY_PROMPT_WITH_ANSWER = """You are an expert evaluator assessing the coverage and specificity of a response to a scientific question about X-ray Free-Electron Lasers (XFELs).

**Question:**
{question}

**Ground Truth Answer:**
{ground_truth}

**Generated Response:**
{response}

**Task:**
Evaluate the coverage and specificity of the Generated Response compared to the Ground Truth. Consider:
- Does the response address all key aspects of the question?
- How comprehensive is the coverage compared to the ground truth?
- Does it provide sufficient technical detail and specificity?
- Are important concepts or details omitted?
- Is the response too vague or generic?

**Scoring Criteria (1-5):**
5 - Excellent coverage: Addresses all key points with appropriate detail
4 - Good coverage: Covers main points but may miss minor details
3 - Moderate coverage: Addresses question partially, lacks some important aspects
2 - Limited coverage: Misses several key points or lacks sufficient detail
1 - Poor coverage: Fails to address the question adequately, too vague

Provide your evaluation in the following JSON format:
{{
  "score": <float 1-10>,
  "reasoning": "<brief explanation of your score>",
  "missing_aspects": ["<list of important aspects not covered>"]
}}
"""

COVERAGE_SPECIFICITY_PROMPT_NO_ANSWER = """You are an expert evaluator assessing the coverage and specificity of a response to a scientific question about X-ray Free-Electron Lasers (XFELs).

**Question:**
{question}

**Generated Response:**
{response}

**Task:**
Evaluate the coverage and specificity of the Generated Response. Consider:
- Does the response fully address all aspects of the question?
- Is the response sufficiently detailed and specific?
- Does it provide concrete examples or technical details when appropriate?
- Is the response comprehensive or does it leave important aspects unaddressed?
- Is the response too vague or generic?

**Scoring Criteria (1-5):**
5 - Excellent coverage: Comprehensive and detailed, addresses all aspects
4 - Good coverage: Addresses main aspects with reasonable detail
3 - Moderate coverage: Covers some aspects but lacks detail or completeness
2 - Limited coverage: Superficial treatment, misses important aspects
1 - Poor coverage: Fails to adequately address the question, too vague

Provide your evaluation in the following JSON format:
{{
  "score": <float 1-10>,
  "reasoning": "<brief explanation of your score>",
  "missing_aspects": ["<list of important aspects not adequately covered>"]
}}
"""


class LLMJudge:
    """
    LLM-as-Judge evaluator using OpenAI GPT-4o-mini.
    Evaluates responses across three dimensions with separate prompts.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-mini"):
        """
        Initialize LLM Judge.

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")
        self.model = model
        print(f"[INFO] Initialized LLM Judge with model: {model}")

    def _call_openai(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Call OpenAI API with retry logic.

        Args:
            prompt: Evaluation prompt
            max_retries: Maximum number of retries on failure

        Returns:
            Parsed JSON response from the model
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert scientific evaluator. Provide your evaluation in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,  # Deterministic evaluation
                    response_format={"type": "json_object"}
                )

                result_text = response.choices[0].message.content
                result = json.loads(result_text)
                return result

            except json.JSONDecodeError as e:
                print(f"[WARNING] JSON parsing error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return {"score": 0, "reasoning": f"JSON parsing error: {e}", "error": str(e)}
                time.sleep(1)

            except Exception as e:
                print(f"[WARNING] API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return {"score": 0, "reasoning": f"API error: {e}", "error": str(e)}
                time.sleep(2 ** attempt)  # Exponential backoff

        return {"score": 0, "reasoning": "Max retries exceeded", "error": "Max retries exceeded"}

    def evaluate_factual_accuracy(self, question: str, response: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate factual accuracy of the response.

        Args:
            question: Question text
            response: Generated response
            ground_truth: Ground truth answer (optional)

        Returns:
            Evaluation result with score and reasoning
        """
        if ground_truth:
            prompt = FACTUAL_ACCURACY_PROMPT_WITH_ANSWER.format(
                question=question,
                ground_truth=ground_truth,
                response=response
            )
        else:
            prompt = FACTUAL_ACCURACY_PROMPT_NO_ANSWER.format(
                question=question,
                response=response
            )

        return self._call_openai(prompt)

    def evaluate_groundedness(self, question: str, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate how well the response is grounded in source documents.

        Args:
            question: Question text
            response: Generated response
            sources: List of source documents with metadata and content

        Returns:
            Evaluation result with score and reasoning
        """
        # Format sources for the prompt
        sources_text = ""
        for i, source in enumerate(sources, 1):
            sources_text += f"\n**Source {i}:**\n"
            sources_text += f"Title: {source.get('title', 'N/A')}\n"
            sources_text += f"DOI: {source.get('doi', 'N/A')}\n"
            sources_text += f"Journal: {source.get('journal', 'N/A')}, Year: {source.get('year', 'N/A')}\n"
            sources_text += f"Content: {source.get('content', '')[:500]}...\n"  # Truncate long content

        prompt = GROUNDEDNESS_PROMPT.format(
            question=question,
            response=response,
            sources=sources_text
        )

        return self._call_openai(prompt)

    def evaluate_coverage_specificity(self, question: str, response: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate coverage and specificity of the response.

        Args:
            question: Question text
            response: Generated response
            ground_truth: Ground truth answer (optional)

        Returns:
            Evaluation result with score and reasoning
        """
        if ground_truth:
            prompt = COVERAGE_SPECIFICITY_PROMPT_WITH_ANSWER.format(
                question=question,
                ground_truth=ground_truth,
                response=response
            )
        else:
            prompt = COVERAGE_SPECIFICITY_PROMPT_NO_ANSWER.format(
                question=question,
                response=response
            )

        return self._call_openai(prompt)

    def evaluate_response(self, question: str, response: str, sources: List[Dict[str, Any]], ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform complete evaluation across all three dimensions.

        Args:
            question: Question text
            response: Generated response
            sources: List of source documents
            ground_truth: Ground truth answer (optional)

        Returns:
            Complete evaluation results
        """
        print(f"[INFO] Evaluating response for question: {question[:80]}...")

        results = {
            "factual_accuracy": self.evaluate_factual_accuracy(question, response, ground_truth),
            "groundedness": self.evaluate_groundedness(question, response, sources),
            "coverage_specificity": self.evaluate_coverage_specificity(question, response, ground_truth)
        }

        # Calculate average score
        scores = [
            results["factual_accuracy"].get("score", 0),
            results["groundedness"].get("score", 0),
            results["coverage_specificity"].get("score", 0)
        ]
        results["average_score"] = sum(scores) / len(scores) if scores else 0

        return results


def load_ground_truth_answers(problem_set_file: str) -> Dict[str, str]:
    """
    Load ground truth answers from problem_set.md file.

    Args:
        problem_set_file: Path to problem_set.md

    Returns:
        Dictionary mapping question IDs to ground truth answers
    """
    ground_truth = {}

    try:
        with open(problem_set_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse QA pairs (QA-1, QA-2, etc.)
        # Simple parsing: look for "QA-X" followed by "Answer"
        import re
        qa_pattern = r'QA-(\d+)[^\n]*\nQuestion\n(.*?)\nAnswer\n(.*?)(?=\n(?:QA-\d+|\d+\.|$))'
        matches = re.finditer(qa_pattern, content, re.DOTALL)

        for match in matches:
            qa_id = f"QA-{match.group(1)}"
            answer = match.group(3).strip()
            ground_truth[qa_id] = answer
            print(f"[INFO] Loaded ground truth for {qa_id}")

    except FileNotFoundError:
        print(f"[WARNING] Problem set file not found: {problem_set_file}")
    except Exception as e:
        print(f"[WARNING] Error loading ground truth: {e}")

    return ground_truth


def evaluate_results_file(results_file: str, output_dir: str, judge: LLMJudge, ground_truth: Dict[str, str]):
    """
    Evaluate all responses in a results.jsonl file.

    Args:
        results_file: Path to results.jsonl from eval_generator
        output_dir: Output directory for evaluation results
        judge: LLMJudge instance
        ground_truth: Dictionary of ground truth answers
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    eval_results_file = output_path / 'evaluation_results.jsonl'
    summary_file = output_path / 'evaluation_summary.json'

    print(f"[INFO] Starting evaluation of {results_file}")
    print(f"[INFO] Output will be saved to {output_dir}")

    all_scores = {
        "factual_accuracy": [],
        "groundedness": [],
        "coverage_specificity": [],
        "average": []
    }

    with open(results_file, 'r', encoding='utf-8') as f_in, \
         open(eval_results_file, 'w', encoding='utf-8') as f_out:

        lines = f_in.readlines()

        for line in tqdm(lines, desc="Evaluating responses"):
            try:
                result = json.loads(line)

                # Skip if error in original result
                if 'error' in result:
                    continue

                question_id = result.get('question_id', '')
                question = result.get('question', '')
                response = result.get('answer', '')
                sources = result.get('sources', [])

                # Check if ground truth exists
                gt_answer = ground_truth.get(question_id)

                # Perform evaluation
                evaluation = judge.evaluate_response(
                    question=question,
                    response=response,
                    sources=sources,
                    ground_truth=gt_answer
                )

                # Store result
                eval_record = {
                    "question_id": question_id,
                    "question": question,
                    "has_ground_truth": gt_answer is not None,
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat()
                }

                f_out.write(json.dumps(eval_record, ensure_ascii=False) + '\n')
                f_out.flush()

                # Collect scores for summary
                all_scores["factual_accuracy"].append(evaluation["factual_accuracy"].get("score", 0))
                all_scores["groundedness"].append(evaluation["groundedness"].get("score", 0))
                all_scores["coverage_specificity"].append(evaluation["coverage_specificity"].get("score", 0))
                all_scores["average"].append(evaluation.get("average_score", 0))

            except Exception as e:
                print(f"[ERROR] Failed to evaluate response: {e}")
                continue

    # Generate summary statistics
    summary = {
        "total_evaluated": len(all_scores["average"]),
        "average_scores": {
            "factual_accuracy": sum(all_scores["factual_accuracy"]) / len(all_scores["factual_accuracy"]) if all_scores["factual_accuracy"] else 0,
            "groundedness": sum(all_scores["groundedness"]) / len(all_scores["groundedness"]) if all_scores["groundedness"] else 0,
            "coverage_specificity": sum(all_scores["coverage_specificity"]) / len(all_scores["coverage_specificity"]) if all_scores["coverage_specificity"] else 0,
            "overall": sum(all_scores["average"]) / len(all_scores["average"]) if all_scores["average"] else 0
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Evaluation complete!")
    print(f"[INFO] Total responses evaluated: {summary['total_evaluated']}")
    print(f"[INFO] Average scores:")
    print(f"  - Factual Accuracy: {summary['average_scores']['factual_accuracy']:.2f}")
    print(f"  - Groundedness: {summary['average_scores']['groundedness']:.2f}")
    print(f"  - Coverage & Specificity: {summary['average_scores']['coverage_specificity']:.2f}")
    print(f"  - Overall: {summary['average_scores']['overall']:.2f}")


def main():
    """Main entry point for LLM judge evaluation"""
    parser = argparse.ArgumentParser(description='XFELBench LLM-as-Judge Evaluator')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results.jsonl file from eval_generator')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for evaluation results')
    parser.add_argument('--problem-set', type=str,
                       default='problem_sets/problem_set.md',
                       help='Path to problem_set.md with ground truth answers')
    parser.add_argument('--api-key', type=str,
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='OpenAI model to use (default: gpt-4o-mini)')

    args = parser.parse_args()

    # Validate results file
    if not os.path.exists(args.results):
        print(f"[ERROR] Results file not found: {args.results}")
        exit(1)

    # Initialize judge
    try:
        judge = LLMJudge(api_key=args.api_key, model=args.model)
    except ValueError as e:
        print(f"[ERROR] {e}")
        exit(1)

    # Load ground truth answers
    ground_truth = load_ground_truth_answers(args.problem_set)
    print(f"[INFO] Loaded {len(ground_truth)} ground truth answers")

    # Run evaluation
    evaluate_results_file(args.results, args.output, judge, ground_truth)


if __name__ == '__main__':
    main()
