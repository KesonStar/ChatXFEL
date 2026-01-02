#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Evaluation Pipeline for XFELBench
Orchestrates: Config Generation -> RAG Evaluation -> LLM Judge Evaluation
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import yaml

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Now import from generation module
from generation.generate_configs import generate_all_configs, EXPERIMENT_CONFIGS


class EvaluationPipeline:
    """
    Orchestrates the complete evaluation pipeline:
    1. Generate configurations
    2. Run RAG evaluation (eval_generator.py)
    3. Run LLM judge evaluation (llm_judge.py)
    4. Aggregate results
    """

    def __init__(self, question_file: str, problem_set_file: str,
                 openai_api_key: str, base_dir: str = None):
        """
        Initialize evaluation pipeline.

        Args:
            question_file: Path to question set JSON
            problem_set_file: Path to problem_set.md with ground truth
            openai_api_key: OpenAI API key for LLM judge
            base_dir: Base directory for XFELBench (default: script directory)
        """
        self.question_file = question_file
        self.problem_set_file = problem_set_file
        self.openai_api_key = openai_api_key

        # Set base directory to XFELBench root
        if base_dir is None:
            # Script is in scripts/orchestration/, so go up 2 levels to get XFELBench root
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)

        # Create directories
        self.config_dir = self.base_dir / "configs" / "generated"
        self.output_dir = self.base_dir / "outputs"
        self.eval_dir = self.base_dir / "evaluations"

        # Timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"[INFO] Evaluation Pipeline initialized")
        print(f"[INFO] Base directory: {self.base_dir}")
        print(f"[INFO] Question file: {self.question_file}")
        print(f"[INFO] Problem set: {self.problem_set_file}")

    def step1_generate_configs(self, selected_configs: List[str] = None) -> List[str]:
        """
        Step 1: Generate configuration files.

        Args:
            selected_configs: List of config names to generate (None = all)

        Returns:
            List of generated config file paths
        """
        print("\n" + "="*60)
        print("STEP 1: Generating Configuration Files")
        print("="*60)

        config_files = generate_all_configs(
            output_dir=str(self.config_dir),
            selected_configs=selected_configs
        )

        return config_files

    def step2_run_rag_evaluation(self, config_file: str) -> Dict[str, str]:
        """
        Step 2: Run RAG evaluation for a single config.

        Args:
            config_file: Path to configuration file

        Returns:
            Dictionary with paths to results
        """
        config_name = Path(config_file).stem

        print(f"\n[INFO] Running RAG evaluation: {config_name}")

        cmd = [
            sys.executable,
            str(self.base_dir / "scripts" / "evaluation" / "eval_generator.py"),
            "--config", config_file,
            "--questions", self.question_file
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)

            # Find the output directory from stdout
            # eval_generator creates: outputs/TIMESTAMP_NAME/
            output_dirs = list(self.output_dir.glob(f"*_{config_name}"))
            if output_dirs:
                latest_output = max(output_dirs, key=lambda p: p.stat().st_mtime)
                results_file = latest_output / "results.jsonl"
                summary_file = latest_output / "summary.json"

                return {
                    "config_name": config_name,
                    "output_dir": str(latest_output),
                    "results_file": str(results_file),
                    "summary_file": str(summary_file),
                    "status": "success"
                }
            else:
                print(f"[WARNING] Could not find output directory for {config_name}")
                return {"config_name": config_name, "status": "failed", "error": "Output dir not found"}

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] RAG evaluation failed for {config_name}")
            print(f"Error: {e.stderr}")
            return {"config_name": config_name, "status": "failed", "error": str(e)}

    def step3_run_llm_judge(self, results_file: str, config_name: str) -> Dict[str, str]:
        """
        Step 3: Run LLM judge evaluation on RAG results.

        Args:
            results_file: Path to results.jsonl from RAG evaluation
            config_name: Name of the configuration

        Returns:
            Dictionary with paths to evaluation results
        """
        print(f"\n[INFO] Running LLM judge evaluation: {config_name}")

        eval_output_dir = self.eval_dir / f"{self.run_timestamp}_{config_name}"

        cmd = [
            sys.executable,
            str(self.base_dir / "scripts" / "evaluation" / "llm_judge.py"),
            "--results", results_file,
            "--output", str(eval_output_dir),
            "--problem-set", self.problem_set_file,
            "--api-key", self.openai_api_key
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)

            return {
                "config_name": config_name,
                "eval_dir": str(eval_output_dir),
                "eval_results": str(eval_output_dir / "evaluation_results.jsonl"),
                "eval_summary": str(eval_output_dir / "evaluation_summary.json"),
                "status": "success"
            }

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] LLM judge evaluation failed for {config_name}")
            print(f"Error: {e.stderr}")
            return {"config_name": config_name, "status": "failed", "error": str(e)}

    def run_full_pipeline(self, selected_configs: List[str] = None,
                         skip_generation: bool = False,
                         skip_llm_judge: bool = False):
        """
        Run the complete evaluation pipeline.

        Args:
            selected_configs: List of config names to evaluate (None = all)
            skip_generation: Skip RAG generation if results already exist
            skip_llm_judge: Skip LLM judge evaluation
        """
        print("\n" + "="*60)
        print("XFELBench Full Evaluation Pipeline")
        print("="*60)
        print(f"Run ID: {self.run_timestamp}")

        # Step 1: Generate configs
        if not skip_generation:
            config_files = self.step1_generate_configs(selected_configs)
        else:
            # Load existing configs
            if selected_configs:
                config_files = [str(self.config_dir / f"{name}.yaml") for name in selected_configs]
            else:
                config_files = list(self.config_dir.glob("*.yaml"))
                config_files = [str(f) for f in config_files if f.name != "CONFIG_SUMMARY.yaml"]

        # Track results
        all_results = []

        # Step 2 & 3: For each config, run RAG eval then LLM judge
        for i, config_file in enumerate(config_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing Configuration {i}/{len(config_files)}")
            print(f"Config: {Path(config_file).stem}")
            print(f"{'='*60}")

            # Run RAG evaluation
            rag_result = self.step2_run_rag_evaluation(config_file)

            if rag_result["status"] != "success":
                print(f"[WARNING] Skipping LLM judge for failed RAG evaluation")
                all_results.append(rag_result)
                continue

            # Run LLM judge evaluation
            if not skip_llm_judge:
                judge_result = self.step3_run_llm_judge(
                    rag_result["results_file"],
                    rag_result["config_name"]
                )

                # Merge results
                combined_result = {**rag_result, **judge_result}
                all_results.append(combined_result)
            else:
                all_results.append(rag_result)

        # Step 4: Generate final summary
        self.generate_final_summary(all_results)

        print(f"\n{'='*60}")
        print("Pipeline Completed!")
        print(f"{'='*60}")

        return all_results

    def generate_final_summary(self, all_results: List[Dict[str, Any]]):
        """
        Generate a final summary comparing all configurations.

        Args:
            all_results: List of result dictionaries from all configs
        """
        print("\n" + "="*60)
        print("STEP 4: Generating Final Summary")
        print("="*60)

        summary_dir = self.eval_dir / f"summary_{self.run_timestamp}"
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Collect evaluation scores
        comparison_data = []

        for result in all_results:
            if result["status"] != "success":
                continue

            config_name = result["config_name"]

            # Load evaluation summary if available
            if "eval_summary" in result and os.path.exists(result["eval_summary"]):
                with open(result["eval_summary"], 'r') as f:
                    eval_summary = json.load(f)

                comparison_data.append({
                    "config": config_name,
                    "total_evaluated": eval_summary.get("total_evaluated", 0),
                    "factual_accuracy": eval_summary["average_scores"].get("factual_accuracy", 0),
                    "groundedness": eval_summary["average_scores"].get("groundedness", 0),
                    "coverage_specificity": eval_summary["average_scores"].get("coverage_specificity", 0),
                    "overall": eval_summary["average_scores"].get("overall", 0)
                })

        # Sort by overall score
        comparison_data.sort(key=lambda x: x["overall"], reverse=True)

        # Save comparison table
        comparison_file = summary_dir / "comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)

        # Generate markdown report
        report_file = summary_dir / "EVALUATION_REPORT.md"
        self.generate_markdown_report(comparison_data, all_results, report_file)

        print(f"[INFO] Summary saved to: {summary_dir}")
        print(f"[INFO] Comparison: {comparison_file}")
        print(f"[INFO] Report: {report_file}")

        # Print ranking table
        print("\n" + "="*60)
        print("Configuration Ranking (by Overall Score)")
        print("="*60)
        print(f"{'Rank':<6} {'Config':<25} {'Overall':<10} {'Factual':<10} {'Grounded':<10} {'Coverage':<10}")
        print("-"*80)

        for i, row in enumerate(comparison_data, 1):
            print(f"{i:<6} {row['config']:<25} {row['overall']:<10.2f} "
                  f"{row['factual_accuracy']:<10.2f} {row['groundedness']:<10.2f} "
                  f"{row['coverage_specificity']:<10.2f}")

    def generate_markdown_report(self, comparison_data: List[Dict],
                                 all_results: List[Dict],
                                 output_file: Path):
        """Generate a markdown evaluation report"""

        report = f"""# XFELBench Evaluation Report

**Run ID:** {self.run_timestamp}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Question Set:** {self.question_file}
**Total Configurations:** {len(all_results)}

---

## Configuration Ranking

Configurations ranked by overall score (average of three dimensions):

| Rank | Configuration | Overall | Factual Accuracy | Groundedness | Coverage & Specificity |
|------|---------------|---------|------------------|--------------|------------------------|
"""

        for i, row in enumerate(comparison_data, 1):
            report += f"| {i} | {row['config']} | {row['overall']:.2f} | {row['factual_accuracy']:.2f} | {row['groundedness']:.2f} | {row['coverage_specificity']:.2f} |\n"

        report += "\n---\n\n## Dimension Analysis\n\n"

        # Best in each dimension
        if comparison_data:
            best_factual = max(comparison_data, key=lambda x: x['factual_accuracy'])
            best_grounded = max(comparison_data, key=lambda x: x['groundedness'])
            best_coverage = max(comparison_data, key=lambda x: x['coverage_specificity'])

            report += f"### Best Configurations by Dimension\n\n"
            report += f"- **Factual Accuracy:** {best_factual['config']} ({best_factual['factual_accuracy']:.2f})\n"
            report += f"- **Groundedness:** {best_grounded['config']} ({best_grounded['groundedness']:.2f})\n"
            report += f"- **Coverage & Specificity:** {best_coverage['config']} ({best_coverage['coverage_specificity']:.2f})\n\n"

        report += "---\n\n## Configuration Details\n\n"

        # Add config descriptions
        for result in all_results:
            if result["status"] == "success":
                config_name = result["config_name"]
                description = EXPERIMENT_CONFIGS.get(config_name, {}).get("description", "N/A")

                report += f"### {config_name}\n\n"
                report += f"**Description:** {description}\n\n"

                if "eval_summary" in result:
                    report += f"- Output: `{result['output_dir']}`\n"
                    report += f"- Evaluation: `{result['eval_dir']}`\n\n"

        report += "---\n\n## Methodology\n\n"
        report += "### Evaluation Dimensions\n\n"
        report += "1. **Factual Accuracy (1-5):** Correctness of facts and concepts\n"
        report += "2. **Groundedness (1-5):** Support from retrieved source documents\n"
        report += "3. **Coverage & Specificity (1-5):** Comprehensiveness and detail level\n\n"

        report += "### Scoring\n\n"
        report += "- **5:** Excellent\n"
        report += "- **4:** Good\n"
        report += "- **3:** Moderate\n"
        report += "- **2:** Poor\n"
        report += "- **1:** Very Poor\n\n"

        report += "### LLM Judge\n\n"
        report += "- **Model:** GPT-4o-mini\n"
        report += "- **Temperature:** 0.0 (deterministic)\n"
        report += "- **Ground Truth:** Used when available for questions QA-1 through QA-9\n\n"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='XFELBench Full Evaluation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on all configs
  python run_full_evaluation.py --questions problem_sets/xfel_qa_basic.json

  # Run on specific configs only
  python run_full_evaluation.py --questions problem_sets/xfel_qa_basic.json \\
      --configs baseline hybrid_search full_features

  # Skip RAG generation (use existing results)
  python run_full_evaluation.py --questions problem_sets/xfel_qa_basic.json \\
      --skip-generation

  # Skip LLM judge (only generate RAG responses)
  python run_full_evaluation.py --questions problem_sets/xfel_qa_basic.json \\
      --skip-llm-judge
        """
    )

    parser.add_argument('--questions', type=str,
                       help='Path to question set JSON file')
    parser.add_argument('--problem-set', type=str,
                       default='problem_sets/problem_set.md',
                       help='Path to problem_set.md with ground truth (default: problem_sets/problem_set.md)')
    parser.add_argument('--configs', type=str, nargs='+',
                       help='Specific configs to evaluate (default: all)')
    parser.add_argument('--api-key', type=str,
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip RAG generation (use existing results)')
    parser.add_argument('--skip-llm-judge', action='store_true',
                       help='Skip LLM judge evaluation')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configuration templates and exit')

    args = parser.parse_args()

    # List configs if requested
    if args.list_configs:
        print("\n[INFO] Available configuration templates:\n")
        for name, cfg in EXPERIMENT_CONFIGS.items():
            print(f"  {name:25s} - {cfg['description']}")
        print(f"\n[INFO] Total: {len(EXPERIMENT_CONFIGS)} templates")
        return

    # Validate required arguments
    if not args.questions:
        print("[ERROR] --questions argument is required")
        parser.print_help()
        sys.exit(1)

    # Validate files
    if not os.path.exists(args.questions):
        print(f"[ERROR] Question file not found: {args.questions}")
        sys.exit(1)

    if not os.path.exists(args.problem_set):
        print(f"[WARNING] Problem set file not found: {args.problem_set}")

    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key and not args.skip_llm_judge:
        print("[ERROR] OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
        sys.exit(1)

    # Initialize pipeline
    pipeline = EvaluationPipeline(
        question_file=args.questions,
        problem_set_file=args.problem_set,
        openai_api_key=api_key
    )

    # Run pipeline
    try:
        results = pipeline.run_full_pipeline(
            selected_configs=args.configs,
            skip_generation=args.skip_generation,
            skip_llm_judge=args.skip_llm_judge
        )

        # Print final status
        success_count = sum(1 for r in results if r.get("status") == "success")
        print(f"\n[INFO] Pipeline completed: {success_count}/{len(results)} configs successful")

    except KeyboardInterrupt:
        print("\n[WARNING] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
