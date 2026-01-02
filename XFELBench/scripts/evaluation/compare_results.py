#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Results Comparison Tool for XFELBench
Compare and visualize evaluation results across different configurations
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import csv


def load_evaluation_summary(eval_dir: str) -> Dict[str, Any]:
    """Load evaluation summary from a directory"""
    summary_file = Path(eval_dir) / "evaluation_summary.json"

    if not summary_file.exists():
        return None

    with open(summary_file, 'r') as f:
        return json.load(f)


def collect_all_results(eval_base_dir: str) -> List[Dict[str, Any]]:
    """
    Collect all evaluation results from evaluations directory.

    Args:
        eval_base_dir: Base directory containing evaluation results

    Returns:
        List of result dictionaries
    """
    eval_path = Path(eval_base_dir)

    if not eval_path.exists():
        print(f"[ERROR] Evaluation directory not found: {eval_base_dir}")
        return []

    results = []

    # Find all evaluation directories (format: TIMESTAMP_CONFIG_NAME)
    for eval_dir in eval_path.iterdir():
        if not eval_dir.is_dir():
            continue

        # Skip summary directories
        if eval_dir.name.startswith("summary_"):
            continue

        # Extract config name from directory name
        # Format: YYYYMMDD_HHMMSS_config_name
        parts = eval_dir.name.split("_", 2)
        if len(parts) >= 3:
            config_name = parts[2]
        else:
            config_name = eval_dir.name

        # Load summary
        summary = load_evaluation_summary(str(eval_dir))

        if summary:
            results.append({
                "config": config_name,
                "eval_dir": str(eval_dir),
                "total_evaluated": summary.get("total_evaluated", 0),
                "factual_accuracy": summary["average_scores"].get("factual_accuracy", 0),
                "groundedness": summary["average_scores"].get("groundedness", 0),
                "coverage_specificity": summary["average_scores"].get("coverage_specificity", 0),
                "overall": summary["average_scores"].get("overall", 0),
                "timestamp": summary.get("timestamp", "")
            })

    return results


def print_comparison_table(results: List[Dict[str, Any]]):
    """Print a formatted comparison table"""

    # Sort by overall score (descending)
    sorted_results = sorted(results, key=lambda x: x["overall"], reverse=True)

    print("\n" + "="*100)
    print("Configuration Comparison Table")
    print("="*100)
    print(f"{'Rank':<6} {'Configuration':<30} {'Overall':<10} {'Factual':<10} {'Grounded':<10} {'Coverage':<10} {'Total Q':<8}")
    print("-"*100)

    for i, result in enumerate(sorted_results, 1):
        print(f"{i:<6} {result['config']:<30} {result['overall']:<10.2f} "
              f"{result['factual_accuracy']:<10.2f} {result['groundedness']:<10.2f} "
              f"{result['coverage_specificity']:<10.2f} {result['total_evaluated']:<8}")

    print("="*100)
    print("")


def generate_csv_report(results: List[Dict[str, Any]], output_file: str):
    """Generate CSV report"""

    # Sort by overall score
    sorted_results = sorted(results, key=lambda x: x["overall"], reverse=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['rank', 'config', 'overall', 'factual_accuracy',
                     'groundedness', 'coverage_specificity', 'total_evaluated', 'eval_dir']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for i, result in enumerate(sorted_results, 1):
            writer.writerow({
                'rank': i,
                'config': result['config'],
                'overall': f"{result['overall']:.2f}",
                'factual_accuracy': f"{result['factual_accuracy']:.2f}",
                'groundedness': f"{result['groundedness']:.2f}",
                'coverage_specificity': f"{result['coverage_specificity']:.2f}",
                'total_evaluated': result['total_evaluated'],
                'eval_dir': result['eval_dir']
            })

    print(f"[INFO] CSV report saved to: {output_file}")


def find_best_configs(results: List[Dict[str, Any]]):
    """Find best configurations for each dimension"""

    if not results:
        return

    best_overall = max(results, key=lambda x: x['overall'])
    best_factual = max(results, key=lambda x: x['factual_accuracy'])
    best_grounded = max(results, key=lambda x: x['groundedness'])
    best_coverage = max(results, key=lambda x: x['coverage_specificity'])

    print("\n" + "="*100)
    print("Best Configurations by Dimension")
    print("="*100)

    print(f"\n{'Dimension':<25} {'Configuration':<30} {'Score':<10}")
    print("-"*65)
    print(f"{'Overall':<25} {best_overall['config']:<30} {best_overall['overall']:<10.2f}")
    print(f"{'Factual Accuracy':<25} {best_factual['config']:<30} {best_factual['factual_accuracy']:<10.2f}")
    print(f"{'Groundedness':<25} {best_grounded['config']:<30} {best_grounded['groundedness']:<10.2f}")
    print(f"{'Coverage & Specificity':<25} {best_coverage['config']:<30} {best_coverage['coverage_specificity']:<10.2f}")
    print("")


def analyze_score_distribution(results: List[Dict[str, Any]]):
    """Analyze score distribution statistics"""

    if not results:
        return

    overall_scores = [r['overall'] for r in results]
    factual_scores = [r['factual_accuracy'] for r in results]
    grounded_scores = [r['groundedness'] for r in results]
    coverage_scores = [r['coverage_specificity'] for r in results]

    def stats(scores):
        return {
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'range': max(scores) - min(scores)
        }

    print("\n" + "="*100)
    print("Score Distribution Statistics")
    print("="*100)

    print(f"\n{'Dimension':<25} {'Mean':<10} {'Min':<10} {'Max':<10} {'Range':<10}")
    print("-"*65)

    for name, scores in [
        ('Overall', overall_scores),
        ('Factual Accuracy', factual_scores),
        ('Groundedness', grounded_scores),
        ('Coverage & Specificity', coverage_scores)
    ]:
        s = stats(scores)
        print(f"{name:<25} {s['mean']:<10.2f} {s['min']:<10.2f} {s['max']:<10.2f} {s['range']:<10.2f}")

    print("")


def compare_specific_configs(results: List[Dict[str, Any]], config_names: List[str]):
    """Compare specific configurations in detail"""

    selected = [r for r in results if r['config'] in config_names]

    if not selected:
        print(f"[WARNING] No results found for configs: {config_names}")
        return

    print("\n" + "="*100)
    print(f"Detailed Comparison: {', '.join(config_names)}")
    print("="*100)

    for result in selected:
        print(f"\n{result['config']}:")
        print(f"  Overall Score:           {result['overall']:.2f}")
        print(f"  Factual Accuracy:        {result['factual_accuracy']:.2f}")
        print(f"  Groundedness:            {result['groundedness']:.2f}")
        print(f"  Coverage & Specificity:  {result['coverage_specificity']:.2f}")
        print(f"  Questions Evaluated:     {result['total_evaluated']}")
        print(f"  Evaluation Directory:    {result['eval_dir']}")

    # Show differences
    if len(selected) == 2:
        print("\n" + "-"*50)
        print("Difference (Config 1 - Config 2):")
        print("-"*50)

        diff_overall = selected[0]['overall'] - selected[1]['overall']
        diff_factual = selected[0]['factual_accuracy'] - selected[1]['factual_accuracy']
        diff_grounded = selected[0]['groundedness'] - selected[1]['groundedness']
        diff_coverage = selected[0]['coverage_specificity'] - selected[1]['coverage_specificity']

        print(f"  Overall:         {diff_overall:+.2f}")
        print(f"  Factual:         {diff_factual:+.2f}")
        print(f"  Groundedness:    {diff_grounded:+.2f}")
        print(f"  Coverage:        {diff_coverage:+.2f}")

    print("")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Compare XFELBench evaluation results across configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show comparison table for all results
  python compare_results.py

  # Generate CSV report
  python compare_results.py --csv comparison.csv

  # Compare specific configs
  python compare_results.py --compare baseline hybrid_search full_features

  # Use custom evaluation directory
  python compare_results.py --eval-dir evaluations
        """
    )

    parser.add_argument('--eval-dir', type=str, default='evaluations',
                       help='Base directory containing evaluation results (default: evaluations)')
    parser.add_argument('--csv', type=str,
                       help='Output CSV file for comparison report')
    parser.add_argument('--compare', type=str, nargs='+',
                       help='Compare specific configurations in detail')
    parser.add_argument('--stats', action='store_true',
                       help='Show score distribution statistics')

    args = parser.parse_args()

    # Collect results
    print(f"[INFO] Collecting results from: {args.eval_dir}")
    results = collect_all_results(args.eval_dir)

    if not results:
        print("[ERROR] No evaluation results found")
        return

    print(f"[INFO] Found {len(results)} evaluation results")

    # Print comparison table
    print_comparison_table(results)

    # Find best configs
    find_best_configs(results)

    # Show statistics if requested
    if args.stats:
        analyze_score_distribution(results)

    # Compare specific configs if requested
    if args.compare:
        compare_specific_configs(results, args.compare)

    # Generate CSV if requested
    if args.csv:
        generate_csv_report(results, args.csv)


if __name__ == '__main__':
    main()
