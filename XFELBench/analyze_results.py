#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple analysis script to compare evaluation results
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


def load_results(result_file: Path) -> List[Dict]:
    """Load results from JSONL file"""
    results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def load_summary(summary_file: Path) -> Dict:
    """Load summary JSON file"""
    with open(summary_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_output_dir(output_dir: Path):
    """Analyze a single evaluation output directory"""
    summary_file = output_dir / 'summary.json'
    results_file = output_dir / 'results.jsonl'

    if not summary_file.exists() or not results_file.exists():
        print(f"⚠️  Incomplete output: {output_dir.name}")
        return None

    summary = load_summary(summary_file)
    results = load_results(results_file)

    # Compute statistics
    stats = {
        'experiment': summary['experiment_name'],
        'total': summary['total_questions'],
        'completed': summary['completed_questions'],
        'failed': summary['failed_questions'],
        'avg_time': summary.get('average_generation_time', 0),
        'timestamp': output_dir.name.split('_')[0],
    }

    # Compute per-category statistics if available
    category_stats = defaultdict(lambda: {'count': 0, 'total_time': 0})
    for result in results:
        if 'error' not in result:
            metadata = result.get('metadata', {})
            category = metadata.get('category', 'unknown')
            category_stats[category]['count'] += 1
            category_stats[category]['total_time'] += result.get('generation_time', 0)

    stats['categories'] = dict(category_stats)

    return stats


def compare_experiments(outputs_dir: Path = Path('outputs')):
    """Compare all experiments in outputs directory"""
    if not outputs_dir.exists():
        print(f"❌ Outputs directory not found: {outputs_dir}")
        return

    # Get all output directories sorted by timestamp (newest first)
    output_dirs = sorted(
        [d for d in outputs_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
        reverse=True
    )

    if not output_dirs:
        print("❌ No evaluation results found in outputs/")
        return

    print("=" * 80)
    print("XFELBench Results Analysis")
    print("=" * 80)
    print()

    # Analyze each experiment
    all_stats = []
    for output_dir in output_dirs:
        stats = analyze_output_dir(output_dir)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("❌ No valid results found")
        return

    # Print summary table
    print("Experiment Summary:")
    print("-" * 80)
    print(f"{'Experiment':<25} {'Total':<8} {'Done':<8} {'Failed':<8} {'Avg Time':<12}")
    print("-" * 80)

    for stats in all_stats:
        exp_name = stats['experiment']
        print(f"{exp_name:<25} {stats['total']:<8} {stats['completed']:<8} "
              f"{stats['failed']:<8} {stats['avg_time']:<12.2f}s")

    print("-" * 80)
    print()

    # Print category breakdown for latest experiment
    if all_stats:
        latest = all_stats[0]
        print(f"Category Breakdown (Latest: {latest['experiment']}):")
        print("-" * 80)
        print(f"{'Category':<20} {'Count':<10} {'Avg Time':<15}")
        print("-" * 80)

        for category, data in latest['categories'].items():
            count = data['count']
            avg_time = data['total_time'] / count if count > 0 else 0
            print(f"{category:<20} {count:<10} {avg_time:<15.2f}s")

        print("-" * 80)
        print()

    # Group by experiment type (for comparing different configs)
    experiment_groups = defaultdict(list)
    for stats in all_stats:
        exp_name = stats['experiment']
        experiment_groups[exp_name].append(stats)

    if len(experiment_groups) > 1:
        print("Performance Comparison Across Experiments:")
        print("-" * 80)

        # Get one representative from each experiment type (most recent)
        comparison = {}
        for exp_name, runs in experiment_groups.items():
            comparison[exp_name] = runs[0]  # Most recent run

        # Sort by average time
        sorted_exps = sorted(comparison.items(), key=lambda x: x[1]['avg_time'])

        print(f"{'Rank':<6} {'Experiment':<25} {'Avg Time':<15} {'Speedup':<10}")
        print("-" * 80)

        baseline_time = sorted_exps[0][1]['avg_time']
        for rank, (exp_name, stats) in enumerate(sorted_exps, 1):
            avg_time = stats['avg_time']
            speedup = baseline_time / avg_time if avg_time > 0 else 0
            print(f"{rank:<6} {exp_name:<25} {avg_time:<15.2f}s {speedup:<10.2f}x")

        print("-" * 80)
        print()

    print("💡 Tips:")
    print("  - View detailed results: cat outputs/<dir>/results.jsonl | python -m json.tool")
    print("  - View config: cat outputs/<dir>/config.yaml")
    print("  - Compare answers for a specific question ID across experiments")
    print()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze XFELBench evaluation results')
    parser.add_argument('--outputs-dir', default='outputs',
                       help='Path to outputs directory (default: outputs)')

    args = parser.parse_args()

    compare_experiments(Path(args.outputs_dir))


if __name__ == '__main__':
    main()
