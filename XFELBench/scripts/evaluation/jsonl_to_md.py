#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert XFELBench evaluation JSONL to a well-formatted Markdown report.

This script is designed for files like:
  XFELBench/evaluations/<run>/evaluation_results.jsonl

It will:
- Parse each JSON line
- Optionally load sibling evaluation_summary.json
- Generate a readable Markdown report with consistent structure
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Metric:
    name: str
    score: Optional[float]
    reasoning: str
    key_errors: List[str]
    unsupported_claims: List[str]
    missing_aspects: List[str]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to read JSON: {path} ({e})")
        return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {idx} of {path}: {e}") from e
    return items


def _parse_metric(name: str, obj: Dict[str, Any]) -> Metric:
    return Metric(
        name=name,
        score=_safe_float(obj.get("score")),
        reasoning=str(obj.get("reasoning", "")).strip(),
        key_errors=[str(x) for x in (obj.get("key_errors") or [])],
        unsupported_claims=[str(x) for x in (obj.get("unsupported_claims") or [])],
        missing_aspects=[str(x) for x in (obj.get("missing_aspects") or [])],
    )


def _extract_metrics(eval_obj: Dict[str, Any]) -> Tuple[List[Metric], Optional[float]]:
    metrics: List[Metric] = []
    for k in ("factual_accuracy", "groundedness", "coverage_specificity"):
        if k in eval_obj and isinstance(eval_obj[k], dict):
            pretty = {
                "factual_accuracy": "Factual accuracy",
                "groundedness": "Groundedness",
                "coverage_specificity": "Coverage & specificity",
            }[k]
            metrics.append(_parse_metric(pretty, eval_obj[k]))
    avg = _safe_float(eval_obj.get("average_score"))
    return metrics, avg


def _md_escape_inline(text: str) -> str:
    # Minimal escaping; keep report readable. Avoid breaking table pipes.
    return text.replace("|", "\\|").replace("\r\n", "\n").replace("\r", "\n")


def _format_iso(ts: str) -> str:
    ts = (ts or "").strip()
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat()
    except Exception:
        return ts


def _render_summary_md(summary: Dict[str, Any]) -> str:
    avg = (summary or {}).get("average_scores") or {}
    lines: List[str] = []
    lines.append("### Summary")
    lines.append("")
    lines.append("| item | value |")
    lines.append("| --- | ---: |")
    lines.append(f"| total_evaluated | {avg_int(summary.get('total_evaluated'))} |")
    lines.append(f"| factual_accuracy (avg) | {fmt_score(avg.get('factual_accuracy'))} |")
    lines.append(f"| groundedness (avg) | {fmt_score(avg.get('groundedness'))} |")
    lines.append(f"| coverage_specificity (avg) | {fmt_score(avg.get('coverage_specificity'))} |")
    lines.append(f"| overall (avg) | {fmt_score(avg.get('overall'))} |")
    ts = _format_iso(str(summary.get("timestamp", "")).strip())
    if ts:
        lines.append(f"| timestamp | {ts} |")
    lines.append("")
    return "\n".join(lines)


def avg_int(v: Any) -> str:
    if v is None:
        return ""
    try:
        return str(int(v))
    except Exception:
        return str(v)


def fmt_score(v: Any) -> str:
    f = _safe_float(v)
    if f is None:
        return ""
    # keep stable formatting
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return f"{f:.3f}".rstrip("0").rstrip(".")


def _render_metric_block(metric: Metric, *, show_lists: bool) -> str:
    lines: List[str] = []
    lines.append(f"#### {metric.name}")
    lines.append("")
    if metric.score is not None:
        lines.append(f"- **score**: {fmt_score(metric.score)}")
    if metric.reasoning:
        lines.append("")
        lines.append("**reasoning**")
        lines.append("")
        lines.append(metric.reasoning)
        lines.append("")

    if show_lists:
        if metric.key_errors:
            lines.append("**key_errors**")
            lines.append("")
            for e in metric.key_errors:
                lines.append(f"- {e}")
            lines.append("")
        if metric.unsupported_claims:
            lines.append("**unsupported_claims**")
            lines.append("")
            for c in metric.unsupported_claims:
                lines.append(f"- {c}")
            lines.append("")
        if metric.missing_aspects:
            lines.append("**missing_aspects**")
            lines.append("")
            for m in metric.missing_aspects:
                lines.append(f"- {m}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_item_md(item: Dict[str, Any], *, show_lists: bool) -> str:
    qid = str(item.get("question_id", "")).strip() or "UNKNOWN_ID"
    question = _md_escape_inline(str(item.get("question", "")).strip())
    has_gt = item.get("has_ground_truth", None)
    ts = _format_iso(str(item.get("timestamp", "")).strip())

    eval_obj = item.get("evaluation") or {}
    if not isinstance(eval_obj, dict):
        eval_obj = {}
    metrics, avg = _extract_metrics(eval_obj)

    lines: List[str] = []
    lines.append(f"### {qid}")
    lines.append("")
    if question:
        lines.append("**question**")
        lines.append("")
        lines.append(f"> {_md_escape_inline(question)}")
        lines.append("")

    meta_parts: List[str] = []
    if has_gt is not None:
        meta_parts.append(f"has_ground_truth: `{has_gt}`")
    if avg is not None:
        meta_parts.append(f"average_score: `{fmt_score(avg)}`")
    if ts:
        meta_parts.append(f"timestamp: `{ts}`")
    if meta_parts:
        lines.append("- **meta**: " + ", ".join(meta_parts))
        lines.append("")

    # Quick score table
    if metrics:
        lines.append("#### Scores")
        lines.append("")
        lines.append("| metric | score |")
        lines.append("| --- | ---: |")
        for m in metrics:
            lines.append(f"| {m.name} | {fmt_score(m.score)} |")
        if avg is not None:
            lines.append(f"| Average | {fmt_score(avg)} |")
        lines.append("")

    # Detail blocks
    for m in metrics:
        lines.append(_render_metric_block(m, show_lists=show_lists))

    return "\n".join(lines).rstrip() + "\n"


def build_markdown(
    items: List[Dict[str, Any]],
    *,
    title: str,
    input_path: Path,
    summary: Optional[Dict[str, Any]],
    show_lists: bool,
) -> str:
    lines: List[str] = []

    lines.append(f"## {title}")
    lines.append("")
    lines.append(f"- **source**: `{input_path.as_posix()}`")
    lines.append(f"- **generated_at**: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- **count**: `{len(items)}`")
    lines.append("")

    if summary:
        lines.append(_render_summary_md(summary))

    # TOC
    lines.append("### Contents")
    lines.append("")
    for item in items:
        qid = str(item.get("question_id", "")).strip() or "UNKNOWN_ID"
        anchor = qid.lower().replace(" ", "-")
        lines.append(f"- [{qid}](#{anchor})")
    lines.append("")

    lines.append("### Details")
    lines.append("")
    for item in items:
        lines.append(_render_item_md(item, show_lists=show_lists))

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert XFELBench evaluation_results.jsonl into a Markdown report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python XFELBench/scripts/evaluation/jsonl_to_md.py \\
    --input XFELBench/evaluations/20260102_190926_5_full_features/evaluation_results.jsonl

  python XFELBench/scripts/evaluation/jsonl_to_md.py \\
    --input /abs/path/evaluation_results.jsonl \\
    --output /abs/path/evaluation_results.md \\
    --no-lists
        """.strip(),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to evaluation_results.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output .md (default: alongside input, same name with .md)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Report title (default: derived from input directory name)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not include sibling evaluation_summary.json if present",
    )
    parser.add_argument(
        "--no-lists",
        action="store_true",
        help="Do not include key_errors/unsupported_claims/missing_aspects lists",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".jsonl":
        print(f"[WARNING] Input does not end with .jsonl: {input_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = args.title
    if not title:
        # Prefer the run directory name if this looks like an XFELBench evaluations file
        title = f"Evaluation report: {input_path.parent.name}"

    items = _read_jsonl(input_path)

    summary = None
    if not args.no_summary:
        summary_path = input_path.parent / "evaluation_summary.json"
        summary = _load_json(summary_path)

    md = build_markdown(
        items,
        title=title,
        input_path=input_path,
        summary=summary,
        show_lists=not args.no_lists,
    )
    with output_path.open("w", encoding="utf-8") as f:
        f.write(md)

    print(f"[INFO] Wrote Markdown report: {output_path}")


if __name__ == "__main__":
    main()


