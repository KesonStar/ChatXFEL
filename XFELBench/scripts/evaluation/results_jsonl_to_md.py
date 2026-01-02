#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert XFELBench generation results JSONL to a well-formatted Markdown report.

Target input example:
  XFELBench/outputs/<run>/results.jsonl

Each JSONL line typically contains:
  - question_id, question, answer
  - sources: [{rank,title,doi,journal,year,page,start_index,content,content_length}, ...]
  - retrieval_stats, rewritten_query, generation_time, timestamp, metadata

This script will:
  - Parse JSONL safely
  - Optionally read sibling summary.json
  - Generate readable Markdown with consistent headings and tables
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_iso(ts: str) -> str:
    ts = (ts or "").strip()
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat()
    except Exception:
        return ts


def _md_escape_inline(text: str) -> str:
    # Minimal escaping; keep report readable. Avoid breaking table pipes.
    return (text or "").replace("|", "\\|").replace("\r\n", "\n").replace("\r", "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {idx} of {path}: {e}") from e
            if isinstance(obj, dict):
                items.append(obj)
            else:
                raise ValueError(f"Line {idx} is not a JSON object: {path}")
    return items


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to read JSON: {path} ({e})")
        return None


def _fmt_num(v: Any, digits: int = 3) -> str:
    f = _safe_float(v)
    if f is None:
        return ""
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    s = f"{f:.{digits}f}".rstrip("0").rstrip(".")
    return s


def _render_summary_md(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("### Summary")
    lines.append("")
    lines.append("| item | value |")
    lines.append("| --- | ---: |")
    if "experiment_name" in summary:
        lines.append(f"| experiment_name | {_md_escape_inline(str(summary.get('experiment_name')))} |")
    for k in ("total_questions", "completed_questions", "failed_questions"):
        if k in summary:
            lines.append(f"| {k} | {_safe_int(summary.get(k)) if summary.get(k) is not None else ''} |")
    if "average_generation_time" in summary:
        lines.append(f"| average_generation_time (s) | {_fmt_num(summary.get('average_generation_time'), digits=3)} |")
    ts = _format_iso(str(summary.get("timestamp", "")).strip())
    if ts:
        lines.append(f"| timestamp | {ts} |")
    lines.append("")
    return "\n".join(lines)


def _anchor(text: str) -> str:
    # Simple GitHub-style anchor approximation
    t = (text or "").strip().lower()
    t = t.replace(" ", "-")
    return "".join(ch for ch in t if ch.isalnum() or ch in "-_")


def _render_sources_table(sources: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("| rank | title | year | doi | journal | page | len |")
    lines.append("| ---: | --- | ---: | --- | --- | ---: | ---: |")
    for s in sources:
        rank = _safe_int(s.get("rank"))
        title = _md_escape_inline(str(s.get("title", "")).strip())
        year = s.get("year")
        doi = _md_escape_inline(str(s.get("doi", "")).strip())
        journal = _md_escape_inline(str(s.get("journal", "")).strip())
        page = s.get("page")
        clen = _safe_int(s.get("content_length"))
        lines.append(
            f"| {rank if rank is not None else ''} | {title} | {year if year is not None else ''} | {doi} | {journal} | {page if page is not None else ''} | {clen if clen is not None else ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_source_contents(
    sources: List[Dict[str, Any]],
    *,
    max_source_chars: int,
) -> str:
    lines: List[str] = []
    for s in sources:
        rank = _safe_int(s.get("rank"))
        title = str(s.get("title", "")).strip()
        content = str(s.get("content", "") or "")
        if max_source_chars >= 0:
            content = content[:max_source_chars]
            if len(str(s.get("content", "") or "")) > len(content):
                content = content.rstrip() + "\n... (truncated) ..."
        lines.append(f"#### Source {rank if rank is not None else ''}: {_md_escape_inline(title)}")
        lines.append("")
        lines.append("```")
        lines.append(content.rstrip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _render_item_md(
    item: Dict[str, Any],
    *,
    include_source_content: bool,
    max_source_chars: int,
) -> str:
    qid = str(item.get("question_id", "")).strip() or "UNKNOWN_ID"
    question = _md_escape_inline(str(item.get("question", "")).strip())
    answer = str(item.get("answer", "")).rstrip()

    sources = item.get("sources") or []
    if not isinstance(sources, list):
        sources = []

    retrieval_stats = item.get("retrieval_stats") or {}
    if not isinstance(retrieval_stats, dict):
        retrieval_stats = {}

    meta = item.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}

    rewritten_query = str(item.get("rewritten_query", "") or "").strip()
    gen_time = _safe_float(item.get("generation_time"))
    ts = _format_iso(str(item.get("timestamp", "")).strip())

    lines: List[str] = []
    lines.append(f"### {qid}")
    lines.append("")
    if question:
        lines.append("**question**")
        lines.append("")
        lines.append(f"> {question}")
        lines.append("")

    meta_parts: List[str] = []
    if ts:
        meta_parts.append(f"timestamp: `{ts}`")
    if gen_time is not None:
        meta_parts.append(f"generation_time: `{_fmt_num(gen_time, digits=3)}s`")
    if meta_parts:
        lines.append("- **meta**: " + ", ".join(meta_parts))
        lines.append("")

    if rewritten_query:
        lines.append("**rewritten_query**")
        lines.append("")
        lines.append(f"> {_md_escape_inline(rewritten_query)}")
        lines.append("")

    lines.append("**answer**")
    lines.append("")
    lines.append(answer if answer else "")
    lines.append("")

    # Retrieval stats + metadata
    if retrieval_stats or meta:
        lines.append("#### Stats")
        lines.append("")
        lines.append("| key | value |")
        lines.append("| --- | --- |")
        for k in ("num_sources", "total_context_length", "unique_papers"):
            if k in retrieval_stats:
                lines.append(f"| {k} | {_md_escape_inline(str(retrieval_stats.get(k)))} |")
        yr = retrieval_stats.get("year_range")
        if isinstance(yr, dict):
            lines.append(f"| year_range | {_md_escape_inline(str(yr))} |")
        for k, v in sorted(meta.items()):
            if v is None:
                continue
            lines.append(f"| metadata.{_md_escape_inline(str(k))} | {_md_escape_inline(str(v))} |")
        lines.append("")

    # Sources
    if sources:
        lines.append("#### Sources")
        lines.append("")
        lines.append(_render_sources_table(sources))
        if include_source_content:
            lines.append(_render_source_contents(sources, max_source_chars=max_source_chars))

    return "\n".join(lines).rstrip() + "\n"


def build_markdown(
    items: List[Dict[str, Any]],
    *,
    title: str,
    input_path: Path,
    summary: Optional[Dict[str, Any]],
    include_source_content: bool,
    max_source_chars: int,
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

    lines.append("### Contents")
    lines.append("")
    for item in items:
        qid = str(item.get("question_id", "")).strip() or "UNKNOWN_ID"
        lines.append(f"- [{qid}](#{_anchor(qid)})")
    lines.append("")

    lines.append("### Details")
    lines.append("")
    for item in items:
        lines.append(
            _render_item_md(
                item,
                include_source_content=include_source_content,
                max_source_chars=max_source_chars,
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert XFELBench results.jsonl into a Markdown report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python XFELBench/scripts/evaluation/results_jsonl_to_md.py \\
    --input XFELBench/outputs/20260102_190926_5_full_features/results.jsonl

  # Include source chunk content (truncated to 800 chars each)
  python XFELBench/scripts/evaluation/results_jsonl_to_md.py \\
    --input XFELBench/outputs/20260102_190926_5_full_features/results.jsonl \\
    --include-source-content --max-source-chars 800
        """.strip(),
    )
    parser.add_argument("--input", required=True, help="Path to results.jsonl")
    parser.add_argument("--output", default=None, help="Path to output .md (default: alongside input, same name with .md)")
    parser.add_argument("--title", default=None, help="Report title (default: derived from input directory name)")
    parser.add_argument("--no-summary", action="store_true", help="Do not include sibling summary.json if present")
    parser.add_argument("--include-source-content", action="store_true", help="Include each source's chunk content in Markdown")
    parser.add_argument(
        "--max-source-chars",
        type=int,
        default=800,
        help="Max chars per source content when --include-source-content is set. Use -1 for no truncation.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = args.title or f"Results report: {input_path.parent.name}"

    items = _read_jsonl(input_path)

    summary = None
    if not args.no_summary:
        summary_path = input_path.parent / "summary.json"
        summary = _load_json(summary_path)

    md = build_markdown(
        items,
        title=title,
        input_path=input_path,
        summary=summary,
        include_source_content=args.include_source_content,
        max_source_chars=args.max_source_chars,
    )
    with output_path.open("w", encoding="utf-8") as f:
        f.write(md)

    print(f"[INFO] Wrote Markdown report: {output_path}")


if __name__ == "__main__":
    main()


