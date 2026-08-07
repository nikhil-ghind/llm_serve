#!/usr/bin/env python3
"""Build the cross-backend comparison report from saved benchmark results.

    python scripts/compare_backends.py benchmarks/results/*.json \
        --output benchmarks/comparison.md --csv benchmarks/comparison.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_serve.bench.aggregate import load_results  # noqa: E402
from llm_serve.bench.report import (  # noqa: E402
    DEFAULT_GPU_HOURLY_USD,
    render_csv,
    render_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare benchmark results across backends.")
    parser.add_argument("results", nargs="+", help="result JSON files or globs")
    parser.add_argument("--output", default="benchmarks/comparison.md")
    parser.add_argument("--csv", default=None, help="also write a CSV here")
    parser.add_argument("--baseline", default="vllm", help="backend to compute speedups against")
    parser.add_argument("--title", default="Cross-backend inference serving comparison")
    parser.add_argument("--gpu-hourly-usd", type=float, default=DEFAULT_GPU_HOURLY_USD)
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="keep only the newest run per (backend, concurrency) pair",
    )
    return parser


def expand(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif os.path.exists(pattern):
            paths.append(pattern)
    # A sweep writes both per-level files and one combined file; drop duplicates.
    return sorted(set(paths))


def deduplicate(results):
    """Keep the newest result for each (backend, concurrency) pair."""
    latest: dict[tuple[str, int], object] = {}
    for result in results:
        key = (result.backend, result.concurrency)
        current = latest.get(key)
        if current is None or str(result.metadata.get("timestamp", "")) >= str(
            current.metadata.get("timestamp", "")
        ):
            latest[key] = result
    return list(latest.values())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = expand(args.results)
    if not paths:
        print("no result files matched", file=sys.stderr)
        return 1

    results = load_results(paths)
    # Combined sweep files repeat the per-level records; collapse them either way.
    results = deduplicate(results)
    if not results:
        print("no usable results found", file=sys.stderr)
        return 1

    markdown = render_markdown(
        results,
        title=args.title,
        baseline=args.baseline,
        gpu_hourly_usd=args.gpu_hourly_usd,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    print(f"wrote {args.output}  ({len(results)} run(s) from {len(paths)} file(s))")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as fh:
            fh.write(render_csv(results))
        print(f"wrote {args.csv}")

    print()
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
