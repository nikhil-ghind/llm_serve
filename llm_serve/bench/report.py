"""Cross-backend comparison report.

Renders a set of :class:`~llm_serve.bench.aggregate.BenchmarkResult` objects into
Markdown (for the repo) and CSV (for a spreadsheet), including a ranking and the
derived cost per million tokens. Pure string formatting so the output shape is
unit-tested rather than eyeballed.
"""

from __future__ import annotations

import csv
import io
from typing import Callable, Sequence

from .aggregate import BenchmarkResult, cost_per_million_tokens, speedup

#: On-demand hourly price of the GPU the run was measured on.
DEFAULT_GPU_HOURLY_USD = 3.67  # single A100 80GB, on-demand

_COLUMNS: list[tuple[str, Callable[[BenchmarkResult], str]]] = [
    ("Backend", lambda r: r.backend),
    ("Conc.", lambda r: str(r.concurrency)),
    ("Req/s", lambda r: f"{r.request_throughput:.2f}"),
    ("Output tok/s", lambda r: f"{r.output_token_throughput:.1f}"),
    ("Total tok/s", lambda r: f"{r.total_token_throughput:.1f}"),
    ("TTFT p50 (ms)", lambda r: f"{r.ttft_s['p50'] * 1000:.1f}"),
    ("TTFT p99 (ms)", lambda r: f"{r.ttft_s['p99'] * 1000:.1f}"),
    ("ITL p50 (ms)", lambda r: f"{r.itl_s['p50'] * 1000:.1f}"),
    ("E2E p99 (s)", lambda r: f"{r.e2e_s['p99']:.2f}"),
    ("SLO %", lambda r: f"{r.slo_attainment * 100:.1f}"),
    ("GPU %", lambda r: f"{r.gpu.get('gpu_mean_sm_util_pct', 0.0):.0f}"),
]


def _markdown_table(rows: Sequence[Sequence[str]], header: Sequence[str]) -> str:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(lines)


def comparison_table(results: Sequence[BenchmarkResult]) -> str:
    """The main results table, one row per run."""
    header = [name for name, _ in _COLUMNS]
    rows = [[fmt(r) for _, fmt in _COLUMNS] for r in results]
    return _markdown_table(rows, header)


def cost_table(
    results: Sequence[BenchmarkResult], gpu_hourly_usd: float = DEFAULT_GPU_HOURLY_USD
) -> str:
    header = ["Backend", "Conc.", "Output tok/s", "$/1M output tokens"]
    rows = []
    for result in results:
        cost = cost_per_million_tokens(result, gpu_hourly_usd)
        rows.append(
            [
                result.backend,
                str(result.concurrency),
                f"{result.output_token_throughput:.1f}",
                "n/a" if cost == float("inf") else f"${cost:.3f}",
            ]
        )
    return _markdown_table(rows, header)


def rank(results: Sequence[BenchmarkResult], key: str = "output_token_throughput") -> list[BenchmarkResult]:
    """Order results best-first for the given metric."""
    lower_is_better = key.startswith("ttft") or key.startswith("itl") or key.startswith("e2e")

    def _value(result: BenchmarkResult) -> float:
        if key.endswith(("_p50", "_p90", "_p95", "_p99")):
            family, _, quantile = key.rpartition("_")
            stats = getattr(result, f"{family}_s", {})
            return stats.get(quantile, 0.0)
        return float(getattr(result, key, 0.0))

    return sorted(results, key=_value, reverse=not lower_is_better)


def best_by_concurrency(results: Sequence[BenchmarkResult]) -> str:
    """Which backend wins at each concurrency level."""
    levels = sorted({r.concurrency for r in results})
    rows = []
    for level in levels:
        at_level = [r for r in results if r.concurrency == level]
        fastest = rank(at_level)[0]
        lowest_ttft = rank(at_level, "ttft_p50")[0]
        rows.append(
            [
                str(level),
                f"{fastest.backend} ({fastest.output_token_throughput:.0f} tok/s)",
                f"{lowest_ttft.backend} ({lowest_ttft.ttft_s['p50'] * 1000:.0f} ms)",
            ]
        )
    return _markdown_table(rows, ["Concurrency", "Best throughput", "Best TTFT p50"])


def render_markdown(
    results: Sequence[BenchmarkResult],
    title: str = "Cross-backend inference serving comparison",
    baseline: str | None = "vllm",
    gpu_hourly_usd: float = DEFAULT_GPU_HOURLY_USD,
) -> str:
    """Full report: setup, results, ranking, cost and speedups."""
    if not results:
        return f"# {title}\n\nNo results.\n"

    ordered = sorted(results, key=lambda r: (r.backend, r.concurrency))
    first = ordered[0]
    devices = first.metadata.get("gpu_devices") or []
    workload = first.workload

    lines = [f"# {title}", ""]
    lines.append(f"- **Model**: `{first.model}`")
    if devices:
        lines.append(f"- **GPU**: {', '.join(devices)}")
    if workload:
        lines.append(
            "- **Workload**: {n:.0f} requests, ~{inp:.0f} input tokens, "
            "~{out:.0f} output tokens, {prefix:.0f}-token shared prefix".format(
                n=workload.get("num_requests", 0),
                inp=workload.get("mean_input_tokens", 0),
                out=workload.get("mean_output_tokens", 0),
                prefix=workload.get("shared_prefix_tokens", 0),
            )
        )
    lines.append(
        f"- **Arrival**: {first.metadata.get('arrival', 'closed_loop')}"
        + (f" at {first.metadata['request_rate']} req/s" if first.metadata.get("request_rate") else "")
    )
    lines.append(f"- **SLO**: TTFT <= {first.ttft_slo_s:.2f}s, mean ITL <= {first.itl_slo_s * 1000:.0f}ms")
    lines.append("")

    lines += ["## Results", "", comparison_table(ordered), ""]
    lines += [
        "## Ranking by output throughput",
        "",
        _markdown_table(
            [
                [str(i + 1), r.backend, str(r.concurrency), f"{r.output_token_throughput:.1f}"]
                for i, r in enumerate(rank(results))
            ],
            ["#", "Backend", "Concurrency", "Output tok/s"],
        ),
        "",
    ]
    if len({r.concurrency for r in results}) > 1:
        lines += ["## Best backend per concurrency level", "", best_by_concurrency(results), ""]

    lines += [
        "## Serving cost",
        "",
        f"At ${gpu_hourly_usd:.2f}/GPU-hour:",
        "",
        cost_table(ordered, gpu_hourly_usd),
        "",
    ]

    base_results = [r for r in results if r.backend == baseline]
    if baseline and base_results:
        base = rank(base_results)[0]
        rows = []
        for result in ordered:
            if result.backend == baseline:
                continue
            ratios = speedup(result, base)
            rows.append(
                [
                    result.backend,
                    f"{ratios.get('throughput_x', 0):.2f}x",
                    f"{ratios.get('ttft_p50_x', 0):.2f}x",
                    f"{ratios.get('ttft_p99_x', 0):.2f}x",
                    f"{ratios.get('itl_p50_x', 0):.2f}x",
                ]
            )
        if rows:
            lines += [
                f"## Relative to `{baseline}`",
                "",
                "Throughput above 1.00x is better; latency ratios below 1.00x are better.",
                "",
                _markdown_table(
                    rows, ["Backend", "Throughput", "TTFT p50", "TTFT p99", "ITL p50"]
                ),
                "",
            ]

    if any(r.prefix_cache_hit_rate for r in results):
        lines += [
            "## KV prefix cache",
            "",
            _markdown_table(
                [
                    [r.backend, str(r.concurrency), f"{r.prefix_cache_hit_rate * 100:.1f}%"]
                    for r in ordered
                ],
                ["Backend", "Concurrency", "Prompt tokens served from cache"],
            ),
            "",
        ]
    return "\n".join(lines) + "\n"


def render_csv(results: Sequence[BenchmarkResult]) -> str:
    """Flat CSV for spreadsheets and plotting."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "backend",
            "model",
            "concurrency",
            "duration_s",
            "completed",
            "failed",
            "request_throughput",
            "output_token_throughput",
            "total_token_throughput",
            "ttft_p50_s",
            "ttft_p90_s",
            "ttft_p99_s",
            "itl_p50_s",
            "itl_p99_s",
            "e2e_p50_s",
            "e2e_p99_s",
            "slo_attainment",
            "goodput",
            "prefix_cache_hit_rate",
            "gpu_mean_sm_util_pct",
        ]
    )
    for r in sorted(results, key=lambda r: (r.backend, r.concurrency)):
        writer.writerow(
            [
                r.backend,
                r.model,
                r.concurrency,
                f"{r.duration_s:.3f}",
                r.completed,
                r.failed,
                f"{r.request_throughput:.4f}",
                f"{r.output_token_throughput:.3f}",
                f"{r.total_token_throughput:.3f}",
                f"{r.ttft_s['p50']:.5f}",
                f"{r.ttft_s['p90']:.5f}",
                f"{r.ttft_s['p99']:.5f}",
                f"{r.itl_s['p50']:.5f}",
                f"{r.itl_s['p99']:.5f}",
                f"{r.e2e_s['p50']:.5f}",
                f"{r.e2e_s['p99']:.5f}",
                f"{r.slo_attainment:.4f}",
                f"{r.goodput:.4f}",
                f"{r.prefix_cache_hit_rate:.4f}",
                f"{r.gpu.get('gpu_mean_sm_util_pct', 0.0):.2f}",
            ]
        )
    return buffer.getvalue()
