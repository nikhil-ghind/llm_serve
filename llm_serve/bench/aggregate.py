"""Turn per-request samples into the numbers that go in the report.

Definitions used throughout (they differ between published benchmarks, so they
are pinned here):

* **TTFT** — arrival to first streamed token. Includes queueing, so it degrades
  under load exactly as a user experiences it.
* **ITL** — gap between consecutive streamed tokens, one sample per gap, pooled
  across requests. Pooling (rather than averaging per-request means) keeps long
  generations from being under-weighted.
* **Output throughput** — generated tokens / wall-clock window, counting the
  whole run window rather than the sum of per-request rates.
* **Goodput** — requests per second that met *both* the TTFT and ITL SLOs. A
  server can post excellent throughput while missing the latency budget on a
  third of requests; goodput is what separates the two.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from ..metrics.math import mean, percentile, stdev


@dataclass
class RequestRecord:
    """Per-request measurements captured by the load generator."""

    index: int
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    start_s: float
    end_s: float
    ttft_s: float | None = None
    itls_s: list[float] = field(default_factory=list)
    finish_reason: str = "stop"
    error: str | None = None
    cached_prompt_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and self.completion_tokens > 0

    @property
    def latency_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def mean_itl_s(self) -> float | None:
        return mean(self.itls_s) if self.itls_s else None


@dataclass
class BenchmarkResult:
    """Aggregated result of one benchmark run against one backend."""

    backend: str
    model: str
    concurrency: int
    duration_s: float
    completed: int
    failed: int
    total_prompt_tokens: int
    total_generation_tokens: int
    request_throughput: float
    output_token_throughput: float
    total_token_throughput: float
    ttft_s: dict[str, float]
    itl_s: dict[str, float]
    e2e_s: dict[str, float]
    ttft_slo_s: float = 0.0
    itl_slo_s: float = 0.0
    slo_attainment: float = 0.0
    goodput: float = 0.0
    cached_prompt_tokens: int = 0
    prefix_cache_hit_rate: float = 0.0
    gpu: dict[str, float] = field(default_factory=dict)
    workload: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


_QUANTILES = (50.0, 90.0, 95.0, 99.0)


def _latency_stats(samples: Sequence[float]) -> dict[str, float]:
    if not samples:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    stats = {
        "mean": mean(samples),
        "std": stdev(samples),
        "min": float(min(samples)),
        "max": float(max(samples)),
    }
    for q in _QUANTILES:
        stats[f"p{int(q)}"] = percentile(samples, q)
    return stats


def aggregate(
    records: Sequence[RequestRecord],
    *,
    backend: str,
    model: str,
    concurrency: int,
    duration_s: float,
    ttft_slo_s: float = 1.0,
    itl_slo_s: float = 0.05,
    gpu: dict[str, float] | None = None,
    workload: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Fold per-request records into a :class:`BenchmarkResult`.

    Failed requests are excluded from latency statistics (a request that errored
    at 3 ms would otherwise flatter the p50) but are still counted in ``failed``
    and in the SLO denominator.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")

    good = [r for r in records if r.ok]
    failed = len(records) - len(good)

    ttfts = [r.ttft_s for r in good if r.ttft_s is not None]
    itls = [value for r in good for value in r.itls_s]
    e2es = [r.latency_s for r in good]

    prompt_tokens = sum(r.prompt_tokens for r in good)
    generation_tokens = sum(r.completion_tokens for r in good)
    cached = sum(r.cached_prompt_tokens for r in good)

    met = 0
    for record in good:
        ttft_ok = record.ttft_s is not None and record.ttft_s <= ttft_slo_s
        itl_ok = not record.itls_s or mean(record.itls_s) <= itl_slo_s
        met += 1 if (ttft_ok and itl_ok) else 0
    attainment = met / len(records) if records else 0.0

    return BenchmarkResult(
        backend=backend,
        model=model,
        concurrency=concurrency,
        duration_s=duration_s,
        completed=len(good),
        failed=failed,
        total_prompt_tokens=prompt_tokens,
        total_generation_tokens=generation_tokens,
        request_throughput=len(good) / duration_s,
        output_token_throughput=generation_tokens / duration_s,
        total_token_throughput=(prompt_tokens + generation_tokens) / duration_s,
        ttft_s=_latency_stats(ttfts),
        itl_s=_latency_stats(itls),
        e2e_s=_latency_stats(e2es),
        ttft_slo_s=ttft_slo_s,
        itl_slo_s=itl_slo_s,
        slo_attainment=attainment,
        goodput=met / duration_s,
        cached_prompt_tokens=cached,
        prefix_cache_hit_rate=(cached / prompt_tokens) if prompt_tokens else 0.0,
        gpu=dict(gpu or {}),
        workload=dict(workload or {}),
        metadata=dict(metadata or {}),
    )


def cost_per_million_tokens(result: BenchmarkResult, gpu_hourly_usd: float) -> float:
    """Serving cost at the measured throughput.

    The headline number for capacity planning: a backend that is 30% faster on
    the same GPU is 30% cheaper per token, and this converts one into the other.
    """
    if gpu_hourly_usd < 0:
        raise ValueError("gpu_hourly_usd must be >= 0")
    if result.output_token_throughput <= 0:
        return float("inf")
    tokens_per_hour = result.output_token_throughput * 3600.0
    return gpu_hourly_usd / (tokens_per_hour / 1_000_000.0)


def speedup(result: BenchmarkResult, baseline: BenchmarkResult) -> dict[str, float]:
    """Ratios against a baseline run: >1 is better for throughput, <1 for latency."""
    out: dict[str, float] = {}
    if baseline.output_token_throughput > 0:
        out["throughput_x"] = result.output_token_throughput / baseline.output_token_throughput
    if baseline.ttft_s.get("p50"):
        out["ttft_p50_x"] = result.ttft_s["p50"] / baseline.ttft_s["p50"]
    if baseline.ttft_s.get("p99"):
        out["ttft_p99_x"] = result.ttft_s["p99"] / baseline.ttft_s["p99"]
    if baseline.itl_s.get("p50"):
        out["itl_p50_x"] = result.itl_s["p50"] / baseline.itl_s["p50"]
    return out


def load_results(paths: Sequence[str]) -> list[BenchmarkResult]:
    """Read result JSON files written by ``scripts/run_benchmark.py``."""
    results = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            results.extend(BenchmarkResult.from_dict(item) for item in data)
        else:
            results.append(BenchmarkResult.from_dict(data))
    return results
