"""Latency and throughput math, dependency-free.

Everything the ``/metrics`` endpoint and the benchmark reports need: percentiles
over raw samples, Prometheus-style histogram bucketing, quantile estimation from
bucket counts, and rate helpers. Kept out of the Prometheus client library so the
same functions can be unit-tested on CPU and reused by the offline benchmark
aggregator.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

INF = float("inf")


def percentile(samples: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100].

    Matches ``numpy.percentile`` with the default 'linear' method, so benchmark
    numbers line up with whatever anyone re-computes in a notebook.
    """
    if not samples:
        raise ValueError("percentile of an empty sample set is undefined")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[int(pos)])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def percentiles(samples: Sequence[float], qs: Iterable[float]) -> dict[str, float]:
    """Several percentiles at once, keyed ``p50``/``p90``/``p99``…"""
    ordered = sorted(samples)
    out: dict[str, float] = {}
    for q in qs:
        label = f"p{q:g}".replace(".", "_")
        out[label] = percentile(ordered, q)
    return out


def mean(samples: Sequence[float]) -> float:
    return sum(samples) / len(samples) if samples else 0.0


def stdev(samples: Sequence[float]) -> float:
    """Population standard deviation (0.0 for fewer than two samples)."""
    n = len(samples)
    if n < 2:
        return 0.0
    mu = mean(samples)
    return math.sqrt(sum((x - mu) ** 2 for x in samples) / n)


def cumulative_bucket_counts(
    samples: Iterable[float], bounds: Sequence[float]
) -> list[tuple[float, int]]:
    """Prometheus ``le`` buckets: cumulative counts, with a final ``+Inf``."""
    ordered_bounds = list(bounds)
    if ordered_bounds != sorted(ordered_bounds):
        raise ValueError("bucket bounds must be sorted ascending")
    counts = [0] * (len(ordered_bounds) + 1)
    total = 0
    for value in samples:
        total += 1
        for i, bound in enumerate(ordered_bounds):
            if value <= bound:
                counts[i] += 1
        counts[-1] += 1
    del total
    return list(zip(list(ordered_bounds) + [INF], counts))


def histogram_quantile(buckets: Sequence[tuple[float, int]], q: float) -> float:
    """Estimate a quantile from cumulative ``(le, count)`` buckets.

    Same interpolation rule as PromQL's ``histogram_quantile``: locate the bucket
    containing the target rank and interpolate linearly inside it. Returns the
    largest finite bound when the target falls in the ``+Inf`` bucket, since the
    true maximum is unknowable from bucket counts alone.
    """
    if not buckets:
        raise ValueError("no buckets")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    total = buckets[-1][1]
    if total == 0:
        return 0.0
    rank = q * total
    prev_bound = 0.0
    prev_count = 0
    for bound, count in buckets:
        if count >= rank:
            if math.isinf(bound):
                return prev_bound
            if count == prev_count:
                return bound
            frac = (rank - prev_count) / (count - prev_count)
            return prev_bound + (bound - prev_bound) * frac
        prev_bound = 0.0 if math.isinf(bound) else bound
        prev_count = count
    return prev_bound


def rate(delta_count: float, delta_seconds: float) -> float:
    """Per-second rate; 0.0 for a non-positive interval."""
    if delta_seconds <= 0:
        return 0.0
    return delta_count / delta_seconds


def throughput(total_tokens: int, elapsed_s: float) -> float:
    """Tokens per second over a completed window."""
    return rate(float(total_tokens), elapsed_s)


def goodput(latencies: Sequence[float], slo_s: float, elapsed_s: float) -> float:
    """Requests per second that met an SLO — the number that actually matters.

    Raw throughput hides the case where a server is fast on average but blows the
    latency budget for a third of requests; goodput counts only the good ones.
    """
    if elapsed_s <= 0:
        return 0.0
    good = sum(1 for latency in latencies if latency <= slo_s)
    return good / elapsed_s


def slo_attainment(latencies: Sequence[float], slo_s: float) -> float:
    """Fraction of requests meeting the SLO, in [0, 1]."""
    if not latencies:
        return 0.0
    return sum(1 for latency in latencies if latency <= slo_s) / len(latencies)
