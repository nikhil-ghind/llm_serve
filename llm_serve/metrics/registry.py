"""A tiny Prometheus-compatible metric registry.

``prometheus_client`` is the right tool in production, but the exposition format
is simple enough that implementing it here keeps the metrics layer testable with
zero dependencies and keeps the CPU-only test suite honest about what the
endpoint actually emits. :func:`render` produces text exposition format v0.0.4.
"""

from __future__ import annotations

import threading
from typing import Iterable, Sequence

from .math import INF, cumulative_bucket_counts, histogram_quantile

_LabelKey = tuple[tuple[str, str], ...]


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_label_key(key: _LabelKey) -> str:
    if not key:
        return ""
    inner = ",".join(f'{name}="{_escape(value)}"' for name, value in key)
    return "{" + inner + "}"


def _format_value(value: float) -> str:
    if value == INF:
        return "+Inf"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> None:
        self.name = name
        self.documentation = documentation
        self.labelnames = tuple(labelnames)
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, str] | None) -> _LabelKey:
        labels = labels or {}
        if set(labels) != set(self.labelnames):
            raise ValueError(
                f"metric {self.name!r} expects labels {self.labelnames}, got {tuple(labels)}"
            )
        return tuple((name, str(labels[name])) for name in self.labelnames)

    def samples(self) -> Iterable[tuple[str, _LabelKey, float]]:
        raise NotImplementedError


class Counter(_Metric):
    """Monotonically increasing count."""

    kind = "counter"

    def __init__(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, documentation, labelnames)
        self._values: dict[_LabelKey, float] = {}

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        if amount < 0:
            raise ValueError("counters cannot decrease")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, labels: dict[str, str] | None = None) -> float:
        return self._values.get(self._key(labels), 0.0)

    def samples(self):
        for key, value in self._values.items():
            yield self.name + "_total", key, value


class Gauge(_Metric):
    """A value that goes up and down (queue depth, cache utilization)."""

    kind = "gauge"

    def __init__(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, documentation, labelnames)
        self._values: dict[_LabelKey, float] = {}

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._values[self._key(labels)] = float(value)

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        self.inc(-amount, labels)

    def value(self, labels: dict[str, str] | None = None) -> float:
        return self._values.get(self._key(labels), 0.0)

    def samples(self):
        for key, value in self._values.items():
            yield self.name, key, value


class Histogram(_Metric):
    """Bucketed observations, used for TTFT / ITL / end-to-end latency."""

    kind = "histogram"

    def __init__(
        self,
        name: str,
        documentation: str,
        buckets: Sequence[float],
        labelnames: Sequence[str] = (),
    ) -> None:
        super().__init__(name, documentation, labelnames)
        bounds = list(buckets)
        if not bounds:
            raise ValueError("a histogram needs at least one bucket bound")
        if bounds != sorted(bounds):
            raise ValueError("bucket bounds must be sorted ascending")
        self.bounds = bounds
        self._counts: dict[_LabelKey, list[int]] = {}
        self._sums: dict[_LabelKey, float] = {}

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.bounds) + 1))
            for i, bound in enumerate(self.bounds):
                if value <= bound:
                    counts[i] += 1
            counts[-1] += 1
            self._sums[key] = self._sums.get(key, 0.0) + value

    def count(self, labels: dict[str, str] | None = None) -> int:
        counts = self._counts.get(self._key(labels))
        return counts[-1] if counts else 0

    def sum(self, labels: dict[str, str] | None = None) -> float:
        return self._sums.get(self._key(labels), 0.0)

    def buckets(self, labels: dict[str, str] | None = None) -> list[tuple[float, int]]:
        counts = self._counts.get(self._key(labels)) or [0] * (len(self.bounds) + 1)
        return list(zip(self.bounds + [INF], counts))

    def quantile(self, q: float, labels: dict[str, str] | None = None) -> float:
        return histogram_quantile(self.buckets(labels), q)

    def observe_all(self, values: Iterable[float], labels: dict[str, str] | None = None) -> None:
        for value in values:
            self.observe(value, labels)

    def samples(self):
        for key, counts in self._counts.items():
            for bound, count in zip(self.bounds + [INF], counts):
                yield (
                    self.name + "_bucket",
                    key + (("le", _format_value(bound)),),
                    float(count),
                )
            yield self.name + "_sum", key, self._sums.get(key, 0.0)
            yield self.name + "_count", key, float(counts[-1])


class MetricsRegistry:
    """Holds the metric objects and renders them in Prometheus text format."""

    def __init__(self, namespace: str = "llm_serve") -> None:
        self.namespace = namespace
        self._metrics: dict[str, _Metric] = {}

    def _qualify(self, name: str) -> str:
        return f"{self.namespace}_{name}" if self.namespace else name

    def _add(self, metric: _Metric) -> _Metric:
        if metric.name in self._metrics:
            raise ValueError(f"metric {metric.name!r} is already registered")
        self._metrics[metric.name] = metric
        return metric

    def counter(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> Counter:
        return self._add(Counter(self._qualify(name), documentation, labelnames))  # type: ignore[return-value]

    def gauge(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> Gauge:
        return self._add(Gauge(self._qualify(name), documentation, labelnames))  # type: ignore[return-value]

    def histogram(
        self,
        name: str,
        documentation: str,
        buckets: Sequence[float],
        labelnames: Sequence[str] = (),
    ) -> Histogram:
        return self._add(  # type: ignore[return-value]
            Histogram(self._qualify(name), documentation, buckets, labelnames)
        )

    def get(self, name: str) -> _Metric | None:
        return self._metrics.get(self._qualify(name))

    def render(self) -> str:
        """Prometheus text exposition format (v0.0.4)."""
        lines: list[str] = []
        for metric in self._metrics.values():
            samples = list(metric.samples())
            if not samples:
                continue
            lines.append(f"# HELP {metric.name} {metric.documentation}")
            lines.append(f"# TYPE {metric.name} {metric.kind}")
            for sample_name, key, value in samples:
                lines.append(f"{sample_name}{_format_label_key(key)} {_format_value(value)}")
        return "\n".join(lines) + ("\n" if lines else "")


class ServingMetrics:
    """The metric set the API server exports.

    Names follow the vLLM/OpenAI serving conventions so existing Grafana
    dashboards and alert rules line up without translation.
    """

    def __init__(self, namespace: str = "llm_serve", config=None) -> None:
        from ..config import MetricsConfig

        cfg = config or MetricsConfig()
        self.registry = MetricsRegistry(namespace)
        labels = ("backend", "model")

        self.requests_total = self.registry.counter(
            "requests_total", "Completion requests received.", labels + ("status",)
        )
        self.prompt_tokens_total = self.registry.counter(
            "prompt_tokens_total", "Prompt tokens processed.", labels
        )
        self.generation_tokens_total = self.registry.counter(
            "generation_tokens_total", "Tokens generated.", labels
        )
        self.cached_prompt_tokens_total = self.registry.counter(
            "cached_prompt_tokens_total", "Prompt tokens served from the KV prefix cache.", labels
        )
        self.ttft_seconds = self.registry.histogram(
            "time_to_first_token_seconds",
            "Time from request arrival to the first streamed token.",
            cfg.ttft_buckets_s,
            labels,
        )
        self.itl_seconds = self.registry.histogram(
            "inter_token_latency_seconds",
            "Latency between consecutive streamed tokens.",
            cfg.itl_buckets_s,
            labels,
        )
        self.e2e_seconds = self.registry.histogram(
            "e2e_request_latency_seconds",
            "End-to-end request latency.",
            cfg.e2e_buckets_s,
            labels,
        )
        self.running = self.registry.gauge("num_requests_running", "Requests decoding now.", labels)
        self.waiting = self.registry.gauge("num_requests_waiting", "Requests queued.", labels)
        self.kv_cache_usage = self.registry.gauge(
            "gpu_kv_cache_usage_perc", "Fraction of the KV block pool in use.", labels
        )
        self.prefix_hit_rate = self.registry.gauge(
            "prefix_cache_hit_rate", "Fraction of prompt blocks served from cache.", labels
        )
        self.preemptions_total = self.registry.counter(
            "preemptions_total", "Sequences preempted for lack of KV blocks.", labels
        )

    def observe_result(self, result, backend: str, model: str, status: str = "ok") -> None:
        """Record one finished request."""
        labels = {"backend": backend, "model": model}
        self.requests_total.inc(labels={**labels, "status": status})
        self.prompt_tokens_total.inc(result.prompt_tokens, labels)
        self.generation_tokens_total.inc(result.completion_tokens, labels)
        if result.cached_prompt_tokens:
            self.cached_prompt_tokens_total.inc(result.cached_prompt_tokens, labels)
        if result.ttft_s is not None:
            self.ttft_seconds.observe(result.ttft_s, labels)
        if result.e2e_latency_s is not None:
            self.e2e_seconds.observe(result.e2e_latency_s, labels)
        self.itl_seconds.observe_all(result.inter_token_latencies_s, labels)

    def observe_stats(self, stats, model: str) -> None:
        """Record a backend stats snapshot onto the gauges."""
        labels = {"backend": stats.backend, "model": model}
        self.running.set(stats.running, labels)
        self.waiting.set(stats.waiting, labels)
        self.kv_cache_usage.set(stats.kv_cache_usage, labels)
        self.prefix_hit_rate.set(stats.prefix_cache_hit_rate, labels)

    def render(self) -> str:
        return self.registry.render()
