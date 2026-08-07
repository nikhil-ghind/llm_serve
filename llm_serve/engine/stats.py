"""Rolling engine statistics shared by the scheduler and the metrics endpoint."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable


@dataclass
class Snapshot:
    """A single observation of engine state at time ``t``."""

    t: float
    running: int
    waiting: int
    kv_cache_usage: float
    batch_size: int
    batched_tokens: int


class EngineStatsTracker:
    """Fixed-size window of engine snapshots with simple derived statistics.

    Bounded by construction so a long-running server cannot grow this without
    limit; the window is what the ``/metrics`` gauges and the benchmark's
    "average batch size" figure are computed from.
    """

    def __init__(self, window: int = 1024) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._snapshots: Deque[Snapshot] = deque(maxlen=window)
        self.total_steps = 0
        self.total_batched_tokens = 0

    def record(
        self,
        t: float,
        running: int,
        waiting: int,
        kv_cache_usage: float,
        batch_size: int,
        batched_tokens: int,
    ) -> Snapshot:
        snap = Snapshot(t, running, waiting, kv_cache_usage, batch_size, batched_tokens)
        self._snapshots.append(snap)
        self.total_steps += 1
        self.total_batched_tokens += batched_tokens
        return snap

    def __len__(self) -> int:
        return len(self._snapshots)

    @property
    def snapshots(self) -> list[Snapshot]:
        return list(self._snapshots)

    def _mean(self, values: Iterable[float]) -> float:
        vals = list(values)
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_batch_size(self) -> float:
        return self._mean(s.batch_size for s in self._snapshots)

    @property
    def avg_running(self) -> float:
        return self._mean(s.running for s in self._snapshots)

    @property
    def avg_waiting(self) -> float:
        return self._mean(s.waiting for s in self._snapshots)

    @property
    def avg_kv_usage(self) -> float:
        return self._mean(s.kv_cache_usage for s in self._snapshots)

    @property
    def peak_kv_usage(self) -> float:
        return max((s.kv_cache_usage for s in self._snapshots), default=0.0)

    def steps_per_second(self) -> float:
        """Engine iteration rate over the retained window."""
        if len(self._snapshots) < 2:
            return 0.0
        span = self._snapshots[-1].t - self._snapshots[0].t
        if span <= 0:
            return 0.0
        return (len(self._snapshots) - 1) / span

    def batch_size_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for snap in self._snapshots:
            hist[snap.batch_size] = hist.get(snap.batch_size, 0) + 1
        return dict(sorted(hist.items()))

    def to_dict(self) -> dict[str, float]:
        return {
            "total_steps": float(self.total_steps),
            "total_batched_tokens": float(self.total_batched_tokens),
            "avg_batch_size": self.avg_batch_size,
            "avg_running": self.avg_running,
            "avg_waiting": self.avg_waiting,
            "avg_kv_cache_usage": self.avg_kv_usage,
            "peak_kv_cache_usage": self.peak_kv_usage,
            "steps_per_second": self.steps_per_second(),
        }
