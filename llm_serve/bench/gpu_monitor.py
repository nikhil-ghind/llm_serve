"""GPU utilization sampling via NVML.

Throughput numbers without utilization are ambiguous: a backend at 900 tok/s and
55% SM utilization is CPU- or scheduler-bound and has headroom; the same
throughput at 98% is genuinely GPU-bound. This samples SM utilization and memory
use on a background thread for the duration of a run.

``pynvml`` is imported lazily and its absence is not an error — the monitor
degrades to a no-op so benchmarks still run on a machine without NVIDIA drivers.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("llm_serve.bench.gpu")


@dataclass
class GPUSample:
    t: float
    device: int
    sm_util_pct: float
    memory_used_mb: float
    memory_total_mb: float
    power_w: float = 0.0

    @property
    def memory_util_pct(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return 100.0 * self.memory_used_mb / self.memory_total_mb


@dataclass
class GPUStats:
    available: bool = False
    device_count: int = 0
    device_names: list[str] = field(default_factory=list)
    samples: int = 0
    mean_sm_util_pct: float = 0.0
    max_sm_util_pct: float = 0.0
    mean_memory_used_mb: float = 0.0
    max_memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    mean_power_w: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "gpu_available": float(self.available),
            "gpu_device_count": float(self.device_count),
            "gpu_samples": float(self.samples),
            "gpu_mean_sm_util_pct": self.mean_sm_util_pct,
            "gpu_max_sm_util_pct": self.max_sm_util_pct,
            "gpu_mean_memory_used_mb": self.mean_memory_used_mb,
            "gpu_max_memory_used_mb": self.max_memory_used_mb,
            "gpu_memory_total_mb": self.memory_total_mb,
            "gpu_mean_power_w": self.mean_power_w,
        }


def summarize(samples: list[GPUSample]) -> GPUStats:
    """Aggregate raw samples. Pure function, testable without a GPU."""
    if not samples:
        return GPUStats(available=False)
    devices = {s.device for s in samples}
    sm = [s.sm_util_pct for s in samples]
    memory = [s.memory_used_mb for s in samples]
    power = [s.power_w for s in samples]
    return GPUStats(
        available=True,
        device_count=len(devices),
        samples=len(samples),
        mean_sm_util_pct=sum(sm) / len(sm),
        max_sm_util_pct=max(sm),
        mean_memory_used_mb=sum(memory) / len(memory),
        max_memory_used_mb=max(memory),
        memory_total_mb=max(s.memory_total_mb for s in samples),
        mean_power_w=sum(power) / len(power),
    )


class GPUMonitor:
    """Background NVML poller. A no-op when NVML is unavailable."""

    def __init__(self, interval_s: float = 0.5, devices: list[int] | None = None) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self.interval_s = interval_s
        self.devices = devices
        self.samples: list[GPUSample] = []
        self.device_names: list[str] = []
        self._nvml: Any = None
        self._handles: list[Any] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def available(self) -> bool:
        return self._nvml is not None

    def _init_nvml(self) -> bool:
        try:
            import pynvml  # noqa: PLC0415
        except ImportError:
            logger.info("pynvml not installed; GPU utilization will not be recorded")
            return False
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            indices = self.devices if self.devices is not None else list(range(count))
            self._handles = [(i, pynvml.nvmlDeviceGetHandleByIndex(i)) for i in indices]
            self.device_names = [
                _as_str(pynvml.nvmlDeviceGetName(handle)) for _, handle in self._handles
            ]
        except Exception as exc:  # pragma: no cover - requires NVIDIA drivers
            logger.info("NVML unavailable (%s); GPU utilization will not be recorded", exc)
            return False
        self._nvml = pynvml
        return True

    def start(self) -> bool:
        """Begin sampling. Returns False when no GPU is visible."""
        if self._thread is not None:
            return self.available
        if not self._init_nvml():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:  # pragma: no cover - requires NVIDIA drivers
        pynvml = self._nvml
        while not self._stop.is_set():
            now = time.monotonic()
            for index, handle in self._handles:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except Exception:
                        power = 0.0
                    self.samples.append(
                        GPUSample(
                            t=now,
                            device=index,
                            sm_util_pct=float(util.gpu),
                            memory_used_mb=memory.used / (1024 * 1024),
                            memory_total_mb=memory.total / (1024 * 1024),
                            power_w=power,
                        )
                    )
                except Exception as exc:
                    logger.debug("NVML sample failed: %s", exc)
            self._stop.wait(self.interval_s)

    def stop(self) -> GPUStats:
        """Stop sampling and return the aggregate."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception as exc:  # pragma: no cover - driver teardown
                logger.debug("nvmlShutdown failed: %s", exc)
            self._nvml = None
        stats = summarize(self.samples)
        stats.device_names = list(self.device_names)
        return stats

    def __enter__(self) -> "GPUMonitor":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stats = self.stop()


def _as_str(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
