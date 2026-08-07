"""Async load generator.

Drives a :class:`~llm_serve.backends.base.Backend` — which may be the in-process
vLLM engine, an HTTP/gRPC client for Triton or Ray, or the CPU mock — and records
per-request TTFT, per-token ITL and end-to-end latency.

Two arrival models:

* **Closed loop** (``request_rate`` unset): ``concurrency`` workers each send the
  next request as soon as the previous one returns. This measures the server's
  capacity — throughput at saturation.
* **Open loop** (``request_rate`` set): requests are issued on a Poisson schedule
  regardless of whether earlier ones have finished. This measures behaviour under
  a fixed offered load, and is the only way to see queueing blow up before
  throughput does.

Warmup requests are issued and discarded first so engine compilation, CUDA graph
capture and the first cold-cache prefill do not land in the measured window.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Sequence

from ..backends.base import Backend
from ..types import GenerationRequest, SamplingParams
from .aggregate import BenchmarkResult, RequestRecord, aggregate
from .gpu_monitor import GPUMonitor
from .workload import WorkloadRequest, workload_summary

logger = logging.getLogger("llm_serve.bench.loadgen")


@dataclass
class RunConfig:
    concurrency: int = 16
    request_rate: float | None = None
    duration_s: float | None = None
    warmup_requests: int = 8
    ttft_slo_s: float = 1.0
    itl_slo_s: float = 0.05
    gpu_sample_interval_s: float = 0.5
    ignore_eos: bool = True
    seed: int | None = None

    def validate(self) -> "RunConfig":
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.request_rate is not None and self.request_rate <= 0:
            raise ValueError("request_rate must be > 0 when set")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError("duration_s must be > 0 when set")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be >= 0")
        return self


async def run_one(
    backend: Backend, item: WorkloadRequest, run: RunConfig, index: int
) -> RequestRecord:
    """Issue one streaming request and time every token."""
    request = GenerationRequest(
        prompt=item.prompt,
        sampling=SamplingParams(
            max_tokens=item.max_tokens,
            temperature=0.0,
            # Fixed output length keeps the comparison honest: an engine that
            # stops early would otherwise post better tokens/s for less work.
            ignore_eos=run.ignore_eos,
            seed=run.seed,
        ),
        stream=True,
    )
    start = time.monotonic()
    ttft: float | None = None
    last: float | None = None
    itls: list[float] = []
    tokens = 0
    finish = "stop"
    error: str | None = None
    try:
        async for chunk in backend.generate_stream(request):
            now = time.monotonic()
            if chunk.text:
                tokens += 1
                if ttft is None:
                    ttft = now - start
                elif last is not None:
                    itls.append(now - last)
                last = now
            if chunk.is_final and chunk.finish_reason is not None:
                finish = chunk.finish_reason.value
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.debug("request %s failed: %s", request.request_id, error)
    end = time.monotonic()
    return RequestRecord(
        index=index,
        request_id=request.request_id,
        prompt_tokens=item.prompt_tokens,
        completion_tokens=tokens,
        start_s=start,
        end_s=end,
        ttft_s=ttft,
        itls_s=itls,
        finish_reason=finish,
        error=error,
    )


async def _closed_loop(
    backend: Backend, workload: Sequence[WorkloadRequest], run: RunConfig, deadline: float | None
) -> list[RequestRecord]:
    records: list[RequestRecord] = []
    queue: asyncio.Queue[tuple[int, WorkloadRequest]] = asyncio.Queue()
    for i, item in enumerate(workload):
        queue.put_nowait((i, item))

    async def worker() -> None:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return
            try:
                index, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            records.append(await run_one(backend, item, run, index))

    await asyncio.gather(*[worker() for _ in range(run.concurrency)])
    return records


async def _open_loop(
    backend: Backend, workload: Sequence[WorkloadRequest], run: RunConfig, deadline: float | None
) -> list[RequestRecord]:
    records: list[RequestRecord] = []
    tasks: list[asyncio.Task] = []
    limiter = asyncio.Semaphore(run.concurrency)
    start = time.monotonic()

    async def issue(index: int, item: WorkloadRequest) -> None:
        async with limiter:
            records.append(await run_one(backend, item, run, index))

    for index, item in enumerate(workload):
        if deadline is not None and time.monotonic() >= deadline:
            break
        wait = item.arrival_offset_s - (time.monotonic() - start)
        if wait > 0:
            await asyncio.sleep(wait)
        tasks.append(asyncio.create_task(issue(index, item)))
    if tasks:
        await asyncio.gather(*tasks)
    return records


async def run_benchmark(
    backend: Backend,
    workload: Sequence[WorkloadRequest],
    run: RunConfig,
    *,
    model: str = "mistral-7b-qlora",
    metadata: dict | None = None,
) -> BenchmarkResult:
    """Warm up, drive the workload, and aggregate the results."""
    run.validate()
    if not workload:
        raise ValueError("workload must not be empty")

    await backend.start()

    if run.warmup_requests:
        logger.info("warmup: %d request(s)", run.warmup_requests)
        warmup = [workload[i % len(workload)] for i in range(run.warmup_requests)]
        await asyncio.gather(
            *[run_one(backend, item, run, -1) for item in warmup]
        )

    monitor = GPUMonitor(interval_s=run.gpu_sample_interval_s)
    monitor.start()

    deadline = time.monotonic() + run.duration_s if run.duration_s else None
    started = time.monotonic()
    try:
        if run.request_rate:
            records = await _open_loop(backend, workload, run, deadline)
        else:
            records = await _closed_loop(backend, workload, run, deadline)
    finally:
        gpu_stats = monitor.stop()
    elapsed = max(time.monotonic() - started, 1e-9)

    stats = await backend.stats()
    meta = dict(metadata or {})
    meta.update(
        {
            "request_rate": run.request_rate,
            "warmup_requests": run.warmup_requests,
            "arrival": "open_loop" if run.request_rate else "closed_loop",
            "gpu_devices": gpu_stats.device_names,
        }
    )

    result = aggregate(
        records,
        backend=backend.name,
        model=model,
        concurrency=run.concurrency,
        duration_s=elapsed,
        ttft_slo_s=run.ttft_slo_s,
        itl_slo_s=run.itl_slo_s,
        gpu=gpu_stats.to_dict(),
        workload=workload_summary(workload),
        metadata=meta,
    )
    if stats.prefix_cache_hit_rate:
        result.prefix_cache_hit_rate = stats.prefix_cache_hit_rate
    logger.info(
        "%s: %d requests in %.2fs -> %.1f tok/s, TTFT p50=%.3fs p99=%.3fs",
        backend.name,
        result.completed,
        elapsed,
        result.output_token_throughput,
        result.ttft_s["p50"],
        result.ttft_s["p99"],
    )
    return result
