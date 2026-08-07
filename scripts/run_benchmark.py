#!/usr/bin/env python3
"""Run one benchmark against one backend and write a result JSON.

    python scripts/run_benchmark.py --backend vllm --concurrency 32 --num-requests 500
    python scripts/run_benchmark.py --backend mock --concurrency 8 --num-requests 40   # CPU smoke test

A concurrency ladder can be swept in one invocation with repeated
``--concurrency`` flags; each level is a separate result record.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_serve.backends.base import available_backends, create_backend  # noqa: E402
from llm_serve.bench.loadgen import RunConfig, run_benchmark  # noqa: E402
from llm_serve.bench.workload import generate_workload  # noqa: E402
from llm_serve.config import build_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark one serving backend.")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--backend", default=None, choices=available_backends())
    parser.add_argument("--endpoint", default=None, help="override backend.endpoint")
    parser.add_argument("--engine-dir", default=None, help="override backend.engine_dir")
    parser.add_argument(
        "--concurrency", type=int, action="append", default=[], help="repeatable, sweeps a ladder"
    )
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None, help="cap the run in seconds")
    parser.add_argument("--request-rate", type=float, default=None, help="open-loop req/s")
    parser.add_argument("--input-len", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--shared-prefix-len", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="benchmarks/results", help="output directory")
    parser.add_argument("--label", default=None, help="tag recorded in the result metadata")
    parser.add_argument("--log-level", default="info")
    return parser


def resolve(args) -> tuple:
    overrides = []
    if args.backend:
        overrides.append(f"backend.kind={args.backend}")
    if args.endpoint:
        overrides.append(f"backend.endpoint={args.endpoint}")
    if args.engine_dir:
        overrides.append(f"backend.engine_dir={args.engine_dir}")
    for flag, option in (
        (args.num_requests, "bench.num_requests"),
        (args.duration, "bench.duration_s"),
        (args.request_rate, "bench.request_rate"),
        (args.input_len, "bench.input_len"),
        (args.output_len, "bench.output_len"),
        (args.shared_prefix_len, "bench.shared_prefix_len"),
        (args.warmup, "bench.warmup_requests"),
        (args.seed, "bench.seed"),
    ):
        if flag is not None:
            overrides.append(f"{option}={flag}")
    config = build_config(args.config, cli_overrides=overrides)
    levels = args.concurrency or [config.bench.concurrency]
    return config, levels


async def main_async(args) -> int:
    config, levels = resolve(args)
    bench = config.bench

    workload = generate_workload(
        num_requests=bench.num_requests,
        input_len=bench.input_len,
        input_len_std=bench.input_len_std,
        output_len=bench.output_len,
        output_len_std=bench.output_len_std,
        shared_prefix_len=bench.shared_prefix_len,
        seed=bench.seed,
        request_rate=bench.request_rate,
    )

    os.makedirs(args.output, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = []

    for level in levels:
        backend = create_backend(config)
        run = RunConfig(
            concurrency=level,
            request_rate=bench.request_rate,
            duration_s=bench.duration_s,
            warmup_requests=bench.warmup_requests,
            ttft_slo_s=bench.slo_ttft_s,
            itl_slo_s=bench.slo_itl_s,
            gpu_sample_interval_s=bench.gpu_sample_interval_s,
        )
        try:
            result = await run_benchmark(
                backend,
                workload,
                run,
                model=config.model.name,
                metadata={
                    "label": args.label,
                    "timestamp": stamp,
                    "tensor_parallel_size": config.backend.tensor_parallel_size,
                    "enable_prefix_caching": config.backend.enable_prefix_caching,
                    "quantization": config.model.quantization,
                },
            )
        finally:
            await backend.stop()
        results.append(result)

        path = os.path.join(args.output, f"{config.backend.kind}-c{level}-{stamp}.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(result.to_json())
        print(f"wrote {path}")
        print(
            f"  {result.backend} c={level}: {result.output_token_throughput:.1f} out-tok/s | "
            f"{result.request_throughput:.2f} req/s | "
            f"TTFT p50={result.ttft_s['p50'] * 1000:.0f}ms p99={result.ttft_s['p99'] * 1000:.0f}ms | "
            f"ITL p50={result.itl_s['p50'] * 1000:.1f}ms | SLO={result.slo_attainment * 100:.1f}%"
        )

    combined = os.path.join(args.output, f"{config.backend.kind}-{stamp}.json")
    with open(combined, "w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in results], fh, indent=2, sort_keys=True)
    print(f"wrote {combined}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
