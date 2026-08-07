"""Load generator end to end against the CPU mock backend.

This is the smoke test for the whole benchmark path: workload -> load generator
-> aggregation -> report, with no GPU involved.
"""

import asyncio
import unittest

from llm_serve.backends.mock import MockBackend
from llm_serve.bench.loadgen import RunConfig, run_benchmark, run_one
from llm_serve.bench.report import render_csv, render_markdown
from llm_serve.bench.workload import generate_workload
from llm_serve.config import build_config


def _config(*overrides):
    return build_config(
        None,
        environ={},
        cli_overrides=["backend.mock_prefill_s=0.0", "backend.mock_decode_s=0.0", *overrides],
    )


def _run(workload, run, config=None, **kw):
    backend = MockBackend(config or _config())

    async def go():
        try:
            return await run_benchmark(backend, workload, run, **kw)
        finally:
            await backend.stop()

    return asyncio.run(go())


class TestRunOne(unittest.TestCase):
    def test_records_timings_and_tokens(self):
        backend = MockBackend(_config())
        item = generate_workload(1, input_len=64, output_len=12, output_len_std=0)[0]
        record = asyncio.run(run_one(backend, item, RunConfig(), index=0))
        self.assertTrue(record.ok)
        self.assertEqual(record.completion_tokens, 12)
        self.assertIsNotNone(record.ttft_s)
        self.assertEqual(len(record.itls_s), 11)
        self.assertGreater(record.latency_s, 0)
        self.assertIsNone(record.error)

    def test_backend_failure_is_captured_not_raised(self):
        class Broken(MockBackend):
            async def generate_stream(self, request):
                raise RuntimeError("engine died")
                yield  # pragma: no cover - unreachable, keeps this a generator

        item = generate_workload(1)[0]
        record = asyncio.run(run_one(Broken(_config()), item, RunConfig(), index=0))
        self.assertFalse(record.ok)
        self.assertIn("engine died", record.error)


class TestClosedLoop(unittest.TestCase):
    def test_full_pipeline(self):
        workload = generate_workload(
            24, input_len=128, input_len_std=0, output_len=8, output_len_std=0, seed=1
        )
        result = _run(workload, RunConfig(concurrency=4, warmup_requests=2))
        self.assertEqual(result.backend, "mock")
        self.assertEqual(result.completed, 24)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.total_generation_tokens, 24 * 8)
        self.assertGreater(result.output_token_throughput, 0)
        self.assertGreater(result.ttft_s["p99"], 0)
        self.assertEqual(result.concurrency, 4)
        self.assertEqual(result.metadata["arrival"], "closed_loop")
        self.assertEqual(result.workload["num_requests"], 24)

    def test_warmup_requests_are_excluded(self):
        workload = generate_workload(10, input_len=64, output_len=4, output_len_std=0)
        result = _run(workload, RunConfig(concurrency=2, warmup_requests=5))
        self.assertEqual(result.completed, 10)

    def test_duration_cap_stops_early(self):
        # A per-token delay makes the run outlast the cap, so the deadline is
        # what ends it rather than the workload running out.
        slow = build_config(
            None,
            environ={},
            cli_overrides=["backend.mock_prefill_s=0.002", "backend.mock_decode_s=0.002"],
        )
        workload = generate_workload(500, input_len=64, output_len=8, output_len_std=0)
        result = _run(
            workload, RunConfig(concurrency=4, warmup_requests=0, duration_s=0.1), config=slow
        )
        self.assertLess(result.completed, 500)
        self.assertGreater(result.completed, 0)


class TestOpenLoop(unittest.TestCase):
    def test_poisson_arrivals_are_honoured(self):
        workload = generate_workload(
            12, input_len=64, output_len=4, output_len_std=0, request_rate=200.0, seed=3
        )
        result = _run(workload, RunConfig(concurrency=8, warmup_requests=0, request_rate=200.0))
        self.assertEqual(result.completed, 12)
        self.assertEqual(result.metadata["arrival"], "open_loop")
        self.assertEqual(result.metadata["request_rate"], 200.0)


class TestRunConfigValidation(unittest.TestCase):
    def test_invalid_values(self):
        for kwargs in (
            {"concurrency": 0},
            {"request_rate": 0.0},
            {"duration_s": -1.0},
            {"warmup_requests": -1},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    RunConfig(**kwargs).validate()

    def test_empty_workload_rejected(self):
        with self.assertRaises(ValueError):
            _run([], RunConfig())


class TestReportFromRealRun(unittest.TestCase):
    def test_markdown_and_csv_render_from_measured_results(self):
        workload = generate_workload(12, input_len=64, output_len=6, output_len_std=0, seed=2)
        results = [
            _run(workload, RunConfig(concurrency=c, warmup_requests=1)) for c in (2, 4)
        ]
        report = render_markdown(results, baseline="mock")
        self.assertIn("## Results", report)
        self.assertIn("mock", report)
        self.assertIn("## Best backend per concurrency level", report)
        csv_text = render_csv(results)
        self.assertEqual(len(csv_text.strip().splitlines()), 3)  # header + 2 runs


if __name__ == "__main__":
    unittest.main()
