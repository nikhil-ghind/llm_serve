import json
import os
import tempfile
import unittest

from llm_serve.bench.aggregate import (
    BenchmarkResult,
    RequestRecord,
    aggregate,
    cost_per_million_tokens,
    load_results,
    speedup,
)
from llm_serve.bench.gpu_monitor import GPUSample, summarize


def record(i, ttft=0.1, itl=0.02, tokens=10, prompt=100, start=0.0, error=None):
    return RequestRecord(
        index=i,
        request_id=f"r{i}",
        prompt_tokens=prompt,
        completion_tokens=0 if error else tokens,
        start_s=start,
        end_s=start + ttft + itl * max(0, tokens - 1),
        ttft_s=None if error else ttft,
        itls_s=[] if error else [itl] * max(0, tokens - 1),
        error=error,
    )


class TestRequestRecord(unittest.TestCase):
    def test_derived_properties(self):
        r = record(0, ttft=0.2, itl=0.05, tokens=5)
        self.assertTrue(r.ok)
        self.assertAlmostEqual(r.latency_s, 0.2 + 0.05 * 4)
        self.assertAlmostEqual(r.mean_itl_s, 0.05)

    def test_failed_record_is_not_ok(self):
        r = record(0, error="ConnectionError")
        self.assertFalse(r.ok)
        self.assertIsNone(r.mean_itl_s)


class TestAggregate(unittest.TestCase):
    def test_throughput_math(self):
        records = [record(i, tokens=10, prompt=100) for i in range(10)]
        result = aggregate(
            records, backend="mock", model="m", concurrency=4, duration_s=2.0
        )
        self.assertEqual(result.completed, 10)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.total_generation_tokens, 100)
        self.assertEqual(result.total_prompt_tokens, 1000)
        self.assertAlmostEqual(result.request_throughput, 5.0)
        self.assertAlmostEqual(result.output_token_throughput, 50.0)
        self.assertAlmostEqual(result.total_token_throughput, 550.0)

    def test_failures_are_excluded_from_latency_but_counted(self):
        records = [record(i) for i in range(8)] + [record(9, error="boom")]
        result = aggregate(records, backend="mock", model="m", concurrency=2, duration_s=1.0)
        self.assertEqual(result.completed, 8)
        self.assertEqual(result.failed, 1)
        self.assertAlmostEqual(result.ttft_s["p50"], 0.1)

    def test_percentiles_present(self):
        records = [record(i, ttft=0.01 * (i + 1)) for i in range(100)]
        result = aggregate(records, backend="mock", model="m", concurrency=1, duration_s=1.0)
        for key in ("mean", "std", "min", "max", "p50", "p90", "p95", "p99"):
            self.assertIn(key, result.ttft_s)
        self.assertLess(result.ttft_s["p50"], result.ttft_s["p99"])

    def test_slo_attainment_and_goodput(self):
        good = [record(i, ttft=0.2, itl=0.01) for i in range(6)]
        slow_ttft = [record(10 + i, ttft=5.0, itl=0.01) for i in range(2)]
        slow_itl = [record(20 + i, ttft=0.2, itl=0.5) for i in range(2)]
        result = aggregate(
            good + slow_ttft + slow_itl,
            backend="mock",
            model="m",
            concurrency=1,
            duration_s=2.0,
            ttft_slo_s=1.0,
            itl_slo_s=0.05,
        )
        self.assertAlmostEqual(result.slo_attainment, 0.6)
        self.assertAlmostEqual(result.goodput, 3.0)

    def test_empty_records_do_not_crash(self):
        result = aggregate([], backend="mock", model="m", concurrency=1, duration_s=1.0)
        self.assertEqual(result.completed, 0)
        self.assertEqual(result.ttft_s["p50"], 0.0)
        self.assertEqual(result.slo_attainment, 0.0)

    def test_zero_duration_rejected(self):
        with self.assertRaises(ValueError):
            aggregate([], backend="m", model="m", concurrency=1, duration_s=0.0)

    def test_prefix_cache_hit_rate_from_records(self):
        records = [record(i, prompt=100) for i in range(4)]
        for r in records:
            r.cached_prompt_tokens = 40
        result = aggregate(records, backend="mock", model="m", concurrency=1, duration_s=1.0)
        self.assertAlmostEqual(result.prefix_cache_hit_rate, 0.4)
        self.assertEqual(result.cached_prompt_tokens, 160)


class TestDerivedComparisons(unittest.TestCase):
    def _result(self, backend, tps, ttft_p50, ttft_p99=1.0, itl_p50=0.02):
        return BenchmarkResult(
            backend=backend,
            model="m",
            concurrency=8,
            duration_s=10.0,
            completed=10,
            failed=0,
            total_prompt_tokens=1000,
            total_generation_tokens=int(tps * 10),
            request_throughput=1.0,
            output_token_throughput=tps,
            total_token_throughput=tps + 100,
            ttft_s={"p50": ttft_p50, "p99": ttft_p99, "mean": ttft_p50},
            itl_s={"p50": itl_p50, "p99": itl_p50 * 2, "mean": itl_p50},
            e2e_s={"p50": 1.0, "p99": 2.0, "mean": 1.2},
        )

    def test_cost_per_million_tokens(self):
        result = self._result("vllm", tps=1000.0, ttft_p50=0.1)
        cost = cost_per_million_tokens(result, gpu_hourly_usd=3.6)
        # 1000 tok/s = 3.6M tokens/hour -> $1 per million
        self.assertAlmostEqual(cost, 1.0, places=6)

    def test_cost_of_zero_throughput_is_infinite(self):
        result = self._result("vllm", tps=0.0, ttft_p50=0.1)
        self.assertEqual(cost_per_million_tokens(result, 3.6), float("inf"))

    def test_negative_price_rejected(self):
        with self.assertRaises(ValueError):
            cost_per_million_tokens(self._result("v", 100.0, 0.1), -1.0)

    def test_speedup_ratios(self):
        base = self._result("vllm", tps=1000.0, ttft_p50=0.2, ttft_p99=1.0, itl_p50=0.02)
        faster = self._result("trtllm", tps=1400.0, ttft_p50=0.1, ttft_p99=0.6, itl_p50=0.015)
        ratios = speedup(faster, base)
        self.assertAlmostEqual(ratios["throughput_x"], 1.4)
        self.assertAlmostEqual(ratios["ttft_p50_x"], 0.5)
        self.assertAlmostEqual(ratios["ttft_p99_x"], 0.6)
        self.assertAlmostEqual(ratios["itl_p50_x"], 0.75)


class TestSerialization(unittest.TestCase):
    def _result(self):
        return aggregate(
            [record(i) for i in range(5)],
            backend="mock",
            model="m",
            concurrency=2,
            duration_s=1.0,
            gpu={"gpu_mean_sm_util_pct": 88.0},
            metadata={"label": "smoke"},
        )

    def test_json_roundtrip(self):
        original = self._result()
        restored = BenchmarkResult.from_dict(json.loads(original.to_json()))
        self.assertEqual(restored.backend, original.backend)
        self.assertAlmostEqual(restored.output_token_throughput, original.output_token_throughput)
        self.assertEqual(restored.metadata["label"], "smoke")

    def test_from_dict_ignores_unknown_keys(self):
        data = json.loads(self._result().to_json())
        data["future_field"] = 1
        self.assertIsInstance(BenchmarkResult.from_dict(data), BenchmarkResult)

    def test_load_results_reads_files_and_lists(self):
        directory = tempfile.mkdtemp()
        single = os.path.join(directory, "one.json")
        with open(single, "w", encoding="utf-8") as fh:
            fh.write(self._result().to_json())
        combined = os.path.join(directory, "many.json")
        with open(combined, "w", encoding="utf-8") as fh:
            json.dump([self._result().to_dict(), self._result().to_dict()], fh)
        results = load_results([single, combined])
        self.assertEqual(len(results), 3)


class TestGPUSummary(unittest.TestCase):
    def test_no_samples_means_unavailable(self):
        stats = summarize([])
        self.assertFalse(stats.available)
        self.assertEqual(stats.to_dict()["gpu_available"], 0.0)

    def test_summary_math(self):
        samples = [
            GPUSample(t=0.0, device=0, sm_util_pct=50.0, memory_used_mb=1000, memory_total_mb=80000, power_w=200),
            GPUSample(t=0.5, device=0, sm_util_pct=90.0, memory_used_mb=2000, memory_total_mb=80000, power_w=300),
        ]
        stats = summarize(samples)
        self.assertTrue(stats.available)
        self.assertEqual(stats.device_count, 1)
        self.assertAlmostEqual(stats.mean_sm_util_pct, 70.0)
        self.assertAlmostEqual(stats.max_sm_util_pct, 90.0)
        self.assertAlmostEqual(stats.mean_memory_used_mb, 1500.0)
        self.assertAlmostEqual(stats.mean_power_w, 250.0)
        self.assertAlmostEqual(samples[0].memory_util_pct, 1.25)

    def test_monitor_is_noop_without_nvml(self):
        from llm_serve.bench.gpu_monitor import GPUMonitor

        monitor = GPUMonitor(interval_s=0.01)
        started = monitor.start()
        stats = monitor.stop()
        # No NVIDIA driver in this environment: must degrade, not raise.
        self.assertEqual(started, monitor.available)
        self.assertIsNotNone(stats)

    def test_invalid_interval(self):
        from llm_serve.bench.gpu_monitor import GPUMonitor

        with self.assertRaises(ValueError):
            GPUMonitor(interval_s=0)


if __name__ == "__main__":
    unittest.main()
