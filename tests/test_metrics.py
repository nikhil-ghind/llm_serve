import unittest

from llm_serve.metrics.math import (
    INF,
    cumulative_bucket_counts,
    goodput,
    histogram_quantile,
    mean,
    percentile,
    percentiles,
    rate,
    slo_attainment,
    stdev,
    throughput,
)
from llm_serve.metrics.registry import Counter, Gauge, Histogram, MetricsRegistry, ServingMetrics
from llm_serve.types import FinishReason, GenerationResult


class TestPercentiles(unittest.TestCase):
    def test_known_values(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertAlmostEqual(percentile(data, 0), 1.0)
        self.assertAlmostEqual(percentile(data, 100), 10.0)
        self.assertAlmostEqual(percentile(data, 50), 5.5)
        self.assertAlmostEqual(percentile(data, 90), 9.1)

    def test_interpolation(self):
        self.assertAlmostEqual(percentile([0.0, 1.0], 25), 0.25)

    def test_single_sample(self):
        self.assertAlmostEqual(percentile([4.2], 99), 4.2)

    def test_unsorted_input_is_handled(self):
        self.assertAlmostEqual(percentile([9, 1, 5], 50), 5.0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            percentile([], 50)
        with self.assertRaises(ValueError):
            percentile([1, 2], 101)

    def test_percentiles_labels(self):
        out = percentiles(range(1, 101), [50, 90, 99])
        self.assertEqual(sorted(out), ["p50", "p90", "p99"])
        self.assertAlmostEqual(out["p50"], 50.5)

    def test_mean_and_stdev(self):
        self.assertEqual(mean([]), 0.0)
        self.assertAlmostEqual(mean([1, 2, 3]), 2.0)
        self.assertEqual(stdev([5]), 0.0)
        self.assertAlmostEqual(stdev([2, 4, 4, 4, 5, 5, 7, 9]), 2.0)


class TestBuckets(unittest.TestCase):
    def test_cumulative_counts(self):
        buckets = cumulative_bucket_counts([0.05, 0.2, 0.7, 3.0], [0.1, 0.5, 1.0])
        self.assertEqual(buckets, [(0.1, 1), (0.5, 2), (1.0, 3), (INF, 4)])

    def test_unsorted_bounds_rejected(self):
        with self.assertRaises(ValueError):
            cumulative_bucket_counts([1.0], [1.0, 0.5])

    def test_histogram_quantile_interpolates(self):
        buckets = [(1.0, 0), (2.0, 10), (INF, 10)]
        self.assertAlmostEqual(histogram_quantile(buckets, 0.5), 1.5)
        self.assertAlmostEqual(histogram_quantile(buckets, 1.0), 2.0)

    def test_histogram_quantile_edge_cases(self):
        self.assertEqual(histogram_quantile([(1.0, 0), (INF, 0)], 0.9), 0.0)
        # target lands in +Inf: report the largest finite bound
        self.assertAlmostEqual(histogram_quantile([(1.0, 5), (INF, 10)], 0.99), 1.0)
        with self.assertRaises(ValueError):
            histogram_quantile([], 0.5)
        with self.assertRaises(ValueError):
            histogram_quantile([(1.0, 1), (INF, 1)], 1.5)

    def test_quantile_matches_sample_percentile_roughly(self):
        samples = [i / 100 for i in range(1, 101)]
        bounds = [i / 20 for i in range(1, 21)]
        est = histogram_quantile(cumulative_bucket_counts(samples, bounds), 0.5)
        self.assertLess(abs(est - percentile(samples, 50)), 0.05)


class TestRates(unittest.TestCase):
    def test_rate_and_throughput(self):
        self.assertAlmostEqual(rate(100, 4), 25.0)
        self.assertEqual(rate(100, 0), 0.0)
        self.assertAlmostEqual(throughput(2048, 2.0), 1024.0)

    def test_goodput_counts_only_slo_meeting_requests(self):
        latencies = [0.5, 0.8, 1.5, 2.0]
        self.assertAlmostEqual(goodput(latencies, slo_s=1.0, elapsed_s=2.0), 1.0)
        self.assertEqual(goodput(latencies, 1.0, 0.0), 0.0)

    def test_slo_attainment(self):
        self.assertAlmostEqual(slo_attainment([0.1, 0.2, 5.0, 6.0], 1.0), 0.5)
        self.assertEqual(slo_attainment([], 1.0), 0.0)


class TestRegistryPrimitives(unittest.TestCase):
    def test_counter(self):
        c = Counter("reqs", "requests", ["backend"])
        c.inc(labels={"backend": "vllm"})
        c.inc(4, labels={"backend": "vllm"})
        self.assertEqual(c.value({"backend": "vllm"}), 5.0)
        self.assertEqual(c.value({"backend": "triton"}), 0.0)
        with self.assertRaises(ValueError):
            c.inc(-1, labels={"backend": "vllm"})
        with self.assertRaises(ValueError):
            c.inc(labels={"wrong": "x"})

    def test_gauge(self):
        g = Gauge("running", "running")
        g.set(5)
        g.inc(2)
        g.dec(3)
        self.assertEqual(g.value(), 4.0)

    def test_histogram_observe_and_quantile(self):
        h = Histogram("lat", "latency", [0.1, 0.5, 1.0])
        h.observe_all([0.05, 0.2, 0.2, 2.0])
        self.assertEqual(h.count(), 4)
        self.assertAlmostEqual(h.sum(), 2.45)
        self.assertEqual(h.buckets(), [(0.1, 1), (0.5, 3), (1.0, 3), (INF, 4)])
        self.assertGreater(h.quantile(0.5), 0.1)

    def test_histogram_requires_sorted_buckets(self):
        with self.assertRaises(ValueError):
            Histogram("x", "x", [1.0, 0.5])
        with self.assertRaises(ValueError):
            Histogram("x", "x", [])


class TestExposition(unittest.TestCase):
    def test_render_format(self):
        reg = MetricsRegistry("llm_serve")
        counter = reg.counter("requests", "Total requests.", ["backend"])
        counter.inc(3, {"backend": "vllm"})
        gauge = reg.gauge("running", "Running requests.")
        gauge.set(2)
        text = reg.render()
        self.assertIn("# HELP llm_serve_requests Total requests.", text)
        self.assertIn("# TYPE llm_serve_requests counter", text)
        self.assertIn('llm_serve_requests_total{backend="vllm"} 3', text)
        self.assertIn("llm_serve_running 2", text)
        self.assertTrue(text.endswith("\n"))

    def test_histogram_render_has_le_and_inf(self):
        reg = MetricsRegistry("ns")
        hist = reg.histogram("ttft", "TTFT.", [0.1, 1.0])
        hist.observe(0.05)
        text = reg.render()
        self.assertIn('ns_ttft_bucket{le="0.1"} 1', text)
        self.assertIn('ns_ttft_bucket{le="+Inf"} 1', text)
        self.assertIn("ns_ttft_count 1", text)

    def test_empty_metric_is_not_rendered(self):
        reg = MetricsRegistry("ns")
        reg.counter("unused", "Never touched.")
        self.assertEqual(reg.render(), "")

    def test_duplicate_registration_rejected(self):
        reg = MetricsRegistry("ns")
        reg.counter("dup", "first")
        with self.assertRaises(ValueError):
            reg.counter("dup", "second")

    def test_label_values_are_escaped(self):
        reg = MetricsRegistry("ns")
        c = reg.counter("c", "c", ["model"])
        c.inc(labels={"model": 'weird"name'})
        self.assertIn('model="weird\\"name"', reg.render())


class TestServingMetrics(unittest.TestCase):
    def _result(self, **kw):
        defaults = dict(
            request_id="r1",
            text="hello",
            prompt_tokens=100,
            completion_tokens=20,
            finish_reason=FinishReason.STOP,
            ttft_s=0.12,
            e2e_latency_s=1.4,
            inter_token_latencies_s=[0.01, 0.02, 0.03],
            cached_prompt_tokens=64,
        )
        defaults.update(kw)
        return GenerationResult(**defaults)

    def test_observe_result_populates_everything(self):
        m = ServingMetrics("llm_serve")
        m.observe_result(self._result(), backend="vllm", model="mistral-7b-qlora")
        labels = {"backend": "vllm", "model": "mistral-7b-qlora"}
        self.assertEqual(m.prompt_tokens_total.value(labels), 100)
        self.assertEqual(m.generation_tokens_total.value(labels), 20)
        self.assertEqual(m.cached_prompt_tokens_total.value(labels), 64)
        self.assertEqual(m.ttft_seconds.count(labels), 1)
        self.assertEqual(m.itl_seconds.count(labels), 3)
        self.assertEqual(m.e2e_seconds.count(labels), 1)
        self.assertEqual(m.requests_total.value({**labels, "status": "ok"}), 1)

    def test_missing_timings_are_skipped(self):
        m = ServingMetrics("llm_serve")
        m.observe_result(
            self._result(ttft_s=None, e2e_latency_s=None, inter_token_latencies_s=[], cached_prompt_tokens=0),
            backend="mock",
            model="m",
        )
        labels = {"backend": "mock", "model": "m"}
        self.assertEqual(m.ttft_seconds.count(labels), 0)
        self.assertEqual(m.cached_prompt_tokens_total.value(labels), 0)

    def test_observe_stats_sets_gauges(self):
        from llm_serve.backends.base import BackendStats

        m = ServingMetrics("llm_serve")
        m.observe_stats(
            BackendStats(backend="triton", running=4, waiting=7, kv_cache_usage=0.61, prefix_cache_hit_rate=0.4),
            model="m",
        )
        labels = {"backend": "triton", "model": "m"}
        self.assertEqual(m.running.value(labels), 4)
        self.assertEqual(m.waiting.value(labels), 7)
        self.assertAlmostEqual(m.kv_cache_usage.value(labels), 0.61)
        self.assertAlmostEqual(m.prefix_hit_rate.value(labels), 0.4)

    def test_render_uses_prometheus_names(self):
        m = ServingMetrics("llm_serve")
        m.observe_result(self._result(), backend="vllm", model="m")
        text = m.render()
        self.assertIn("llm_serve_time_to_first_token_seconds_bucket", text)
        self.assertIn("llm_serve_generation_tokens_total", text)


if __name__ == "__main__":
    unittest.main()
