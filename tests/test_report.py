import csv
import io
import unittest

from llm_serve.bench.aggregate import BenchmarkResult
from llm_serve.bench.report import (
    best_by_concurrency,
    comparison_table,
    cost_table,
    rank,
    render_csv,
    render_markdown,
)


def result(backend, concurrency=16, tps=1000.0, ttft=0.2, itl=0.02, gpu=90.0, hits=0.0):
    return BenchmarkResult(
        backend=backend,
        model="mistral-7b-qlora",
        concurrency=concurrency,
        duration_s=30.0,
        completed=200,
        failed=0,
        total_prompt_tokens=100_000,
        total_generation_tokens=int(tps * 30),
        request_throughput=tps / 128,
        output_token_throughput=tps,
        total_token_throughput=tps * 1.5,
        ttft_s={"mean": ttft, "std": 0.0, "min": ttft, "max": ttft * 3, "p50": ttft, "p90": ttft * 2, "p95": ttft * 2, "p99": ttft * 3},
        itl_s={"mean": itl, "std": 0.0, "min": itl, "max": itl, "p50": itl, "p90": itl, "p95": itl, "p99": itl},
        e2e_s={"mean": 2.0, "std": 0.0, "min": 1.0, "max": 4.0, "p50": 2.0, "p90": 3.0, "p95": 3.5, "p99": 4.0},
        slo_attainment=0.97,
        goodput=5.0,
        prefix_cache_hit_rate=hits,
        gpu={"gpu_mean_sm_util_pct": gpu},
        workload={"num_requests": 200, "mean_input_tokens": 512, "mean_output_tokens": 128, "shared_prefix_tokens": 0},
        metadata={"gpu_devices": ["NVIDIA A100-SXM4-80GB"], "arrival": "closed_loop"},
    )


class TestTables(unittest.TestCase):
    def test_comparison_table_shape(self):
        table = comparison_table([result("vllm"), result("trtllm", tps=1400)])
        lines = table.splitlines()
        self.assertEqual(len(lines), 4)  # header, rule, two rows
        self.assertIn("Backend", lines[0])
        self.assertIn("Output tok/s", lines[0])
        self.assertTrue(all(line.startswith("|") and line.endswith("|") for line in lines))
        self.assertIn("vllm", table)
        self.assertIn("trtllm", table)

    def test_latencies_are_reported_in_milliseconds(self):
        table = comparison_table([result("vllm", ttft=0.25)])
        self.assertIn("250.0", table)

    def test_cost_table(self):
        table = cost_table([result("vllm", tps=1000.0)], gpu_hourly_usd=3.6)
        self.assertIn("$1.000", table)

    def test_best_by_concurrency(self):
        table = best_by_concurrency(
            [
                result("vllm", concurrency=8, tps=800, ttft=0.10),
                result("trtllm", concurrency=8, tps=1200, ttft=0.15),
                result("vllm", concurrency=64, tps=2000, ttft=0.5),
                result("trtllm", concurrency=64, tps=1800, ttft=0.4),
            ]
        )
        lines = table.splitlines()
        self.assertIn("trtllm", lines[2])   # best throughput at c=8
        self.assertIn("vllm", lines[2])     # best TTFT at c=8
        self.assertIn("vllm", lines[3])     # best throughput at c=64


class TestRanking(unittest.TestCase):
    def test_throughput_ranking_is_descending(self):
        ordered = rank([result("a", tps=100), result("b", tps=300), result("c", tps=200)])
        self.assertEqual([r.backend for r in ordered], ["b", "c", "a"])

    def test_latency_ranking_is_ascending(self):
        ordered = rank(
            [result("a", ttft=0.3), result("b", ttft=0.1), result("c", ttft=0.2)], key="ttft_p50"
        )
        self.assertEqual([r.backend for r in ordered], ["b", "c", "a"])

    def test_itl_ranking(self):
        ordered = rank([result("a", itl=0.05), result("b", itl=0.01)], key="itl_p50")
        self.assertEqual([r.backend for r in ordered], ["b", "a"])


class TestMarkdownReport(unittest.TestCase):
    def _report(self, **kw):
        results = [
            result("vllm", tps=1000.0, ttft=0.2),
            result("triton", tps=900.0, ttft=0.25),
            result("ray", tps=850.0, ttft=0.3),
            result("trtllm", tps=1400.0, ttft=0.12),
        ]
        return render_markdown(results, **kw)

    def test_sections_present(self):
        report = self._report()
        for heading in ("# Cross-backend", "## Results", "## Ranking", "## Serving cost", "## Relative to `vllm`"):
            self.assertIn(heading, report)

    def test_setup_metadata_is_recorded(self):
        report = self._report()
        self.assertIn("mistral-7b-qlora", report)
        self.assertIn("NVIDIA A100-SXM4-80GB", report)
        self.assertIn("closed_loop", report)

    def test_speedups_against_baseline(self):
        report = self._report()
        self.assertIn("1.40x", report)   # trtllm throughput vs vllm
        self.assertIn("0.60x", report)   # trtllm TTFT p50 vs vllm

    def test_baseline_can_be_changed(self):
        self.assertIn("## Relative to `triton`", self._report(baseline="triton"))

    def test_missing_baseline_is_skipped(self):
        self.assertNotIn("## Relative to", self._report(baseline="nonexistent"))

    def test_empty_results(self):
        self.assertIn("No results", render_markdown([]))

    def test_prefix_cache_section_only_when_measured(self):
        self.assertNotIn("## KV prefix cache", self._report())
        with_hits = render_markdown([result("vllm", hits=0.62)])
        self.assertIn("## KV prefix cache", with_hits)
        self.assertIn("62.0%", with_hits)

    def test_concurrency_section_only_for_a_sweep(self):
        single = render_markdown([result("vllm", concurrency=8)])
        self.assertNotIn("## Best backend per concurrency", single)
        sweep = render_markdown(
            [result("vllm", concurrency=8), result("vllm", concurrency=64)]
        )
        self.assertIn("## Best backend per concurrency", sweep)


class TestCSV(unittest.TestCase):
    def test_csv_parses_and_has_expected_columns(self):
        text = render_csv([result("vllm"), result("trtllm", tps=1400)])
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(len(rows), 2)
        self.assertEqual(sorted(r["backend"] for r in rows), ["trtllm", "vllm"])
        for column in ("output_token_throughput", "ttft_p99_s", "slo_attainment", "gpu_mean_sm_util_pct"):
            self.assertIn(column, rows[0])

    def test_rows_are_sorted_by_backend_then_concurrency(self):
        text = render_csv(
            [result("vllm", concurrency=64), result("vllm", concurrency=8), result("ray")]
        )
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(
            [(r["backend"], int(r["concurrency"])) for r in rows],
            [("ray", 16), ("vllm", 8), ("vllm", 64)],
        )


if __name__ == "__main__":
    unittest.main()
