import random
import unittest

from llm_serve.bench.workload import (
    generate_workload,
    poisson_arrivals,
    workload_summary,
)


class TestGenerateWorkload(unittest.TestCase):
    def test_count_and_shape(self):
        workload = generate_workload(20, input_len=100, output_len=32)
        self.assertEqual(len(workload), 20)
        self.assertTrue(all(r.prompt for r in workload))
        self.assertTrue(all(r.max_tokens >= 1 for r in workload))
        self.assertEqual([r.index for r in workload], list(range(20)))

    def test_deterministic_for_a_seed(self):
        a = generate_workload(10, seed=7)
        b = generate_workload(10, seed=7)
        self.assertEqual([r.prompt for r in a], [r.prompt for r in b])
        self.assertEqual([r.max_tokens for r in a], [r.max_tokens for r in b])

    def test_different_seeds_differ(self):
        a = generate_workload(10, seed=1)
        b = generate_workload(10, seed=2)
        self.assertNotEqual([r.prompt for r in a], [r.prompt for r in b])

    def test_lengths_track_the_requested_mean(self):
        workload = generate_workload(200, input_len=256, input_len_std=16, output_len=64, seed=3)
        mean_in = sum(r.prompt_tokens for r in workload) / len(workload)
        mean_out = sum(r.max_tokens for r in workload) / len(workload)
        self.assertLess(abs(mean_in - 256), 25)
        self.assertLess(abs(mean_out - 64), 8)

    def test_zero_std_gives_fixed_lengths(self):
        workload = generate_workload(10, input_len=128, input_len_std=0, output_len=32, output_len_std=0)
        self.assertEqual({r.max_tokens for r in workload}, {32})
        self.assertEqual({r.prompt_tokens for r in workload}, {128})

    def test_shared_prefix_is_common_to_every_prompt(self):
        workload = generate_workload(8, input_len=256, shared_prefix_len=64, seed=5)
        prefixes = {r.prompt[:200] for r in workload}
        self.assertEqual(len(prefixes), 1)
        self.assertTrue(all(r.shared_prefix_tokens == 64 for r in workload))

    def test_no_shared_prefix_by_default(self):
        workload = generate_workload(8, input_len=256, seed=5)
        self.assertGreater(len({r.prompt[:100] for r in workload}), 1)

    def test_open_loop_arrivals_are_increasing(self):
        workload = generate_workload(30, request_rate=10.0, seed=11)
        offsets = [r.arrival_offset_s for r in workload]
        self.assertEqual(offsets, sorted(offsets))
        self.assertGreater(offsets[-1], 0)

    def test_closed_loop_arrivals_are_zero(self):
        workload = generate_workload(5)
        self.assertEqual({r.arrival_offset_s for r in workload}, {0.0})

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            generate_workload(0)
        with self.assertRaises(ValueError):
            generate_workload(4, input_len=64, shared_prefix_len=64)


class TestPoissonArrivals(unittest.TestCase):
    def test_rate_is_approximately_honoured(self):
        offsets = poisson_arrivals(random.Random(1), rate=50.0, count=2000)
        self.assertAlmostEqual(len(offsets) / offsets[-1], 50.0, delta=6.0)

    def test_monotonic(self):
        offsets = poisson_arrivals(random.Random(2), rate=5.0, count=50)
        self.assertEqual(offsets, sorted(offsets))

    def test_invalid_rate(self):
        with self.assertRaises(ValueError):
            poisson_arrivals(random.Random(), rate=0.0, count=5)


class TestSummary(unittest.TestCase):
    def test_summary_fields(self):
        workload = generate_workload(20, input_len=100, output_len=40, shared_prefix_len=16, seed=9)
        summary = workload_summary(workload)
        self.assertEqual(summary["num_requests"], 20.0)
        self.assertEqual(summary["shared_prefix_tokens"], 16.0)
        self.assertGreater(summary["total_output_tokens"], 0)
        self.assertLessEqual(summary["min_input_tokens"], summary["mean_input_tokens"])
        self.assertGreaterEqual(summary["max_input_tokens"], summary["mean_input_tokens"])

    def test_empty_summary(self):
        self.assertEqual(workload_summary([]), {})


if __name__ == "__main__":
    unittest.main()
