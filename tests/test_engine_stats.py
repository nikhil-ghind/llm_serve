import unittest

from llm_serve.engine.stats import EngineStatsTracker


class TestEngineStatsTracker(unittest.TestCase):
    def _tracker(self, n=4):
        tr = EngineStatsTracker(window=8)
        for i in range(n):
            tr.record(
                t=float(i),
                running=i,
                waiting=n - i,
                kv_cache_usage=i / 10.0,
                batch_size=i + 1,
                batched_tokens=(i + 1) * 10,
            )
        return tr

    def test_empty_tracker_is_all_zero(self):
        tr = EngineStatsTracker()
        self.assertEqual(len(tr), 0)
        self.assertEqual(tr.avg_batch_size, 0.0)
        self.assertEqual(tr.peak_kv_usage, 0.0)
        self.assertEqual(tr.steps_per_second(), 0.0)
        self.assertEqual(tr.batch_size_histogram(), {})

    def test_averages(self):
        tr = self._tracker(4)
        self.assertEqual(len(tr), 4)
        self.assertAlmostEqual(tr.avg_batch_size, 2.5)
        self.assertAlmostEqual(tr.avg_running, 1.5)
        self.assertAlmostEqual(tr.avg_waiting, 2.5)
        self.assertAlmostEqual(tr.avg_kv_usage, 0.15)
        self.assertAlmostEqual(tr.peak_kv_usage, 0.3)

    def test_steps_per_second(self):
        tr = self._tracker(4)  # timestamps 0..3 -> 3 intervals over 3 seconds
        self.assertAlmostEqual(tr.steps_per_second(), 1.0)

    def test_window_is_bounded(self):
        tr = EngineStatsTracker(window=3)
        for i in range(10):
            tr.record(float(i), 1, 0, 0.5, 2, 20)
        self.assertEqual(len(tr), 3)
        self.assertEqual(tr.total_steps, 10)          # totals keep counting
        self.assertEqual(tr.total_batched_tokens, 200)

    def test_histogram(self):
        tr = EngineStatsTracker()
        for size in (1, 2, 2, 3, 2):
            tr.record(0.0, 1, 0, 0.1, size, size)
        self.assertEqual(tr.batch_size_histogram(), {1: 1, 2: 3, 3: 1})

    def test_to_dict_keys(self):
        d = self._tracker().to_dict()
        self.assertIn("avg_batch_size", d)
        self.assertIn("peak_kv_cache_usage", d)
        self.assertEqual(d["total_steps"], 4.0)

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            EngineStatsTracker(window=0)


if __name__ == "__main__":
    unittest.main()
