import unittest

from llm_serve.engine.scheduler import Scheduler, SeqStatus


def prompt(n, offset=0):
    return list(range(offset, offset + n))


class TestAdmission(unittest.TestCase):
    def _sched(self, **kw):
        defaults = dict(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            block_size=4,
            num_gpu_blocks=64,
            watermark=0.0,
            enable_prefix_caching=False,
        )
        defaults.update(kw)
        return Scheduler(**defaults)

    def test_empty_scheduler_produces_empty_batch(self):
        sched = self._sched()
        out = sched.step()
        self.assertTrue(out.is_empty)
        self.assertFalse(sched.has_work())

    def test_prompt_prefills_then_decodes(self):
        sched = self._sched()
        sched.add_request(prompt(16), max_tokens=3, seq_id="s0")
        first = sched.step()
        self.assertEqual(len(first.prefills), 1)
        self.assertEqual(first.prefills[0].num_tokens, 16)
        self.assertFalse(first.prefills[0].is_chunk)
        self.assertEqual(first.decodes, [])

        second = sched.step()
        self.assertEqual(second.decodes, ["s0"])
        self.assertEqual(second.prefills, [])

    def test_sequence_finishes_at_max_tokens(self):
        sched = self._sched()
        sched.add_request(prompt(8), max_tokens=2, seq_id="s0")
        outs = sched.run_until_idle()
        finished = [sid for o in outs for sid in o.finished]
        self.assertEqual(finished, ["s0"])
        self.assertIs(sched.get("s0").status, SeqStatus.FINISHED)
        self.assertEqual(sched.get("s0").num_generated, 2)
        self.assertEqual(sched.block_manager.num_used_blocks, 0)

    def test_max_num_seqs_caps_concurrency(self):
        sched = self._sched(max_num_seqs=2)
        for i in range(5):
            sched.add_request(prompt(4, i * 100), max_tokens=4, seq_id=f"s{i}")
        out = sched.step()
        self.assertEqual(len(out.prefills), 2)
        self.assertEqual(len(sched.running), 2)
        self.assertEqual(len(sched.waiting), 3)

    def test_token_budget_caps_batch(self):
        sched = self._sched(max_num_batched_tokens=16, enable_chunked_prefill=False)
        for i in range(4):
            sched.add_request(prompt(8, i * 100), max_tokens=2, seq_id=f"s{i}")
        out = sched.step()
        self.assertEqual(out.batched_tokens, 16)
        self.assertEqual(len(out.prefills), 2)

    def test_duplicate_seq_id_rejected(self):
        sched = self._sched()
        sched.add_request(prompt(4), seq_id="dup")
        with self.assertRaises(ValueError):
            sched.add_request(prompt(4), seq_id="dup")

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValueError):
            self._sched().add_request([])

    def test_waiting_queue_bound(self):
        sched = self._sched(max_waiting=2)
        sched.add_request(prompt(4, 0))
        sched.add_request(prompt(4, 10))
        with self.assertRaises(RuntimeError):
            sched.add_request(prompt(4, 20))

    def test_all_requests_complete_under_load(self):
        sched = self._sched(max_num_seqs=3, max_num_batched_tokens=32, num_gpu_blocks=32)
        for i in range(12):
            sched.add_request(prompt(12, i * 1000), max_tokens=5, seq_id=f"s{i}")
        sched.run_until_idle()
        self.assertEqual(sched.finished_count, 12)
        self.assertFalse(sched.has_work())
        self.assertEqual(sched.block_manager.num_used_blocks, 0)
        self.assertEqual(sched.total_generation_tokens, 60)


class TestChunkedPrefill(unittest.TestCase):
    def test_long_prompt_is_split_across_steps(self):
        sched = Scheduler(
            max_num_seqs=4,
            max_num_batched_tokens=8,
            block_size=4,
            num_gpu_blocks=64,
            watermark=0.0,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
        )
        sched.add_request(prompt(20), max_tokens=1, seq_id="long")
        first = sched.step()
        self.assertEqual(first.prefills[0].num_tokens, 8)
        self.assertTrue(first.prefills[0].is_chunk)
        second = sched.step()
        self.assertEqual(second.prefills[0].num_tokens, 8)
        third = sched.step()
        self.assertEqual(third.prefills[0].num_tokens, 4)
        self.assertFalse(third.prefills[0].is_chunk)
        self.assertTrue(sched.get("long").prefill_done)

    def test_disabled_chunking_waits_for_full_budget(self):
        sched = Scheduler(
            max_num_seqs=4,
            max_num_batched_tokens=8,
            block_size=4,
            num_gpu_blocks=64,
            watermark=0.0,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
        )
        sched.add_request(prompt(20), max_tokens=1, seq_id="long")
        out = sched.step()
        self.assertEqual(out.prefills, [])
        self.assertEqual(len(sched.waiting), 1)

    def test_decode_of_running_seq_shares_budget_with_prefill(self):
        sched = Scheduler(
            max_num_seqs=4,
            max_num_batched_tokens=10,
            block_size=4,
            num_gpu_blocks=64,
            watermark=0.0,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
        )
        sched.add_request(prompt(8, 0), max_tokens=5, seq_id="a")
        sched.step()  # a prefills (8 tokens)
        sched.add_request(prompt(20, 500), max_tokens=1, seq_id="b")
        out = sched.step()
        self.assertEqual(out.decodes, ["a"])           # 1 token for the decode
        self.assertEqual(out.prefills[0].num_tokens, 9)  # remaining budget to prefill
        self.assertEqual(out.batched_tokens, 10)


class TestPreemption(unittest.TestCase):
    def _tight(self, mode="recompute"):
        return Scheduler(
            max_num_seqs=8,
            max_num_batched_tokens=256,
            block_size=4,
            num_gpu_blocks=6,
            watermark=0.0,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            preemption_mode=mode,
        )

    def test_preemption_happens_when_kv_runs_out(self):
        sched = self._tight()
        for i in range(4):
            sched.add_request(prompt(8, i * 1000), max_tokens=8, seq_id=f"s{i}")
        sched.run_until_idle()
        self.assertGreater(sched.total_preemptions, 0)
        self.assertEqual(sched.finished_count, 4)

    def test_recompute_preemption_requeues_at_head(self):
        sched = self._tight()
        for i in range(3):
            sched.add_request(prompt(8, i * 1000), max_tokens=20, seq_id=f"s{i}")
        preempted = []
        for _ in range(40):
            out = sched.step()
            preempted.extend(out.preempted)
            if preempted:
                break
        self.assertTrue(preempted)
        victim = sched.get(preempted[0])
        self.assertEqual(victim.num_generated, 0)
        self.assertGreaterEqual(victim.preemption_count, 1)
        self.assertIn(victim, sched.waiting)

    def test_swap_preemption_keeps_progress_and_resumes(self):
        sched = self._tight(mode="swap")
        for i in range(4):
            sched.add_request(prompt(8, i * 1000), max_tokens=6, seq_id=f"s{i}")
        seen_swapped = False
        for _ in range(200):
            out = sched.step()
            if out.preempted:
                seen_swapped = True
            if not sched.has_work():
                break
        self.assertTrue(seen_swapped)
        self.assertEqual(sched.finished_count, 4)
        self.assertEqual(len(sched.swapped), 0)

    def test_newest_sequence_is_preempted_first(self):
        sched = self._tight()
        for i in range(3):
            sched.add_request(prompt(4, i * 1000), max_tokens=30, seq_id=f"s{i}")
        for _ in range(50):
            out = sched.step()
            if out.preempted:
                self.assertEqual(out.preempted[0], "s2")
                return
        self.fail("expected a preemption")

    def test_oversized_prompt_is_detected_not_deadlocked(self):
        sched = self._tight()
        sched.add_request(prompt(1000), max_tokens=1, seq_id="huge")
        self.assertFalse(sched.waiting_can_progress())
        with self.assertRaises(RuntimeError):
            sched.run_until_idle(max_steps=10)


class TestPrefixCacheIntegration(unittest.TestCase):
    def _sched(self):
        return Scheduler(
            max_num_seqs=4,
            max_num_batched_tokens=256,
            block_size=4,
            num_gpu_blocks=256,
            watermark=0.0,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
        )

    def test_shared_prefix_shrinks_the_second_prefill(self):
        sched = self._sched()
        shared = prompt(32)
        sched.add_request(shared + [900, 901, 902, 903], max_tokens=1, seq_id="a")
        first = sched.step()
        self.assertEqual(first.prefills[0].cached_tokens, 0)
        self.assertEqual(first.prefills[0].num_tokens, 36)
        sched.run_until_idle()

        sched.add_request(shared + [910, 911, 912, 913], max_tokens=1, seq_id="b")
        second = sched.step()
        self.assertEqual(second.prefills[0].cached_tokens, 32)
        self.assertEqual(second.prefills[0].num_tokens, 4)
        self.assertGreater(sched.prefix_cache.hit_rate, 0.0)

    def test_no_reuse_when_caching_disabled(self):
        sched = Scheduler(
            max_num_seqs=4,
            max_num_batched_tokens=256,
            block_size=4,
            num_gpu_blocks=256,
            watermark=0.0,
            enable_prefix_caching=False,
        )
        shared = prompt(32)
        sched.add_request(shared + [900], max_tokens=1, seq_id="a")
        sched.run_until_idle()
        sched.add_request(shared + [910], max_tokens=1, seq_id="b")
        out = sched.step()
        self.assertEqual(out.prefills[0].cached_tokens, 0)
        self.assertEqual(out.prefills[0].num_tokens, 33)

    def test_cached_tokens_are_not_counted_as_computed_prompt_tokens(self):
        sched = self._sched()
        shared = prompt(32)
        sched.add_request(shared + [900, 901, 902, 903], max_tokens=1, seq_id="a")
        sched.run_until_idle()
        before = sched.total_prompt_tokens
        sched.add_request(shared + [910, 911, 912, 913], max_tokens=1, seq_id="b")
        sched.run_until_idle()
        self.assertEqual(sched.total_prompt_tokens - before, 4)


class TestLifecycleAndStats(unittest.TestCase):
    def _sched(self):
        return Scheduler(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            block_size=4,
            num_gpu_blocks=64,
            watermark=0.0,
            enable_prefix_caching=False,
        )

    def test_abort_releases_blocks(self):
        sched = self._sched()
        sched.add_request(prompt(16), max_tokens=100, seq_id="s0")
        sched.step()
        self.assertGreater(sched.block_manager.num_used_blocks, 0)
        self.assertTrue(sched.abort("s0"))
        self.assertEqual(sched.block_manager.num_used_blocks, 0)
        self.assertIs(sched.get("s0").status, SeqStatus.ABORTED)
        self.assertFalse(sched.has_work())

    def test_abort_unknown_or_finished(self):
        sched = self._sched()
        self.assertFalse(sched.abort("nope"))
        sched.add_request(prompt(4), max_tokens=1, seq_id="s0")
        sched.run_until_idle()
        self.assertFalse(sched.abort("s0"))

    def test_early_finish_from_stop_token(self):
        sched = self._sched()
        sched.add_request(prompt(8), max_tokens=100, seq_id="s0")
        sched.step()
        sched.finish("s0")
        self.assertIs(sched.get("s0").status, SeqStatus.FINISHED)
        self.assertEqual(sched.finished_count, 1)
        self.assertFalse(sched.has_work())

    def test_stats_snapshot(self):
        sched = self._sched()
        for i in range(3):
            sched.add_request(prompt(8, i * 100), max_tokens=4, seq_id=f"s{i}")
        sched.run_until_idle()
        stats = sched.stats()
        self.assertEqual(stats.finished, 3)
        self.assertEqual(stats.running, 0)
        self.assertEqual(stats.waiting, 0)
        self.assertEqual(stats.total_generation_tokens, 12)
        self.assertEqual(stats.total_prompt_tokens, 24)
        self.assertGreater(stats.avg_batch_size, 0)
        self.assertGreaterEqual(stats.max_batch_size, 3)
        self.assertEqual(stats.kv_cache_usage, 0.0)

    def test_invalid_preemption_mode(self):
        with self.assertRaises(ValueError):
            Scheduler(preemption_mode="teleport")


if __name__ == "__main__":
    unittest.main()
