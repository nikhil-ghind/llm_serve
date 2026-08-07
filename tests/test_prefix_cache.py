import unittest

from llm_serve.engine.block_manager import BlockManager
from llm_serve.engine.prefix_cache import PrefixCache, hash_blocks


class TestHashBlocks(unittest.TestCase):
    def test_only_full_blocks_are_hashed(self):
        self.assertEqual(hash_blocks(list(range(15)), 4), hash_blocks(list(range(15)), 4))
        self.assertEqual(len(hash_blocks(list(range(15)), 4)), 3)
        self.assertEqual(len(hash_blocks(list(range(16)), 4)), 4)
        self.assertEqual(hash_blocks([1, 2], 4), [])

    def test_shared_prefix_produces_shared_hashes(self):
        a = hash_blocks([1, 2, 3, 4, 5, 6, 7, 8], 4)
        b = hash_blocks([1, 2, 3, 4, 9, 9, 9, 9], 4)
        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])

    def test_hash_is_chained_not_per_block(self):
        # Same second block, different first block -> different chained hash.
        a = hash_blocks([1, 1, 1, 1, 7, 7, 7, 7], 4)
        b = hash_blocks([2, 2, 2, 2, 7, 7, 7, 7], 4)
        self.assertNotEqual(a[1], b[1])

    def test_bad_block_size(self):
        with self.assertRaises(ValueError):
            hash_blocks([1, 2], 0)


class TestPrefixCache(unittest.TestCase):
    def setUp(self):
        self.bm = BlockManager(block_size=4, num_blocks=32, watermark=0.0)
        self.cache = PrefixCache(self.bm)

    def _publish(self, seq_id, tokens):
        res = self.bm.allocate(seq_id, len(tokens))
        self.cache.insert(tokens, res.block_ids)
        return res

    def test_miss_on_empty_cache(self):
        match = self.cache.lookup([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertFalse(match.hit)
        self.assertEqual(match.num_cached_tokens, 0)
        self.assertEqual(match.fraction, 0.0)

    def test_hit_on_shared_prefix(self):
        self._publish("a", list(range(16)))
        match = self.cache.lookup(list(range(8)) + [99, 99, 99, 99])
        self.assertTrue(match.hit)
        self.assertEqual(match.num_cached_blocks, 2)
        self.assertEqual(match.num_cached_tokens, 8)
        self.assertAlmostEqual(match.fraction, 8 / 12)

    def test_match_stops_at_first_miss(self):
        self._publish("a", [1, 2, 3, 4, 5, 6, 7, 8])
        match = self.cache.lookup([1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8])
        self.assertEqual(match.num_cached_blocks, 1)

    def test_identical_prompt_keeps_one_block_uncached(self):
        # The last block must be recomputed so there is a position to decode from.
        tokens = list(range(16))
        self._publish("a", tokens)
        match = self.cache.lookup(tokens)
        self.assertEqual(match.num_cached_blocks, 3)
        self.assertEqual(match.num_cached_tokens, 12)

    def test_reused_blocks_are_the_same_physical_blocks(self):
        res = self._publish("a", list(range(16)))
        match = self.cache.lookup(list(range(8)) + [42, 42, 42, 42])
        self.assertEqual(match.block_ids, res.block_ids[:2])

    def test_pinned_blocks_survive_owner_free(self):
        res = self._publish("a", list(range(16)))
        self.bm.free("a")
        self.assertEqual(self.bm.ref_count(res.block_ids[0]), 1)
        self.assertTrue(self.cache.lookup(list(range(16))).hit)

    def test_hit_rate_accounting(self):
        self._publish("a", list(range(16)))
        self.cache.lookup([90, 91, 92, 93])          # 1 block queried, 0 hits
        self.cache.lookup(list(range(8)) + [7, 7, 7, 7])  # 3 queried, 2 hits
        stats = self.cache.stats()
        self.assertEqual(stats.queried_blocks, 4)
        self.assertEqual(stats.hit_blocks, 2)
        self.assertAlmostEqual(stats.hit_rate, 0.5)
        self.assertEqual(stats.queries, 2)
        self.assertEqual(stats.queries_with_hit, 1)

    def test_disabled_cache_never_hits(self):
        cache = PrefixCache(self.bm, enabled=False)
        res = self.bm.allocate("a", 16)
        self.assertEqual(cache.insert(list(range(16)), res.block_ids), 0)
        self.assertFalse(cache.lookup(list(range(16))).hit)
        self.assertEqual(cache.hit_rate, 0.0)

    def test_lru_eviction_frees_blocks(self):
        cache = PrefixCache(self.bm, max_entries=2)
        r1 = self.bm.allocate("a", 8)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], r1.block_ids)
        self.bm.free("a")
        self.assertEqual(len(cache), 2)
        r2 = self.bm.allocate("b", 8)
        cache.insert([9, 9, 9, 9, 8, 8, 8, 8], r2.block_ids)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.stats().evictions, 2)
        # the evicted entries released their pins back to the pool
        self.assertEqual(self.bm.ref_count(r1.block_ids[0]), 0)

    def test_lookup_refreshes_lru_order(self):
        cache = PrefixCache(self.bm, max_entries=2)
        r1 = self.bm.allocate("a", 4)
        cache.insert([1, 2, 3, 4], r1.block_ids)
        r2 = self.bm.allocate("b", 4)
        cache.insert([5, 6, 7, 8], r2.block_ids)
        cache.lookup([1, 2, 3, 4, 0])  # touch entry 1
        r3 = self.bm.allocate("c", 4)
        cache.insert([9, 9, 9, 9], r3.block_ids)
        self.assertTrue(cache.lookup([1, 2, 3, 4, 0]).hit)
        self.assertFalse(cache.lookup([5, 6, 7, 8, 0]).hit)

    def test_reinsert_is_idempotent(self):
        res = self._publish("a", list(range(16)))
        added = self.cache.insert(list(range(16)), res.block_ids)
        self.assertEqual(added, 0)
        self.assertEqual(len(self.cache), 4)

    def test_clear_releases_all_pins(self):
        self._publish("a", list(range(16)))
        self.bm.free("a")
        self.cache.clear()
        self.assertEqual(self.bm.num_free_blocks, 32)
        self.assertEqual(len(self.cache), 0)


if __name__ == "__main__":
    unittest.main()
