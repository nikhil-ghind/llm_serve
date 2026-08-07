import unittest

from llm_serve.engine.block_manager import BlockAllocationError, BlockManager


class TestBlockMath(unittest.TestCase):
    def setUp(self):
        self.bm = BlockManager(block_size=16, num_blocks=64, watermark=0.0)

    def test_blocks_for_tokens_rounds_up(self):
        self.assertEqual(self.bm.blocks_for_tokens(0), 0)
        self.assertEqual(self.bm.blocks_for_tokens(1), 1)
        self.assertEqual(self.bm.blocks_for_tokens(16), 1)
        self.assertEqual(self.bm.blocks_for_tokens(17), 2)
        self.assertEqual(self.bm.blocks_for_tokens(160), 10)

    def test_invalid_construction(self):
        for kwargs in ({"block_size": 0}, {"num_blocks": 0}, {"watermark": 1.0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    BlockManager(**kwargs)


class TestAllocation(unittest.TestCase):
    def setUp(self):
        self.bm = BlockManager(block_size=16, num_blocks=10, watermark=0.0)

    def test_allocate_and_free(self):
        res = self.bm.allocate("a", 40)
        self.assertEqual(res.num_blocks, 3)
        self.assertEqual(res.new_blocks, 3)
        self.assertEqual(self.bm.num_free_blocks, 7)
        self.assertAlmostEqual(self.bm.usage, 0.3)
        self.assertEqual(self.bm.free("a"), 3)
        self.assertEqual(self.bm.num_free_blocks, 10)

    def test_duplicate_allocation_rejected(self):
        self.bm.allocate("a", 16)
        with self.assertRaises(BlockAllocationError):
            self.bm.allocate("a", 16)

    def test_out_of_memory(self):
        self.bm.allocate("a", 16 * 10)
        self.assertEqual(self.bm.num_free_blocks, 0)
        with self.assertRaises(BlockAllocationError):
            self.bm.allocate("b", 1)

    def test_append_allocates_only_on_block_boundary(self):
        self.bm.allocate("a", 16)  # exactly one full block
        self.assertEqual(self.bm.num_free_blocks, 9)
        new = self.bm.append_tokens("a", 1)  # spills into a second block
        self.assertEqual(len(new), 1)
        self.assertEqual(self.bm.num_free_blocks, 8)
        for _ in range(15):
            self.assertEqual(self.bm.append_tokens("a", 1), [])
        self.assertEqual(self.bm.num_free_blocks, 8)
        self.assertEqual(len(self.bm.append_tokens("a", 1)), 1)

    def test_append_to_unknown_sequence(self):
        with self.assertRaises(BlockAllocationError):
            self.bm.append_tokens("ghost")

    def test_append_out_of_memory(self):
        self.bm.allocate("a", 16 * 10)
        with self.assertRaises(BlockAllocationError):
            self.bm.append_tokens("a", 1)

    def test_can_allocate_respects_watermark(self):
        bm = BlockManager(block_size=16, num_blocks=10, watermark=0.2)
        self.assertEqual(bm.watermark_blocks, 2)
        self.assertTrue(bm.can_allocate(16 * 8))
        self.assertFalse(bm.can_allocate(16 * 9))

    def test_can_append_ignores_watermark(self):
        bm = BlockManager(block_size=16, num_blocks=10, watermark=0.5)
        bm.allocate("a", 16 * 4)
        self.assertTrue(bm.can_append("a", 1))

    def test_free_unknown_is_noop(self):
        self.assertEqual(self.bm.free("nobody"), 0)


class TestSharing(unittest.TestCase):
    def setUp(self):
        self.bm = BlockManager(block_size=4, num_blocks=16, watermark=0.0)

    def test_reused_blocks_are_shared_not_copied(self):
        first = self.bm.allocate("a", 16)  # 4 blocks
        shared = first.block_ids[:2]
        used_before = self.bm.num_used_blocks
        second = self.bm.allocate("b", 16, reused_block_ids=shared)
        self.assertEqual(second.reused_blocks, 2)
        self.assertEqual(second.new_blocks, 2)
        self.assertEqual(second.block_ids[:2], shared)
        # only two *new* blocks were consumed
        self.assertEqual(self.bm.num_used_blocks, used_before + 2)
        self.assertEqual(self.bm.ref_count(shared[0]), 2)
        self.assertEqual(self.bm.num_shared_blocks, 2)

    def test_freeing_one_owner_keeps_shared_blocks(self):
        first = self.bm.allocate("a", 16)
        self.bm.allocate("b", 16, reused_block_ids=first.block_ids[:2])
        reclaimed = self.bm.free("a")
        self.assertEqual(reclaimed, 2)  # only the unshared tail comes back
        self.assertEqual(self.bm.ref_count(first.block_ids[0]), 1)

    def test_reused_blocks_cannot_exceed_length(self):
        first = self.bm.allocate("a", 16)
        with self.assertRaises(BlockAllocationError):
            self.bm.allocate("b", 4, reused_block_ids=first.block_ids)

    def test_pin_and_unpin(self):
        res = self.bm.allocate("a", 4)
        block = res.block_ids[0]
        self.bm.pin(block)
        self.assertEqual(self.bm.free("a"), 0)  # pin keeps it alive
        self.assertEqual(self.bm.ref_count(block), 1)
        self.assertTrue(self.bm.unpin(block))
        self.assertEqual(self.bm.num_free_blocks, 16)

    def test_pin_unallocated_block_rejected(self):
        with self.assertRaises(BlockAllocationError):
            self.bm.pin(0)
        with self.assertRaises(BlockAllocationError):
            self.bm.pin(999)

    def test_double_free_detected(self):
        res = self.bm.allocate("a", 4)
        self.bm.free("a")
        with self.assertRaises(BlockAllocationError):
            self.bm.unpin(res.block_ids[0])


class TestStatsAndReset(unittest.TestCase):
    def test_stats(self):
        bm = BlockManager(block_size=8, num_blocks=8, watermark=0.0)
        bm.allocate("a", 16)
        stats = bm.stats()
        self.assertEqual(stats.total_blocks, 8)
        self.assertEqual(stats.used_blocks, 2)
        self.assertEqual(stats.free_blocks, 6)
        self.assertEqual(stats.allocated_sequences, 1)
        self.assertAlmostEqual(stats.usage, 0.25)

    def test_reset_returns_everything(self):
        bm = BlockManager(block_size=8, num_blocks=8, watermark=0.0)
        bm.allocate("a", 32)
        bm.reset()
        self.assertEqual(bm.num_free_blocks, 8)
        self.assertEqual(bm.block_table("a"), [])


if __name__ == "__main__":
    unittest.main()
