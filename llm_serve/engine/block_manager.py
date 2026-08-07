"""Paged KV-cache block manager.

Models what PagedAttention does on the GPU: the KV cache is a pool of fixed-size
blocks (``block_size`` tokens each) rather than one contiguous buffer per
sequence. Sequences hold a *block table* — an ordered list of physical block ids —
so memory is allocated a block at a time instead of reserving ``max_model_len``
up front, and blocks holding an identical prefix can be shared between sequences
through reference counting.

Pure Python and dependency-free so the admission policy is unit-testable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BlockAllocationError(RuntimeError):
    """Raised when the block pool cannot satisfy an allocation."""


@dataclass
class AllocationResult:
    """Outcome of allocating blocks for a sequence."""

    seq_id: str
    block_ids: list[int]
    reused_blocks: int = 0
    new_blocks: int = 0

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)


@dataclass
class BlockManagerStats:
    total_blocks: int
    free_blocks: int
    used_blocks: int
    shared_blocks: int
    usage: float
    allocated_sequences: int
    total_allocations: int = 0
    total_reused_blocks: int = 0
    field_padding: dict[str, float] = field(default_factory=dict)


class BlockManager:
    """Allocates, shares and frees fixed-size KV blocks.

    Parameters
    ----------
    block_size:
        Tokens stored per block (16 matches the vLLM default).
    num_blocks:
        Size of the physical block pool. On a real GPU this is derived from free
        VRAM after weights are loaded, divided by the per-block KV byte size.
    watermark:
        Fraction of the pool kept in reserve. Admission of a *new* sequence must
        leave at least this many blocks free, which stops the scheduler from
        filling the cache so completely that running sequences cannot grow and
        immediately thrash into preemption.
    """

    def __init__(self, block_size: int = 16, num_blocks: int = 1024, watermark: float = 0.01) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if not 0.0 <= watermark < 1.0:
            raise ValueError("watermark must be in [0, 1)")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.watermark_blocks = int(watermark * num_blocks)
        # Free list ordered so the lowest ids are handed out first (stable tests).
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self._ref_counts: list[int] = [0] * num_blocks
        self._tables: dict[str, list[int]] = {}
        self._lengths: dict[str, int] = {}
        self.total_allocations = 0
        self.total_reused_blocks = 0

    # ---------------------------------------------------------------- queries

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def usage(self) -> float:
        """Fraction of the pool currently allocated, in [0, 1]."""
        return self.num_used_blocks / self.num_blocks

    @property
    def num_shared_blocks(self) -> int:
        return sum(1 for rc in self._ref_counts if rc > 1)

    def blocks_for_tokens(self, num_tokens: int) -> int:
        """Blocks needed to hold ``num_tokens`` tokens (ceiling division)."""
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def ref_count(self, block_id: int) -> int:
        return self._ref_counts[block_id]

    def block_table(self, seq_id: str) -> list[int]:
        return list(self._tables.get(seq_id, ()))

    def seq_length(self, seq_id: str) -> int:
        return self._lengths.get(seq_id, 0)

    def can_allocate(self, num_tokens: int, reused_blocks: int = 0) -> bool:
        """True if a *new* sequence fits while respecting the watermark."""
        needed = max(0, self.blocks_for_tokens(num_tokens) - reused_blocks)
        return self.num_free_blocks - needed >= self.watermark_blocks

    def can_append(self, seq_id: str, num_tokens: int = 1) -> bool:
        """True if ``seq_id`` can grow by ``num_tokens`` (no watermark reserve).

        Running sequences may dip into the watermark: evicting them would waste
        the work already spent on their prefill.
        """
        return self.num_free_blocks >= self._blocks_needed_to_append(seq_id, num_tokens)

    def _blocks_needed_to_append(self, seq_id: str, num_tokens: int) -> int:
        current = self._lengths.get(seq_id, 0)
        have = len(self._tables.get(seq_id, ()))
        return max(0, self.blocks_for_tokens(current + num_tokens) - have)

    # ------------------------------------------------------------ allocation

    def allocate(
        self, seq_id: str, num_tokens: int, reused_block_ids: list[int] | None = None
    ) -> AllocationResult:
        """Allocate a block table for ``seq_id``.

        ``reused_block_ids`` come from the prefix cache: they are adopted by
        reference instead of being copied, so a shared system prompt costs one
        copy of KV memory no matter how many sequences use it.
        """
        if seq_id in self._tables:
            raise BlockAllocationError(f"sequence {seq_id!r} already has a block table")
        reused = list(reused_block_ids or ())
        if len(reused) * self.block_size > num_tokens:
            raise BlockAllocationError("reused blocks exceed the sequence length")
        needed = self.blocks_for_tokens(num_tokens) - len(reused)
        if needed > self.num_free_blocks:
            raise BlockAllocationError(
                f"need {needed} blocks for {seq_id!r} but only {self.num_free_blocks} are free"
            )
        table = []
        for block_id in reused:
            self._ref_counts[block_id] += 1
            table.append(block_id)
        for _ in range(needed):
            block_id = self._free.pop()
            self._ref_counts[block_id] += 1
            table.append(block_id)
        self._tables[seq_id] = table
        self._lengths[seq_id] = num_tokens
        self.total_allocations += 1
        self.total_reused_blocks += len(reused)
        return AllocationResult(seq_id, list(table), reused_blocks=len(reused), new_blocks=needed)

    def append_tokens(self, seq_id: str, num_tokens: int = 1) -> list[int]:
        """Grow a sequence, returning any newly appended block ids."""
        if seq_id not in self._tables:
            raise BlockAllocationError(f"unknown sequence {seq_id!r}")
        needed = self._blocks_needed_to_append(seq_id, num_tokens)
        if needed > self.num_free_blocks:
            raise BlockAllocationError(
                f"cannot grow {seq_id!r} by {num_tokens} token(s): out of KV blocks"
            )
        new_blocks = []
        for _ in range(needed):
            block_id = self._free.pop()
            self._ref_counts[block_id] += 1
            self._tables[seq_id].append(block_id)
            new_blocks.append(block_id)
        self._lengths[seq_id] += num_tokens
        return new_blocks

    def free(self, seq_id: str) -> int:
        """Release a sequence's blocks. Returns the number actually reclaimed.

        Blocks still referenced elsewhere (a sibling sequence, or the prefix
        cache holding them for reuse) stay allocated.
        """
        table = self._tables.pop(seq_id, None)
        if table is None:
            return 0
        self._lengths.pop(seq_id, None)
        reclaimed = 0
        for block_id in table:
            if self._release_block(block_id):
                reclaimed += 1
        return reclaimed

    def pin(self, block_id: int) -> None:
        """Take an extra reference, e.g. when the prefix cache adopts a block."""
        if not 0 <= block_id < self.num_blocks:
            raise BlockAllocationError(f"block {block_id} is out of range")
        if self._ref_counts[block_id] == 0:
            raise BlockAllocationError(f"cannot pin unallocated block {block_id}")
        self._ref_counts[block_id] += 1

    def unpin(self, block_id: int) -> bool:
        """Drop a reference taken with :meth:`pin`; True if the block was freed."""
        return self._release_block(block_id)

    def _release_block(self, block_id: int) -> bool:
        if self._ref_counts[block_id] <= 0:
            raise BlockAllocationError(f"double free of block {block_id}")
        self._ref_counts[block_id] -= 1
        if self._ref_counts[block_id] == 0:
            self._free.append(block_id)
            return True
        return False

    def reset(self) -> None:
        self._free = list(range(self.num_blocks - 1, -1, -1))
        self._ref_counts = [0] * self.num_blocks
        self._tables.clear()
        self._lengths.clear()

    def stats(self) -> BlockManagerStats:
        return BlockManagerStats(
            total_blocks=self.num_blocks,
            free_blocks=self.num_free_blocks,
            used_blocks=self.num_used_blocks,
            shared_blocks=self.num_shared_blocks,
            usage=self.usage,
            allocated_sequences=len(self._tables),
            total_allocations=self.total_allocations,
            total_reused_blocks=self.total_reused_blocks,
        )
