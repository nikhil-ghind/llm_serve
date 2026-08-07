"""Automatic prefix caching for the KV block pool.

Two requests that share a leading token span — a system prompt, a few-shot
preamble, the earlier turns of a chat — produce byte-identical KV for that span.
Prefix caching keeps those blocks around and hands them to the next request by
reference, so its prefill only computes the *novel* suffix. That is the single
biggest TTFT win for chat and RAG traffic.

Blocks are keyed by a **chained** hash: a block's key covers its own tokens *and*
every token before it. Without chaining, block ``k`` of two different prompts
could collide even though the preceding context differs, which would serve
incorrect KV.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

from .block_manager import BlockManager


@dataclass
class PrefixMatch:
    """Result of a cache lookup for one prompt."""

    block_ids: list[int]
    num_cached_tokens: int
    num_prompt_tokens: int

    @property
    def num_cached_blocks(self) -> int:
        return len(self.block_ids)

    @property
    def hit(self) -> bool:
        return bool(self.block_ids)

    @property
    def fraction(self) -> float:
        if self.num_prompt_tokens <= 0:
            return 0.0
        return self.num_cached_tokens / self.num_prompt_tokens


@dataclass
class PrefixCacheStats:
    entries: int
    queried_blocks: int
    hit_blocks: int
    hit_rate: float
    evictions: int
    queries: int
    queries_with_hit: int


def hash_blocks(
    token_ids: Sequence[int], block_size: int, prefix_seed: str = "llm_serve/v1"
) -> list[str]:
    """Chained hashes for each *full* block of ``token_ids``.

    A trailing partial block gets no hash: its KV is still being written, so it
    is not safe to share.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    hashes: list[str] = []
    parent = prefix_seed
    num_full = len(token_ids) // block_size
    for i in range(num_full):
        chunk = token_ids[i * block_size : (i + 1) * block_size]
        payload = parent.encode("utf-8") + b"|" + b",".join(str(t).encode("ascii") for t in chunk)
        parent = hashlib.blake2b(payload, digest_size=16).hexdigest()
        hashes.append(parent)
    return hashes


class PrefixCache:
    """LRU map from chained block hash to a pinned physical block id."""

    def __init__(
        self,
        block_manager: BlockManager,
        enabled: bool = True,
        max_entries: int | None = None,
    ) -> None:
        self.block_manager = block_manager
        self.enabled = enabled
        self.max_entries = max_entries
        self._entries: OrderedDict[str, int] = OrderedDict()
        self.queried_blocks = 0
        self.hit_blocks = 0
        self.evictions = 0
        self.queries = 0
        self.queries_with_hit = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        """Fraction of queried prompt blocks served from cache."""
        if not self.queried_blocks:
            return 0.0
        return self.hit_blocks / self.queried_blocks

    def lookup(self, token_ids: Sequence[int]) -> PrefixMatch:
        """Longest cached *prefix* of ``token_ids``, block-aligned.

        Matching stops at the first miss — KV for block ``k`` is only valid if
        every block before it is present too.
        """
        num_prompt = len(token_ids)
        if not self.enabled:
            return PrefixMatch([], 0, num_prompt)
        hashes = hash_blocks(token_ids, self.block_manager.block_size)
        self.queries += 1
        self.queried_blocks += len(hashes)
        matched: list[int] = []
        for h in hashes:
            block_id = self._entries.get(h)
            if block_id is None:
                break
            self._entries.move_to_end(h)
            matched.append(block_id)
        # Never reuse the entire prompt: at least one token must be recomputed so
        # the model has a position to generate the next token from.
        if matched and len(matched) * self.block_manager.block_size >= num_prompt:
            matched.pop()
        self.hit_blocks += len(matched)
        if matched:
            self.queries_with_hit += 1
        return PrefixMatch(matched, len(matched) * self.block_manager.block_size, num_prompt)

    def insert(self, token_ids: Sequence[int], block_ids: Sequence[int]) -> int:
        """Publish a sequence's full blocks for reuse. Returns entries added."""
        if not self.enabled:
            return 0
        hashes = hash_blocks(token_ids, self.block_manager.block_size)
        added = 0
        for h, block_id in zip(hashes, block_ids):
            if h in self._entries:
                self._entries.move_to_end(h)
                continue
            # Pin so freeing the owning sequence does not reclaim the block.
            self.block_manager.pin(block_id)
            self._entries[h] = block_id
            added += 1
        self._enforce_capacity()
        return added

    def _enforce_capacity(self) -> None:
        if self.max_entries is None:
            return
        while len(self._entries) > self.max_entries:
            self._evict_one()

    def evict(self, count: int = 1) -> int:
        """Drop up to ``count`` least-recently-used entries, freeing their pins."""
        freed = 0
        for _ in range(count):
            if not self._entries:
                break
            freed += 1 if self._evict_one() else 0
        return freed

    def _evict_one(self) -> bool:
        _, block_id = self._entries.popitem(last=False)
        self.evictions += 1
        return self.block_manager.unpin(block_id)

    def clear(self) -> None:
        while self._entries:
            self._evict_one()

    def stats(self) -> PrefixCacheStats:
        return PrefixCacheStats(
            entries=len(self._entries),
            queried_blocks=self.queried_blocks,
            hit_blocks=self.hit_blocks,
            hit_rate=self.hit_rate,
            evictions=self.evictions,
            queries=self.queries,
            queries_with_hit=self.queries_with_hit,
        )
