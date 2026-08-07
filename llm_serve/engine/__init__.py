"""Pure-Python engine internals: paged KV blocks, prefix cache, scheduler.

These model the admission and memory policy that a real engine applies on the
GPU. They are used directly when fronting an engine that has no scheduler of its
own, and they make the policy inspectable and unit-testable without CUDA.
"""

from .block_manager import AllocationResult, BlockAllocationError, BlockManager
from .prefix_cache import PrefixCache, PrefixMatch, hash_blocks
from .scheduler import (
    ScheduledPrefill,
    Scheduler,
    SchedulerOutput,
    SchedulerStats,
    SequenceGroup,
    SeqStatus,
)
from .stats import EngineStatsTracker, Snapshot

__all__ = [
    "AllocationResult",
    "BlockAllocationError",
    "BlockManager",
    "EngineStatsTracker",
    "PrefixCache",
    "PrefixMatch",
    "ScheduledPrefill",
    "Scheduler",
    "SchedulerOutput",
    "SchedulerStats",
    "SeqStatus",
    "SequenceGroup",
    "Snapshot",
    "hash_blocks",
]
