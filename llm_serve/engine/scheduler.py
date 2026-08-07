"""Continuous-batching scheduler.

Static batching makes every request in a batch wait for the slowest one to
finish. Continuous batching (a.k.a. iteration-level scheduling) instead rebuilds
the batch on *every decode step*: a sequence that hits its stop token leaves
immediately and a waiting request takes its slot in the same step. That is what
keeps GPU utilization high and queueing delay low under bursty traffic.

Each :meth:`Scheduler.step` decides, under a token budget and a sequence-count
cap, which sequences prefill (possibly *chunked*, a slice at a time, so one long
prompt cannot stall every decode in flight) and which decode. When the KV pool
runs dry, running sequences are preempted newest-first and re-queued.

Pure Python: no GPU, no engine. The vLLM backend delegates admission to vLLM's
own scheduler; the Triton and Ray fronting layers use this one directly, and it
is what the unit tests exercise.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from .block_manager import BlockAllocationError, BlockManager
from .prefix_cache import PrefixCache


def _priority(seq: "SequenceGroup") -> tuple[float, str]:
    """FCFS ordering key: earlier arrival wins, seq id breaks ties."""
    return (seq.arrival_time, seq.seq_id)


class SeqStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class SequenceGroup:
    """One request as tracked by the scheduler."""

    seq_id: str
    prompt_token_ids: list[int]
    max_tokens: int = 128
    arrival_time: float = field(default_factory=time.monotonic)
    status: SeqStatus = SeqStatus.WAITING
    num_prefilled: int = 0
    num_generated: int = 0
    num_cached_tokens: int = 0
    preemption_count: int = 0
    first_scheduled_time: float | None = None
    finish_time: float | None = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def prefill_done(self) -> bool:
        return self.num_prefilled >= self.prompt_len

    @property
    def num_computed_tokens(self) -> int:
        return self.num_prefilled + self.num_generated

    @property
    def remaining_prefill(self) -> int:
        return max(0, self.prompt_len - self.num_prefilled)

    @property
    def is_finished(self) -> bool:
        return self.status in (SeqStatus.FINISHED, SeqStatus.ABORTED)

    def reset_for_recompute(self) -> None:
        """Drop computed KV, keeping cached-prefix credit for the retry."""
        self.num_prefilled = self.num_cached_tokens
        self.num_generated = 0
        self.preemption_count += 1
        self.status = SeqStatus.WAITING


@dataclass
class ScheduledPrefill:
    seq_id: str
    num_tokens: int
    is_chunk: bool = False
    cached_tokens: int = 0


@dataclass
class SchedulerOutput:
    """The batch to run for one engine step."""

    step: int
    prefills: list[ScheduledPrefill] = field(default_factory=list)
    decodes: list[str] = field(default_factory=list)
    preempted: list[str] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)
    batched_tokens: int = 0

    @property
    def batch_size(self) -> int:
        return len(self.prefills) + len(self.decodes)

    @property
    def is_empty(self) -> bool:
        return self.batch_size == 0


@dataclass
class SchedulerStats:
    step: int
    running: int
    waiting: int
    swapped: int
    finished: int
    kv_cache_usage: float
    prefix_cache_hit_rate: float
    preemptions: int
    total_prompt_tokens: int
    total_generation_tokens: int
    avg_batch_size: float
    max_batch_size: int


class Scheduler:
    """FCFS continuous-batching scheduler with chunked prefill and preemption."""

    def __init__(
        self,
        *,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        block_size: int = 16,
        num_gpu_blocks: int = 1024,
        watermark: float = 0.01,
        enable_chunked_prefill: bool = True,
        enable_prefix_caching: bool = True,
        max_waiting: int = 2048,
        preemption_mode: str = "recompute",
    ) -> None:
        if preemption_mode not in ("recompute", "swap"):
            raise ValueError("preemption_mode must be 'recompute' or 'swap'")
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_waiting = max_waiting
        self.preemption_mode = preemption_mode

        self.block_manager = BlockManager(block_size, num_gpu_blocks, watermark)
        self.prefix_cache = PrefixCache(self.block_manager, enabled=enable_prefix_caching)

        self.waiting: list[SequenceGroup] = []
        self.running: list[SequenceGroup] = []
        self.swapped: list[SequenceGroup] = []
        self._by_id: dict[str, SequenceGroup] = {}

        self.step_count = 0
        self.total_preemptions = 0
        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        self.finished_count = 0
        self._batch_sizes: list[int] = []
        self._ids = itertools.count()

    # ------------------------------------------------------------ admission

    def add_request(
        self,
        prompt_token_ids: Sequence[int],
        max_tokens: int = 128,
        seq_id: str | None = None,
        arrival_time: float | None = None,
    ) -> SequenceGroup:
        if len(self.waiting) >= self.max_waiting:
            raise RuntimeError(
                f"waiting queue is full ({self.max_waiting}); shed load or raise max_waiting"
            )
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        seq = SequenceGroup(
            seq_id=seq_id or f"seq-{next(self._ids)}",
            prompt_token_ids=list(prompt_token_ids),
            max_tokens=max_tokens,
            arrival_time=arrival_time if arrival_time is not None else time.monotonic(),
        )
        if seq.seq_id in self._by_id:
            raise ValueError(f"duplicate seq_id {seq.seq_id!r}")
        self.waiting.append(seq)
        self._by_id[seq.seq_id] = seq
        return seq

    def get(self, seq_id: str) -> SequenceGroup | None:
        return self._by_id.get(seq_id)

    @property
    def num_unfinished(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def has_work(self) -> bool:
        return self.num_unfinished > 0

    # ----------------------------------------------------------------- step

    def step(self) -> SchedulerOutput:
        """Build the batch for one engine iteration."""
        self.step_count += 1
        out = SchedulerOutput(step=self.step_count)
        budget = self.max_num_batched_tokens

        budget = self._schedule_decodes(out, budget)
        self._schedule_swapped()
        budget = self._schedule_prefills(out, budget)

        out.batched_tokens = self.max_num_batched_tokens - budget
        self._batch_sizes.append(out.batch_size)
        return out

    def _schedule_decodes(self, out: SchedulerOutput, budget: int) -> int:
        """Continue every running sequence; preempt newest-first if KV runs out."""
        for seq in list(self.running):
            # A sequence can leave `running` mid-loop when it is chosen as a
            # preemption victim for an earlier one; skip anything that moved.
            if seq.status is not SeqStatus.RUNNING or seq not in self.running:
                continue
            if seq.is_finished or not seq.prefill_done:
                continue
            if budget < 1:
                break
            while not self.block_manager.can_append(seq.seq_id, 1):
                victim = self._pick_preemption_victim(exclude=seq.seq_id)
                if victim is None or _priority(victim) < _priority(seq):
                    # Nothing newer than this sequence to give up its blocks, so
                    # this one yields. Preempting an older sequence instead would
                    # let two requests trade places forever without finishing.
                    victim = seq
                self._preempt(victim, out)
                if victim is seq:
                    break
            if seq.status is not SeqStatus.RUNNING:
                continue
            self.block_manager.append_tokens(seq.seq_id, 1)
            seq.num_generated += 1
            self.total_generation_tokens += 1
            out.decodes.append(seq.seq_id)
            budget -= 1
            if seq.num_generated >= seq.max_tokens:
                self._finish(seq, out)
        return budget

    def _schedule_swapped(self) -> None:
        """Bring swapped-out sequences back once their KV blocks fit again.

        Under ``swap`` preemption the sequence keeps its computed-token count, so
        resuming costs a block re-allocation rather than a full re-prefill.
        """
        while self.swapped and len(self.running) < self.max_num_seqs:
            seq = self.swapped[0]
            if not self.block_manager.can_allocate(seq.num_computed_tokens):
                break
            self.swapped.pop(0)
            self.block_manager.allocate(seq.seq_id, max(1, seq.num_computed_tokens))
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)

    def _schedule_prefills(self, out: SchedulerOutput, budget: int) -> int:
        """Admit waiting requests, and advance any in-flight chunked prefills."""
        # In-flight chunked prefills continue before new admissions so a partly
        # prefilled sequence cannot be starved by newer arrivals.
        for seq in list(self.running):
            if seq.status is not SeqStatus.RUNNING or seq not in self.running:
                continue
            if seq.prefill_done or seq.is_finished or budget < 1:
                continue
            chunk = min(seq.remaining_prefill, budget)
            if not self.enable_chunked_prefill:
                chunk = seq.remaining_prefill
                if chunk > budget:
                    continue
            try:
                self.block_manager.append_tokens(seq.seq_id, chunk)
            except BlockAllocationError:
                self._preempt(seq, out)
                continue
            seq.num_prefilled += chunk
            budget -= chunk
            out.prefills.append(
                ScheduledPrefill(seq.seq_id, chunk, is_chunk=not seq.prefill_done)
            )

        preempted_this_step = set(out.preempted)
        while self.waiting and len(self.running) < self.max_num_seqs and budget >= 1:
            seq = self.waiting[0]
            if seq.seq_id in preempted_this_step:
                # It was evicted a moment ago for lack of memory; re-admitting it
                # in the same step would only evict it again.
                break
            match = self.prefix_cache.lookup(seq.prompt_token_ids)
            uncached = seq.prompt_len - match.num_cached_tokens
            chunk = min(uncached, budget)
            if not self.enable_chunked_prefill and uncached > budget:
                break  # cannot fit this prompt whole in the remaining budget
            if chunk < 1:
                break
            if not self.block_manager.can_allocate(seq.prompt_len, match.num_cached_blocks):
                # Never evict a running sequence to admit a queued one: the work
                # already spent on its prefill would be thrown away, and the two
                # would trade places every step without either finishing.
                break
            self.waiting.pop(0)
            try:
                self.block_manager.allocate(
                    seq.seq_id,
                    match.num_cached_tokens + chunk,
                    reused_block_ids=match.block_ids,
                )
            except BlockAllocationError:
                self.waiting.insert(0, seq)
                break
            seq.num_cached_tokens = match.num_cached_tokens
            seq.num_prefilled = match.num_cached_tokens + chunk
            seq.status = SeqStatus.RUNNING
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = time.monotonic()
            self.running.append(seq)
            self.total_prompt_tokens += chunk
            budget -= chunk
            out.prefills.append(
                ScheduledPrefill(
                    seq.seq_id,
                    chunk,
                    is_chunk=not seq.prefill_done,
                    cached_tokens=match.num_cached_tokens,
                )
            )
            if seq.prefill_done:
                self.prefix_cache.insert(
                    seq.prompt_token_ids, self.block_manager.block_table(seq.seq_id)
                )
        return budget

    # ----------------------------------------------------------- preemption

    def _pick_preemption_victim(self, exclude: str | None = None) -> SequenceGroup | None:
        """Newest running sequence — it has the least work invested in it.

        Choosing by arrival time (rather than position in ``running``) keeps the
        policy strictly FCFS: the oldest request in flight is never the victim,
        which is what guarantees the queue drains under sustained memory
        pressure instead of livelocking.
        """
        candidates = [s for s in self.running if s.seq_id != exclude and not s.is_finished]
        if not candidates:
            return None
        return max(candidates, key=_priority)

    def _preempt(self, seq: SequenceGroup, out: SchedulerOutput) -> None:
        self.block_manager.free(seq.seq_id)
        if seq in self.running:
            self.running.remove(seq)
        self.total_preemptions += 1
        out.preempted.append(seq.seq_id)
        if self.preemption_mode == "swap":
            seq.status = SeqStatus.SWAPPED
            seq.preemption_count += 1
            self.swapped.append(seq)
        else:
            seq.reset_for_recompute()
            self.waiting.insert(0, seq)

    # -------------------------------------------------------------- lifecycle

    def _finish(self, seq: SequenceGroup, out: SchedulerOutput) -> None:
        if seq.prefill_done:
            self.prefix_cache.insert(
                seq.prompt_token_ids, self.block_manager.block_table(seq.seq_id)
            )
        self.block_manager.free(seq.seq_id)
        if seq in self.running:
            self.running.remove(seq)
        seq.status = SeqStatus.FINISHED
        seq.finish_time = time.monotonic()
        self.finished_count += 1
        out.finished.append(seq.seq_id)

    def finish(self, seq_id: str) -> None:
        """Mark a sequence complete early (stop string / EOS from the engine)."""
        seq = self._by_id.get(seq_id)
        if seq is None or seq.is_finished:
            return
        self._finish(seq, SchedulerOutput(step=self.step_count))

    def abort(self, seq_id: str) -> bool:
        """Cancel a request wherever it currently sits."""
        seq = self._by_id.get(seq_id)
        if seq is None or seq.is_finished:
            return False
        self.block_manager.free(seq_id)
        for queue in (self.waiting, self.running, self.swapped):
            if seq in queue:
                queue.remove(seq)
        seq.status = SeqStatus.ABORTED
        seq.finish_time = time.monotonic()
        return True

    def run_until_idle(self, max_steps: int = 100_000) -> list[SchedulerOutput]:
        """Drive steps until every queue drains. Used by tests and simulations."""
        outputs: list[SchedulerOutput] = []
        for _ in range(max_steps):
            if not self.has_work():
                break
            out = self.step()
            outputs.append(out)
            if out.is_empty and not self.waiting_can_progress():
                raise RuntimeError("scheduler deadlocked: no batch can be formed")
        return outputs

    def waiting_can_progress(self) -> bool:
        """False when the pool is too small to ever admit the head of the queue."""
        if not self.waiting:
            return bool(self.running or self.swapped)
        head = self.waiting[0]
        need = self.block_manager.blocks_for_tokens(head.prompt_len)
        return need <= self.block_manager.num_blocks - self.block_manager.watermark_blocks

    # ------------------------------------------------------------------ stats

    def stats(self) -> SchedulerStats:
        sizes: Iterable[int] = self._batch_sizes
        n = len(self._batch_sizes)
        return SchedulerStats(
            step=self.step_count,
            running=len(self.running),
            waiting=len(self.waiting),
            swapped=len(self.swapped),
            finished=self.finished_count,
            kv_cache_usage=self.block_manager.usage,
            prefix_cache_hit_rate=self.prefix_cache.hit_rate,
            preemptions=self.total_preemptions,
            total_prompt_tokens=self.total_prompt_tokens,
            total_generation_tokens=self.total_generation_tokens,
            avg_batch_size=(sum(sizes) / n) if n else 0.0,
            max_batch_size=max(self._batch_sizes) if n else 0,
        )
