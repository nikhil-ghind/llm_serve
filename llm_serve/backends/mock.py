"""CPU-only mock backend.

Produces a deterministic pseudo-token stream with configurable prefill and decode
delays. It exists so the API server, load generator, aggregation and report
pipeline can be exercised end to end — and unit tested — on a machine with no GPU
and no model weights. The token stream is seeded from the request, so the same
prompt always yields the same completion, which makes assertions stable.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from typing import AsyncIterator

from ..types import FinishReason, GenerationRequest, TokenChunk
from .base import Backend, BackendStats, HealthStatus

_VOCAB = (
    "the model serves tokens through a paged attention kernel while the scheduler "
    "admits new sequences into the running batch every step and reuses cached "
    "prefix blocks so that time to first token stays low even under heavy load"
).split()


class MockBackend(Backend):
    """Deterministic in-process generator that mimics prefill + decode timing."""

    name = "mock"

    def __init__(self, config) -> None:
        super().__init__(config)
        self._aborted: set[str] = set()
        self._active: set[str] = set()
        self._prompt_tokens = 0
        self._generation_tokens = 0
        self._completed = 0

    async def start(self) -> None:
        # Mimic engine warmup so benchmarks that time startup see something sane.
        await asyncio.sleep(0)
        self._started = True

    async def stop(self) -> None:
        self._aborted.clear()
        self._active.clear()
        self._started = False

    def _tokens_for(self, request: GenerationRequest) -> list[str]:
        digest = hashlib.sha256(request.prompt.encode("utf-8")).digest()
        seed = request.sampling.seed
        if seed is None:
            seed = int.from_bytes(digest[:8], "big")
        rng = random.Random(seed)
        n = request.sampling.max_tokens
        return [_VOCAB[rng.randrange(len(_VOCAB))] for _ in range(n)]

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[TokenChunk]:
        if not self._started:
            await self.start()
        self._active.add(request.request_id)
        self._prompt_tokens += request.prompt_len
        cfg = self.config.backend

        # Prefill cost scales with prompt length, as it does on a real engine.
        prefill = cfg.mock_prefill_s * max(1.0, request.prompt_len / 512.0)
        try:
            await asyncio.sleep(prefill)
            tokens = self._tokens_for(request)
            stops = request.sampling.stop
            emitted = 0
            finish = FinishReason.LENGTH
            for i, token in enumerate(tokens):
                if request.request_id in self._aborted:
                    yield TokenChunk(
                        request_id=request.request_id,
                        index=emitted,
                        text="",
                        finish_reason=FinishReason.ABORT,
                    )
                    return
                if i:
                    await asyncio.sleep(cfg.mock_decode_s)
                text = token if i == 0 else " " + token
                if any(s and s in text for s in stops):
                    finish = FinishReason.STOP
                    break
                emitted += 1
                self._generation_tokens += 1
                yield TokenChunk(
                    request_id=request.request_id,
                    index=i,
                    text=text,
                    token_id=abs(hash(token)) % 32000,
                    timestamp=time.monotonic(),
                )
            yield TokenChunk(
                request_id=request.request_id,
                index=emitted,
                text="",
                finish_reason=finish,
            )
            self._completed += 1
        finally:
            self._active.discard(request.request_id)
            self._aborted.discard(request.request_id)

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)

    async def health(self) -> HealthStatus:
        return HealthStatus(
            ok=self._started,
            backend=self.name,
            detail="mock engine (no GPU required)",
            model=self.config.model.name,
        )

    async def stats(self) -> BackendStats:
        return BackendStats(
            backend=self.name,
            running=len(self._active),
            total_prompt_tokens=self._prompt_tokens,
            total_generation_tokens=self._generation_tokens,
            extra={"completed_requests": float(self._completed)},
        )
