"""vLLM backend.

Runs ``AsyncLLMEngine`` in-process: vLLM does its own continuous batching, paged
KV management and (with ``enable_prefix_caching``) automatic prefix reuse, so
this adapter's job is translation — our :class:`GenerationRequest` in, our
:class:`TokenChunk` stream out — plus loading the QLoRA adapter.

``import vllm`` pulls in torch and CUDA, so it happens inside :meth:`start`.
Importing this module on a CPU-only laptop is free.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from ..types import FinishReason, GenerationRequest, TokenChunk
from .base import Backend, BackendStats, BackendUnavailable, HealthStatus

logger = logging.getLogger("llm_serve.backends.vllm")

_FINISH_MAP = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "abort": FinishReason.ABORT,
}


class VLLMBackend(Backend):
    """In-process vLLM engine."""

    name = "vllm"

    def __init__(self, config) -> None:
        super().__init__(config)
        self._engine: Any = None
        self._lora_request: Any = None
        self._prompt_tokens = 0
        self._generation_tokens = 0

    # ------------------------------------------------------------- lifecycle

    def _engine_args(self) -> dict[str, Any]:
        model = self.config.model
        backend = self.config.backend
        scheduler = self.config.scheduler
        args: dict[str, Any] = {
            "model": model.base_model,
            "tokenizer": model.tokenizer_id,
            "dtype": model.dtype,
            "max_model_len": model.max_model_len,
            "trust_remote_code": model.trust_remote_code,
            "tensor_parallel_size": backend.tensor_parallel_size,
            "gpu_memory_utilization": backend.gpu_memory_utilization,
            "enable_prefix_caching": backend.enable_prefix_caching,
            "enforce_eager": backend.enforce_eager,
            "swap_space": backend.swap_space_gb,
            "max_num_seqs": scheduler.max_num_seqs,
            "max_num_batched_tokens": scheduler.max_num_batched_tokens,
            "block_size": scheduler.block_size,
            "enable_chunked_prefill": scheduler.enable_chunked_prefill,
            "disable_log_requests": True,
        }
        if model.quantization:
            args["quantization"] = model.quantization
        if model.lora_adapter:
            # QLoRA adapters are served as vLLM LoRA requests rather than merged
            # into the base weights, so several adapters can share one engine.
            args["enable_lora"] = True
            args["max_lora_rank"] = 64
            args["max_loras"] = 1
        return args

    async def start(self) -> None:
        if self._started:
            return
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailable(
                self.name,
                f"vLLM is not installed ({exc})",
                "pip install vllm  (requires an NVIDIA GPU with CUDA 12.x)",
            ) from exc

        args = self._engine_args()
        logger.info("initializing vLLM engine: %s", args["model"])
        self._engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**args))

        if self.config.model.lora_adapter:
            from vllm.lora.request import LoRARequest  # noqa: PLC0415

            self._lora_request = LoRARequest(
                lora_name="qlora-adapter",
                lora_int_id=1,
                lora_path=self.config.model.lora_adapter,
            )
            logger.info("loaded QLoRA adapter from %s", self.config.model.lora_adapter)
        self._started = True

    async def stop(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            shutdown = getattr(engine, "shutdown_background_loop", None)
            if callable(shutdown):
                shutdown()
        self._started = False

    # -------------------------------------------------------------- generate

    def _sampling_params(self, request: GenerationRequest):
        from vllm import SamplingParams as VLLMSamplingParams  # noqa: PLC0415

        s = request.sampling
        return VLLMSamplingParams(
            n=s.n,
            max_tokens=s.max_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
            top_k=s.top_k,
            presence_penalty=s.presence_penalty,
            frequency_penalty=s.frequency_penalty,
            repetition_penalty=s.repetition_penalty,
            stop=list(s.stop) or None,
            seed=s.seed,
            ignore_eos=s.ignore_eos,
        )

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[TokenChunk]:
        if not self._started:
            await self.start()
        params = self._sampling_params(request)
        kwargs: dict[str, Any] = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request

        previous = ""
        index = 0
        finish = FinishReason.LENGTH
        # vLLM yields the cumulative text so far; the API streams deltas.
        async for output in self._engine.generate(
            request.prompt, params, request.request_id, **kwargs
        ):
            completion = output.outputs[0]
            delta = completion.text[len(previous) :]
            previous = completion.text
            if delta:
                yield TokenChunk(
                    request_id=request.request_id,
                    index=index,
                    text=delta,
                    token_id=completion.token_ids[-1] if completion.token_ids else None,
                )
                index += 1
            if completion.finish_reason:
                finish = _FINISH_MAP.get(completion.finish_reason, FinishReason.STOP)
                self._prompt_tokens += len(output.prompt_token_ids or ())
                self._generation_tokens += len(completion.token_ids or ())
        yield TokenChunk(request_id=request.request_id, index=index, text="", finish_reason=finish)

    async def abort(self, request_id: str) -> None:
        if self._engine is not None:
            await self._engine.abort(request_id)

    # ---------------------------------------------------------------- status

    async def health(self) -> HealthStatus:
        if not self._started or self._engine is None:
            return HealthStatus(False, self.name, "engine not started", self.config.model.name)
        try:
            await self._engine.check_health()
            return HealthStatus(True, self.name, "engine healthy", self.config.model.name)
        except Exception as exc:  # pragma: no cover - requires a live engine
            return HealthStatus(False, self.name, f"engine unhealthy: {exc}", self.config.model.name)

    async def stats(self) -> BackendStats:
        stats = BackendStats(
            backend=self.name,
            total_prompt_tokens=self._prompt_tokens,
            total_generation_tokens=self._generation_tokens,
        )
        engine = self._engine
        if engine is None:
            return stats
        # Read the engine's own scheduler state when the version exposes it;
        # different vLLM releases move this around, so it is best-effort.
        try:  # pragma: no cover - requires a live engine
            inner = getattr(engine, "engine", None)
            scheduler = getattr(inner, "scheduler", None)
            if isinstance(scheduler, list):
                scheduler = scheduler[0] if scheduler else None
            if scheduler is not None:
                stats.running = len(getattr(scheduler, "running", ()))
                stats.waiting = len(getattr(scheduler, "waiting", ()))
                stats.swapped = len(getattr(scheduler, "swapped", ()))
                block_manager = getattr(scheduler, "block_manager", None)
                if block_manager is not None:
                    total = getattr(block_manager, "num_total_gpu_blocks", 0)
                    free = block_manager.get_num_free_gpu_blocks()
                    if total:
                        stats.kv_cache_usage = (total - free) / total
        except Exception as exc:
            logger.debug("could not read vLLM scheduler stats: %s", exc)
        return stats
