"""TensorRT-LLM backend.

TensorRT-LLM compiles the model into a hardware-specific engine ahead of time:
kernels are fused, the plugin path is fixed and the batch/sequence dimensions are
baked in. That removes most per-step Python and dispatch overhead, which is where
the latency and cost win over a general-purpose runtime comes from — at the price
of a build step (``scripts/export_trtllm.py``) that must be redone for every GPU
architecture and every change to the max batch/sequence shape.

``tensorrt_llm`` is imported inside :meth:`start`; the engine directory is
validated first so a missing build fails with a clear message rather than a
CUDA error.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator

from ..types import FinishReason, GenerationRequest, TokenChunk
from .base import Backend, BackendStats, BackendUnavailable, HealthStatus

logger = logging.getLogger("llm_serve.backends.trtllm")

ENGINE_CONFIG = "config.json"


def inspect_engine_dir(engine_dir: str) -> dict[str, Any]:
    """Read a built engine's ``config.json`` and report its build shape.

    Pure filesystem work, so the engine layout can be validated (and tested)
    without a GPU present.
    """
    if not os.path.isdir(engine_dir):
        raise FileNotFoundError(f"engine directory {engine_dir!r} does not exist")
    config_path = os.path.join(engine_dir, ENGINE_CONFIG)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"{config_path} not found — build the engine with scripts/export_trtllm.py"
        )
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    build = raw.get("build_config", {})
    pretrained = raw.get("pretrained_config", {})
    engines = sorted(f for f in os.listdir(engine_dir) if f.endswith(".engine"))
    return {
        "engine_dir": engine_dir,
        "engine_files": engines,
        "dtype": pretrained.get("dtype"),
        "quantization": (pretrained.get("quantization") or {}).get("quant_algo"),
        "tensor_parallel_size": (pretrained.get("mapping") or {}).get("tp_size", 1),
        "max_batch_size": build.get("max_batch_size"),
        "max_input_len": build.get("max_input_len"),
        "max_seq_len": build.get("max_seq_len") or build.get("max_output_len"),
        "paged_kv_cache": (build.get("plugin_config") or {}).get("paged_kv_cache", True),
    }


class TensorRTLLMBackend(Backend):
    """Serves a prebuilt TensorRT-LLM engine through its async LLM API."""

    name = "trtllm"

    def __init__(self, config) -> None:
        super().__init__(config)
        self._llm: Any = None
        self._engine_info: dict[str, Any] = {}
        self._aborted: set[str] = set()
        self._prompt_tokens = 0
        self._generation_tokens = 0

    async def start(self) -> None:
        if self._started:
            return
        engine_dir = self.config.backend.engine_dir
        if not engine_dir:
            raise BackendUnavailable(
                self.name,
                "backend.engine_dir is not set",
                "point it at the directory produced by scripts/export_trtllm.py",
            )
        try:
            self._engine_info = inspect_engine_dir(engine_dir)
        except FileNotFoundError as exc:
            raise BackendUnavailable(self.name, str(exc), "run scripts/export_trtllm.py") from exc

        try:
            from tensorrt_llm import LLM  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailable(
                self.name,
                f"tensorrt_llm is not installed ({exc})",
                "use the NGC container nvcr.io/nvidia/tritonserver:*-trtllm-python-py3",
            ) from exc

        logger.info(
            "loading TensorRT-LLM engine from %s (max_batch_size=%s, max_seq_len=%s)",
            engine_dir,
            self._engine_info.get("max_batch_size"),
            self._engine_info.get("max_seq_len"),
        )
        self._llm = LLM(model=engine_dir, tokenizer=self.config.model.tokenizer_id)
        self._started = True

    async def stop(self) -> None:
        llm, self._llm = self._llm, None
        if llm is not None:
            shutdown = getattr(llm, "shutdown", None)
            if callable(shutdown):
                shutdown()
        self._started = False

    def _sampling_params(self, request: GenerationRequest):
        from tensorrt_llm import SamplingParams as TRTSamplingParams  # noqa: PLC0415

        s = request.sampling
        return TRTSamplingParams(
            max_tokens=s.max_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
            top_k=None if s.top_k == -1 else s.top_k,
            presence_penalty=s.presence_penalty,
            frequency_penalty=s.frequency_penalty,
            repetition_penalty=s.repetition_penalty,
            stop=list(s.stop) or None,
            seed=s.seed,
        )

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[TokenChunk]:
        if not self._started:
            await self.start()
        params = self._sampling_params(request)
        previous = ""
        index = 0
        finish = FinishReason.LENGTH
        async for output in self._llm.generate_async(request.prompt, params, streaming=True):
            if request.request_id in self._aborted:
                finish = FinishReason.ABORT
                break
            completion = output.outputs[0]
            delta = completion.text[len(previous) :]
            previous = completion.text
            if delta:
                self._generation_tokens += 1
                yield TokenChunk(request_id=request.request_id, index=index, text=delta)
                index += 1
            if getattr(completion, "finish_reason", None) == "stop":
                finish = FinishReason.STOP
        self._aborted.discard(request.request_id)
        self._prompt_tokens += request.prompt_len
        yield TokenChunk(request_id=request.request_id, index=index, text="", finish_reason=finish)

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)

    async def health(self) -> HealthStatus:
        return HealthStatus(
            ok=self._started and self._llm is not None,
            backend=self.name,
            detail=f"engine {self._engine_info.get('engine_files') or 'not loaded'}",
            model=self.config.model.name,
        )

    async def stats(self) -> BackendStats:
        return BackendStats(
            backend=self.name,
            total_prompt_tokens=self._prompt_tokens,
            total_generation_tokens=self._generation_tokens,
            extra={
                "max_batch_size": float(self._engine_info.get("max_batch_size") or 0),
                "tensor_parallel_size": float(self._engine_info.get("tensor_parallel_size") or 1),
            },
        )
