"""NVIDIA Triton Inference Server backend (client side).

Triton hosts the model through the Python backend in
``deploy/triton/model_repository/mistral7b`` — that model script wraps the same
vLLM engine and is declared *decoupled*, meaning one request may produce many
responses, which is how Triton expresses token streaming.

This class is the client: it opens a gRPC bidirectional stream, pushes an
inference request, and converts the response stream into :class:`TokenChunk`
objects. ``tritonclient`` is imported inside :meth:`start`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
from typing import Any, AsyncIterator

from ..types import FinishReason, GenerationRequest, TokenChunk
from .base import Backend, BackendStats, BackendUnavailable, HealthStatus

logger = logging.getLogger("llm_serve.backends.triton")

DEFAULT_MODEL_NAME = "mistral7b"


class TritonBackend(Backend):
    """Streaming gRPC client for a Triton-hosted vLLM model."""

    name = "triton"

    def __init__(self, config) -> None:
        super().__init__(config)
        self._client: Any = None
        self._grpc: Any = None
        self._model_name = DEFAULT_MODEL_NAME
        self._aborted: set[str] = set()
        self._prompt_tokens = 0
        self._generation_tokens = 0

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._started:
            return
        endpoint = self.config.backend.endpoint
        if not endpoint:
            raise BackendUnavailable(
                self.name, "backend.endpoint is not set", "set it to the Triton gRPC host:port"
            )
        try:
            import tritonclient.grpc as grpcclient  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailable(
                self.name,
                f"tritonclient is not installed ({exc})",
                "pip install 'tritonclient[all]'",
            ) from exc
        self._grpc = grpcclient
        self._client = grpcclient.InferenceServerClient(url=endpoint, verbose=False)
        if not self._client.is_server_live():
            raise BackendUnavailable(
                self.name, f"Triton at {endpoint} is not live", "check `docker compose ps triton`"
            )
        logger.info("connected to Triton at %s (model=%s)", endpoint, self._model_name)
        self._started = True

    async def stop(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # pragma: no cover - network teardown
                logger.debug("error closing Triton client: %s", exc)
        self._started = False

    # ---------------------------------------------------------------- inputs

    def _build_inputs(self, request: GenerationRequest) -> list[Any]:
        import numpy as np  # noqa: PLC0415

        grpcclient = self._grpc
        sampling = {
            "max_tokens": request.sampling.max_tokens,
            "temperature": request.sampling.temperature,
            "top_p": request.sampling.top_p,
            "top_k": request.sampling.top_k,
            "presence_penalty": request.sampling.presence_penalty,
            "frequency_penalty": request.sampling.frequency_penalty,
            "repetition_penalty": request.sampling.repetition_penalty,
            "stop": list(request.sampling.stop),
            "seed": request.sampling.seed,
            "ignore_eos": request.sampling.ignore_eos,
        }
        payloads = {
            "text_input": np.array([request.prompt.encode("utf-8")], dtype=object),
            "sampling_parameters": np.array(
                [json.dumps(sampling).encode("utf-8")], dtype=object
            ),
            "stream": np.array([True], dtype=bool),
        }
        inputs = []
        for name, value in payloads.items():
            dtype = "BOOL" if value.dtype == bool else "BYTES"
            tensor = grpcclient.InferInput(name, [1], dtype)
            tensor.set_data_from_numpy(value)
            inputs.append(tensor)
        return inputs

    # -------------------------------------------------------------- generate

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[TokenChunk]:
        if not self._started:
            await self.start()

        results: "queue.Queue[Any]" = queue.Queue()
        loop = asyncio.get_running_loop()

        def _callback(result: Any, error: Any) -> None:
            results.put(error if error is not None else result)

        self._client.start_stream(callback=_callback)
        try:
            self._client.async_stream_infer(
                model_name=self._model_name,
                inputs=self._build_inputs(request),
                outputs=[self._grpc.InferRequestedOutput("text_output")],
                request_id=request.request_id,
                enable_empty_final_response=True,
            )
            index = 0
            finish = FinishReason.STOP
            while True:
                item = await loop.run_in_executor(None, results.get)
                if isinstance(item, Exception):
                    raise item
                if request.request_id in self._aborted:
                    finish = FinishReason.ABORT
                    break
                text = self._decode(item)
                if text:
                    self._generation_tokens += 1
                    yield TokenChunk(request_id=request.request_id, index=index, text=text)
                    index += 1
                if self._is_final(item):
                    break
            self._prompt_tokens += request.prompt_len
            yield TokenChunk(
                request_id=request.request_id, index=index, text="", finish_reason=finish
            )
        finally:
            self._aborted.discard(request.request_id)
            try:
                self._client.stop_stream()
            except Exception as exc:  # pragma: no cover - network teardown
                logger.debug("error stopping Triton stream: %s", exc)

    @staticmethod
    def _decode(result: Any) -> str:
        output = result.as_numpy("text_output")
        if output is None or len(output) == 0:
            return ""
        value = output[0]
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    @staticmethod
    def _is_final(result: Any) -> bool:
        response = result.get_response()
        params = getattr(response, "parameters", {}) or {}
        flag = params.get("triton_final_response")
        if flag is None:
            return False
        return bool(getattr(flag, "bool_param", flag))

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)

    # ---------------------------------------------------------------- status

    async def health(self) -> HealthStatus:
        if self._client is None:
            return HealthStatus(False, self.name, "not connected", self.config.model.name)
        try:
            ready = self._client.is_model_ready(self._model_name)
            return HealthStatus(
                ready,
                self.name,
                "model ready" if ready else f"model {self._model_name} not ready",
                self.config.model.name,
            )
        except Exception as exc:  # pragma: no cover - requires a live server
            return HealthStatus(False, self.name, str(exc), self.config.model.name)

    async def stats(self) -> BackendStats:
        return BackendStats(
            backend=self.name,
            total_prompt_tokens=self._prompt_tokens,
            total_generation_tokens=self._generation_tokens,
        )
