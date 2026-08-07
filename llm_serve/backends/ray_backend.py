"""Ray Serve backend (client side).

Ray Serve gives autoscaling and multi-replica routing on top of the same vLLM
engine; the deployment itself lives in ``deploy/ray/serve_app.py``. Talking to it
over plain HTTP (rather than a Ray handle) keeps this process free of a Ray
runtime and means the benchmark measures the same network path a real client
would take.

``httpx`` is imported inside :meth:`start`.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from ..types import FinishReason, GenerationRequest, TokenChunk
from .base import Backend, BackendStats, BackendUnavailable, HealthStatus

logger = logging.getLogger("llm_serve.backends.ray")

_FINISH_MAP = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "abort": FinishReason.ABORT,
}


class RayBackend(Backend):
    """HTTP/SSE client for the Ray Serve deployment."""

    name = "ray"

    def __init__(self, config) -> None:
        super().__init__(config)
        self._client: Any = None
        self._aborted: set[str] = set()
        self._prompt_tokens = 0
        self._generation_tokens = 0

    @property
    def base_url(self) -> str:
        return (self.config.backend.endpoint or "").rstrip("/")

    async def start(self) -> None:
        if self._started:
            return
        if not self.base_url:
            raise BackendUnavailable(
                self.name,
                "backend.endpoint is not set",
                "set it to the Ray Serve base URL, e.g. http://localhost:8000",
            )
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailable(
                self.name, f"httpx is not installed ({exc})", "pip install httpx"
            ) from exc
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=self.config.server.request_timeout_s
        )
        logger.info("Ray Serve client pointed at %s", self.base_url)
        self._started = True

    async def stop(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()
        self._started = False

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        s = request.sampling
        return {
            "model": request.model,
            "prompt": request.prompt,
            "stream": True,
            "max_tokens": s.max_tokens,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "top_k": s.top_k,
            "presence_penalty": s.presence_penalty,
            "frequency_penalty": s.frequency_penalty,
            "repetition_penalty": s.repetition_penalty,
            "stop": list(s.stop),
            "seed": s.seed,
            "ignore_eos": s.ignore_eos,
            "request_id": request.request_id,
        }

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[TokenChunk]:
        if not self._started:
            await self.start()
        index = 0
        finish = FinishReason.STOP
        async with self._client.stream(
            "POST", "/v1/completions", json=self._payload(request)
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                if request.request_id in self._aborted:
                    finish = FinishReason.ABORT
                    break
                payload = json.loads(data)
                choice = payload["choices"][0]
                text = choice.get("text") or choice.get("delta", {}).get("content", "")
                if text:
                    self._generation_tokens += 1
                    yield TokenChunk(request_id=request.request_id, index=index, text=text)
                    index += 1
                if choice.get("finish_reason"):
                    finish = _FINISH_MAP.get(choice["finish_reason"], FinishReason.STOP)
        self._aborted.discard(request.request_id)
        self._prompt_tokens += request.prompt_len
        yield TokenChunk(request_id=request.request_id, index=index, text="", finish_reason=finish)

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)

    async def health(self) -> HealthStatus:
        if self._client is None:
            return HealthStatus(False, self.name, "not connected", self.config.model.name)
        try:
            response = await self._client.get("/health")
            ok = response.status_code == 200
            return HealthStatus(ok, self.name, response.text[:200], self.config.model.name)
        except Exception as exc:  # pragma: no cover - requires a live deployment
            return HealthStatus(False, self.name, str(exc), self.config.model.name)

    async def stats(self) -> BackendStats:
        stats = BackendStats(
            backend=self.name,
            total_prompt_tokens=self._prompt_tokens,
            total_generation_tokens=self._generation_tokens,
        )
        if self._client is None:
            return stats
        try:  # pragma: no cover - requires a live deployment
            response = await self._client.get("/stats")
            if response.status_code == 200:
                data = response.json()
                stats.running = int(data.get("running", 0))
                stats.waiting = int(data.get("waiting", 0))
                stats.kv_cache_usage = float(data.get("kv_cache_usage", 0.0))
                stats.prefix_cache_hit_rate = float(data.get("prefix_cache_hit_rate", 0.0))
                stats.extra["replicas"] = float(data.get("replicas", 0))
        except Exception as exc:
            logger.debug("could not read Ray Serve stats: %s", exc)
        return stats
