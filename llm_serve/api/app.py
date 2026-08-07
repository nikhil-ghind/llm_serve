"""OpenAI-compatible FastAPI application.

One app fronts whichever backend the config selects, so a client can be pointed
at vLLM, Triton, Ray Serve or TensorRT-LLM without changing a line — which is
what makes the cross-backend benchmark a fair comparison.

FastAPI is imported lazily inside :func:`create_app` so that importing this
module (for the handler functions, or during a CPU-only test run) does not
require the web stack to be installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from ..backends.base import Backend, BackendUnavailable, create_backend
from ..config import Config
from ..metrics.registry import ServingMetrics
from ..types import FinishReason, GenerationRequest, ResultAccumulator, ValidationError
from . import openai_schemas as schemas
from .chat_template import render_mistral_prompt
from .sse import SSE_HEADERS, SSE_MEDIA_TYPE, done_event, format_json_sse

logger = logging.getLogger("llm_serve.api")


class ServingContext:
    """Holds the backend and metrics for the lifetime of the process."""

    def __init__(self, config: Config, backend: Backend | None = None) -> None:
        self.config = config
        self.backend = backend or create_backend(config)
        self.metrics = ServingMetrics(config.metrics.namespace, config.metrics)
        self.semaphore = asyncio.Semaphore(config.server.max_concurrent_requests)
        self.started_at = time.time()

    async def startup(self) -> None:
        logger.info("starting backend %s for model %s", self.backend.name, self.config.model.name)
        await self.backend.start()

    async def shutdown(self) -> None:
        await self.backend.stop()

    async def refresh_gauges(self) -> None:
        stats = await self.backend.stats()
        self.metrics.observe_stats(stats, self.config.model.name)


async def stream_completion(
    ctx: ServingContext, request: GenerationRequest, chat: bool
) -> AsyncIterator[str]:
    """Yield SSE frames for one streaming request."""
    created = int(time.time())
    accumulator = ResultAccumulator(request)
    if chat:
        # OpenAI clients expect a role-only delta before any content.
        yield format_json_sse(schemas.chat_chunk(request, "", created=created, role="assistant"))
    try:
        async for chunk in ctx.backend.generate_stream(request):
            accumulator.add(chunk)
            if chunk.text:
                payload = (
                    schemas.chat_chunk(request, chunk.text, created=created)
                    if chat
                    else schemas.completion_chunk(request, chunk.text, created=created)
                )
                yield format_json_sse(payload)
            if chunk.is_final:
                payload = (
                    schemas.chat_chunk(request, "", chunk.finish_reason, created=created)
                    if chat
                    else schemas.completion_chunk(request, "", chunk.finish_reason, created=created)
                )
                yield format_json_sse(payload)
    except asyncio.CancelledError:
        # Client hung up: stop the engine work rather than paying to finish it.
        await ctx.backend.abort(request.request_id)
        raise
    except Exception as exc:  # pragma: no cover - defensive, surfaced to client
        logger.exception("generation failed for %s", request.request_id)
        yield format_json_sse(schemas.error_response(str(exc), "server_error"))
        ctx.metrics.observe_result(
            accumulator.finish(), ctx.backend.name, ctx.config.model.name, status="error"
        )
        yield done_event()
        return
    result = accumulator.finish()
    ctx.metrics.observe_result(result, ctx.backend.name, ctx.config.model.name)
    yield done_event()


async def run_completion(ctx: ServingContext, request: GenerationRequest):
    """Collect a full (non-streaming) generation."""
    accumulator = ResultAccumulator(request)
    async for chunk in ctx.backend.generate_stream(request):
        accumulator.add(chunk)
    result = accumulator.finish()
    ctx.metrics.observe_result(result, ctx.backend.name, ctx.config.model.name)
    return result


def create_app(config: Config | None = None, backend: Backend | None = None):
    """Build the ASGI application. Imports FastAPI on demand."""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise BackendUnavailable(
            "api",
            f"FastAPI is not installed ({exc})",
            "pip install 'fastapi' 'uvicorn[standard]'",
        ) from exc

    cfg = config or Config()
    ctx = ServingContext(cfg, backend)

    app = FastAPI(
        title="llm_serve",
        version="0.1.0",
        description="OpenAI-compatible serving for Mistral 7B (QLoRA) across vLLM, "
        "Triton, Ray Serve and TensorRT-LLM.",
    )
    app.state.ctx = ctx

    @app.on_event("startup")
    async def _startup() -> None:
        await ctx.startup()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await ctx.shutdown()

    def _authorize(request: "Request") -> None:
        if not cfg.server.api_key:
            return
        header = request.headers.get("authorization", "")
        token = header[len("Bearer ") :] if header.startswith("Bearer ") else ""
        if token != cfg.server.api_key:
            raise HTTPException(status_code=401, detail="invalid API key")

    async def _body(request: "Request") -> dict[str, Any]:
        raw = await request.body()
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"malformed JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        return payload

    async def _handle(request: "Request", chat: bool):
        _authorize(request)
        payload = await _body(request)
        try:
            if chat:
                gen_request = schemas.parse_chat_request(
                    payload, cfg.server.served_model_name, render_mistral_prompt
                )
            else:
                gen_request = schemas.parse_completion_request(
                    payload, cfg.server.served_model_name
                )
        except ValidationError as exc:
            return JSONResponse(status_code=400, content=schemas.error_response(str(exc)))

        if gen_request.stream:
            async def body() -> AsyncIterator[str]:
                async with ctx.semaphore:
                    async for frame in stream_completion(ctx, gen_request, chat):
                        yield frame

            return StreamingResponse(body(), media_type=SSE_MEDIA_TYPE, headers=dict(SSE_HEADERS))

        async with ctx.semaphore:
            try:
                result = await asyncio.wait_for(
                    run_completion(ctx, gen_request), timeout=cfg.server.request_timeout_s
                )
            except asyncio.TimeoutError:
                await ctx.backend.abort(gen_request.request_id)
                return JSONResponse(
                    status_code=504,
                    content=schemas.error_response("request timed out", "timeout_error"),
                )
        builder = schemas.chat_completion_response if chat else schemas.completion_response
        return JSONResponse(content=builder(gen_request, result))

    @app.post("/v1/completions")
    async def completions(request: "Request"):
        return await _handle(request, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: "Request"):
        return await _handle(request, chat=True)

    @app.get("/v1/models")
    async def models():
        return JSONResponse(content=schemas.models_response([cfg.server.served_model_name]))

    @app.get("/health")
    async def health():
        status = await ctx.backend.health()
        code = 200 if status.ok else 503
        return JSONResponse(
            status_code=code,
            content={
                "status": "ok" if status.ok else "unavailable",
                "backend": status.backend,
                "model": status.model,
                "detail": status.detail,
                "uptime_s": time.time() - ctx.started_at,
            },
        )

    @app.get("/stats")
    async def stats():
        return JSONResponse(content=(await ctx.backend.stats()).to_dict())

    @app.get("/metrics")
    async def metrics():
        if not cfg.metrics.enabled:
            return PlainTextResponse("metrics are disabled\n", status_code=404)
        await ctx.refresh_gauges()
        return PlainTextResponse(ctx.metrics.render(), media_type="text/plain; version=0.0.4")

    return app


__all__ = ["ServingContext", "create_app", "run_completion", "stream_completion", "FinishReason"]
