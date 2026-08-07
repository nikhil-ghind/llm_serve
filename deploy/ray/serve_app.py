"""Ray Serve deployment fronting the vLLM engine.

Ray Serve adds what a single-process server lacks: several replicas behind one
route, autoscaling on queue depth, and independent rollout of the engine from the
HTTP layer. Each replica owns its own vLLM engine and therefore its own KV cache,
so ``max_ongoing_requests`` should stay well above one — Serve must let requests
queue *inside* the replica for continuous batching to have anything to batch.

Run with:
    serve run deploy.ray.serve_app:app
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from ray import serve

api = FastAPI(title="llm_serve on Ray Serve")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@serve.deployment(
    name="mistral7b",
    autoscaling_config={
        "min_replicas": int(_env("LLM_SERVE_MIN_REPLICAS", "1")),
        "max_replicas": int(_env("LLM_SERVE_MAX_REPLICAS", "4")),
        # Scale on queue depth per replica: one replica saturates long before
        # its request count reaches max_ongoing_requests.
        "target_ongoing_requests": int(_env("LLM_SERVE_TARGET_ONGOING", "32")),
        "upscale_delay_s": 10.0,
        "downscale_delay_s": 120.0,
    },
    max_ongoing_requests=int(_env("LLM_SERVE_MAX_ONGOING", "256")),
    ray_actor_options={"num_gpus": float(_env("LLM_SERVE_NUM_GPUS", "1"))},
    health_check_period_s=10,
)
@serve.ingress(api)
class MistralDeployment:
    def __init__(
        self,
        base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        lora_adapter: str | None = None,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        self.base_model = base_model
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=base_model,
                dtype="bfloat16",
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                enable_lora=bool(lora_adapter),
                disable_log_requests=True,
            )
        )
        self.lora_request = None
        if lora_adapter:
            from vllm.lora.request import LoRARequest

            self.lora_request = LoRARequest("qlora-adapter", 1, lora_adapter)

    # ------------------------------------------------------------------ HTTP

    @api.post("/v1/completions")
    async def completions(self, request: Request):
        payload = await request.json()
        if payload.get("stream"):
            return StreamingResponse(
                self._stream(payload), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        text, finish, prompt_tokens, completion_tokens = await self._collect(payload)
        return JSONResponse(
            {
                "id": payload.get("request_id", f"cmpl-{int(time.time() * 1e6)}"),
                "object": "text_completion",
                "created": int(time.time()),
                "model": payload.get("model", self.base_model),
                "choices": [{"index": 0, "text": text, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    @api.get("/health")
    async def health(self):
        await self.engine.check_health()
        return {"status": "ok", "model": self.base_model}

    @api.get("/stats")
    async def stats(self):
        inner = getattr(self.engine, "engine", None)
        scheduler = getattr(inner, "scheduler", None)
        if isinstance(scheduler, list):
            scheduler = scheduler[0] if scheduler else None
        running = len(getattr(scheduler, "running", ()))
        waiting = len(getattr(scheduler, "waiting", ()))
        usage = 0.0
        block_manager = getattr(scheduler, "block_manager", None)
        if block_manager is not None:
            total = getattr(block_manager, "num_total_gpu_blocks", 0)
            if total:
                usage = (total - block_manager.get_num_free_gpu_blocks()) / total
        return {"running": running, "waiting": waiting, "kv_cache_usage": usage, "replicas": 1}

    # ------------------------------------------------------------- internals

    def _sampling(self, payload: dict[str, Any]):
        from vllm import SamplingParams

        return SamplingParams(
            max_tokens=int(payload.get("max_tokens", 128)),
            temperature=float(payload.get("temperature", 0.7)),
            top_p=float(payload.get("top_p", 1.0)),
            top_k=int(payload.get("top_k", -1)),
            presence_penalty=float(payload.get("presence_penalty", 0.0)),
            frequency_penalty=float(payload.get("frequency_penalty", 0.0)),
            repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
            stop=payload.get("stop") or None,
            seed=payload.get("seed"),
            ignore_eos=bool(payload.get("ignore_eos", False)),
        )

    def _kwargs(self) -> dict[str, Any]:
        return {"lora_request": self.lora_request} if self.lora_request else {}

    async def _stream(self, payload: dict[str, Any]) -> AsyncGenerator[str, None]:
        request_id = payload.get("request_id", f"cmpl-{int(time.time() * 1e6)}")
        created = int(time.time())
        previous = ""
        async for output in self.engine.generate(
            payload["prompt"], self._sampling(payload), request_id, **self._kwargs()
        ):
            completion = output.outputs[0]
            delta = completion.text[len(previous):]
            previous = completion.text
            if delta:
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": payload.get("model", self.base_model),
                    "choices": [{"index": 0, "text": delta, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
            if completion.finish_reason:
                final = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": payload.get("model", self.base_model),
                    "choices": [
                        {"index": 0, "text": "", "finish_reason": completion.finish_reason}
                    ],
                }
                yield f"data: {json.dumps(final, separators=(',', ':'))}\n\n"
        yield "data: [DONE]\n\n"

    async def _collect(self, payload: dict[str, Any]):
        request_id = payload.get("request_id", f"cmpl-{int(time.time() * 1e6)}")
        final = None
        async for output in self.engine.generate(
            payload["prompt"], self._sampling(payload), request_id, **self._kwargs()
        ):
            final = output
        completion = final.outputs[0]
        return (
            completion.text,
            completion.finish_reason or "stop",
            len(final.prompt_token_ids or ()),
            len(completion.token_ids or ()),
        )


app = MistralDeployment.bind(
    base_model=_env("LLM_SERVE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
    lora_adapter=os.environ.get("LLM_SERVE_LORA_ADAPTER") or None,
    max_model_len=int(_env("LLM_SERVE_MAX_MODEL_LEN", "8192")),
    tensor_parallel_size=int(_env("LLM_SERVE_TP_SIZE", "1")),
    gpu_memory_utilization=float(_env("LLM_SERVE_GPU_MEM_UTIL", "0.90")),
)
