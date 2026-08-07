"""OpenAI-compatible request parsing and response building.

Plain dicts rather than pydantic models: these functions are pure, so the exact
wire shape can be asserted in unit tests without standing up a server, and the
same builders serve both the FastAPI app and the Triton/Ray front ends.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from ..types import FinishReason, GenerationRequest, GenerationResult, SamplingParams, ValidationError

OBJECT_COMPLETION = "text_completion"
OBJECT_COMPLETION_CHUNK = "text_completion"
OBJECT_CHAT_COMPLETION = "chat.completion"
OBJECT_CHAT_CHUNK = "chat.completion.chunk"

_VALID_ROLES = ("system", "user", "assistant", "tool")


def new_request_id(prefix: str = "cmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _require(payload: dict[str, Any], key: str) -> Any:
    if key not in payload or payload[key] is None:
        raise ValidationError(f"missing required field {key!r}")
    return payload[key]


def parse_completion_request(payload: dict[str, Any], default_model: str) -> GenerationRequest:
    """``POST /v1/completions`` body -> :class:`GenerationRequest`."""
    if not isinstance(payload, dict):
        raise ValidationError("request body must be a JSON object")
    prompt = _require(payload, "prompt")
    if isinstance(prompt, list):
        if not prompt:
            raise ValidationError("prompt list must not be empty")
        if not all(isinstance(p, str) for p in prompt):
            raise ValidationError("token-id prompts are not supported; send a string")
        if len(prompt) > 1:
            raise ValidationError("batched prompts are not supported; send one prompt per request")
        prompt = prompt[0]
    if not isinstance(prompt, str):
        raise ValidationError("prompt must be a string")
    sampling = SamplingParams.from_dict(payload)
    if sampling.n > 1:
        raise ValidationError("n > 1 is not supported")
    return GenerationRequest(
        prompt=prompt,
        sampling=sampling,
        request_id=new_request_id(),
        model=payload.get("model") or default_model,
        stream=bool(payload.get("stream", False)),
        metadata={"endpoint": "completions", "user": payload.get("user")},
    )


def parse_chat_request(
    payload: dict[str, Any], default_model: str, render_template
) -> GenerationRequest:
    """``POST /v1/chat/completions`` body -> :class:`GenerationRequest`.

    ``render_template`` turns the message list into the model's prompt format;
    for Mistral that is the ``[INST] … [/INST]`` convention.
    """
    messages = _require(payload, "messages")
    if not isinstance(messages, list) or not messages:
        raise ValidationError("messages must be a non-empty list")
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValidationError(f"messages[{i}] must be an object")
        role = message.get("role")
        if role not in _VALID_ROLES:
            raise ValidationError(
                f"messages[{i}].role must be one of {', '.join(_VALID_ROLES)}, got {role!r}"
            )
        if not isinstance(message.get("content", ""), str):
            raise ValidationError(f"messages[{i}].content must be a string")
    prompt = render_template(messages)
    sampling = SamplingParams.from_dict(payload)
    if sampling.n > 1:
        raise ValidationError("n > 1 is not supported")
    return GenerationRequest(
        prompt=prompt,
        sampling=sampling,
        request_id=new_request_id("chatcmpl"),
        model=payload.get("model") or default_model,
        stream=bool(payload.get("stream", False)),
        metadata={"endpoint": "chat", "user": payload.get("user")},
    )


def usage(result: GenerationResult) -> dict[str, int]:
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
    }


def completion_response(
    request: GenerationRequest, result: GenerationResult, created: int | None = None
) -> dict[str, Any]:
    return {
        "id": request.request_id,
        "object": OBJECT_COMPLETION,
        "created": created if created is not None else int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": result.text,
                "logprobs": None,
                "finish_reason": _finish(result.finish_reason),
            }
        ],
        "usage": usage(result),
    }


def chat_completion_response(
    request: GenerationRequest, result: GenerationResult, created: int | None = None
) -> dict[str, Any]:
    return {
        "id": request.request_id,
        "object": OBJECT_CHAT_COMPLETION,
        "created": created if created is not None else int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "logprobs": None,
                "finish_reason": _finish(result.finish_reason),
            }
        ],
        "usage": usage(result),
    }


def completion_chunk(
    request: GenerationRequest,
    text: str,
    finish_reason: FinishReason | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    return {
        "id": request.request_id,
        "object": OBJECT_COMPLETION_CHUNK,
        "created": created if created is not None else int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": _finish(finish_reason),
            }
        ],
    }


def chat_chunk(
    request: GenerationRequest,
    text: str,
    finish_reason: FinishReason | None = None,
    created: int | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """One ``chat.completion.chunk``.

    The first chunk of a stream carries ``delta.role`` and no content, matching
    what OpenAI clients expect before any text arrives.
    """
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if text:
        delta["content"] = text
    return {
        "id": request.request_id,
        "object": OBJECT_CHAT_CHUNK,
        "created": created if created is not None else int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": _finish(finish_reason),
            }
        ],
    }


def models_response(model_names: list[str], owner: str = "llm-serve") -> dict[str, Any]:
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": created, "owned_by": owner}
            for name in model_names
        ],
    }


def error_response(
    message: str, err_type: str = "invalid_request_error", code: str | None = None
) -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "param": None, "code": code}}


def _finish(reason: FinishReason | None) -> str | None:
    if reason is None:
        return None
    # `abort`/`error` are not OpenAI finish reasons; clients treat them as a
    # truncated stream, which is exactly what happened.
    if reason in (FinishReason.ABORT, FinishReason.ERROR):
        return "length"
    return reason.value
